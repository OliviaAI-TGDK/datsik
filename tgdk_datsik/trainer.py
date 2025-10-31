# tgdk_datsik.py
"""
TGDK DATSIK+ — Direct Adaptive Trainer for Symbolic Instructional Knowledge (extended)
- Deterministic seeding
- Checkpoint save/load (model + optim + sched + RNG states)
- Toggle gradient_checkpointing on/off for submodules
- AMP (native torch.cuda.amp) optional
- Single-node multi-GPU distributed via torch.distributed (init from script)
- Simple telemetry callbacks (user-provided)
- Deterministic dataloader/collate options
- No EOS injection, no HF/TRL dependencies

Example usage:
from tgdk_datsik import DATSIKTrainer, default_seed
trainer = DATSIKTrainer(..., config={"deterministic": True, "amp": True, "grad_ckpt": True})
trainer.fit(epochs=3, batch_size=8)
"""

import os
import time
import math
import json
import logging
from typing import Optional, Callable, Dict
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# -------------------------
# Defaults & Utilities
# -------------------------
DEFAULT_SAVE_INTERVAL = 1  # epochs
DEFAULT_CHECKPOINT = "datsik_checkpoint.pt"
logger = logging.getLogger("DATSIK")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    logger.addHandler(ch)

def default_seed(seed: int = 1337):
    """Seed everything for reproducibility (best-effort)."""
    import random
    import numpy as np
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic flags (may reduce perf)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed

def save_checkpoint(path: str,
                    model: nn.Module,
                    optimizer,
                    scheduler,
                    scaler: Optional[GradScaler],
                    epoch: int,
                    extra: Optional[dict] = None):
    """Save model + optim + sched + rng states for deterministic resume."""
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict() if optimizer is not None else None,
        "sched_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "python_env": dict(os.environ),
        "extra": extra or {}
    }
    torch.save(state, path)
    logger.info(f"[DATSIK] Checkpoint saved → {path}")

def load_checkpoint(path: str, model: nn.Module, optimizer=None, scheduler=None, scaler=None, map_location=None):
    chk = torch.load(path, map_location=map_location)
    model.load_state_dict(chk["model_state"])
    if optimizer is not None and chk.get("optim_state") is not None:
        optimizer.load_state_dict(chk["optim_state"])
    if scheduler is not None and chk.get("sched_state") is not None:
        scheduler.load_state_dict(chk["sched_state"])
    if scaler is not None and chk.get("scaler_state") is not None:
        scaler.load_state_dict(chk["scaler_state"])
    if chk.get("torch_rng") is not None:
        torch.set_rng_state(chk["torch_rng"])
    if chk.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(chk["cuda_rng"])
    logger.info(f"[DATSIK] Checkpoint loaded → {path}")
    return chk

def enable_gradient_checkpointing(model: nn.Module, enable: bool = True):
    """Try to toggle gradient checkpointing on supported modules."""
    toggled = False
    for m in model.modules():
        if hasattr(m, "gradient_checkpointing_enable") and hasattr(m, "gradient_checkpointing_disable"):
            if enable:
                try:
                    m.gradient_checkpointing_enable()
                    toggled = True
                except Exception:
                    pass
            else:
                try:
                    m.gradient_checkpointing_disable()
                    toggled = True
                except Exception:
                    pass
    logger.info(f"[DATSIK] gradient_checkpointing set to {enable} (toggled={toggled})")
    return toggled

# -------------------------
# Telemetry callback type
# -------------------------
TelemetryCallback = Callable[[Dict], None]

# -------------------------
# DATSIKTrainer
# -------------------------
class DATSIKTrainer:
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        train_dataset,
        val_dataset=None,
        optimizers=None,          # tuple(opt, sched) OR (opt,) OR None
        config: Optional[dict]=None,
        device: Optional[str]=None,
        log_prefix: str="[DATSIK]",
        telemetry_cb: Optional[TelemetryCallback]=None
    ):
        cfg = config or {}
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._opt = optimizers[0] if optimizers else None
        self._sched = optimizers[1] if optimizers and len(optimizers) > 1 else None
        self.log_prefix = log_prefix
        self.telemetry_cb = telemetry_cb
        self.epochs_trained = 0

        # deterministic seed
        if cfg.get("deterministic", False):
            seed = cfg.get("seed", 1337)
            default_seed(seed)
            logger.info(f"{log_prefix} Deterministic mode enabled (seed={seed})")

        # AMP
        self.use_amp = bool(cfg.get("amp", False)) and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.use_amp) if self.use_amp else None
        if self.use_amp:
            logger.info(f"{log_prefix} AMP enabled (native)")

        # Gradient checkpoint toggle
        if "grad_ckpt" in cfg:
            enable_gradient_checkpointing(self.model, cfg["grad_ckpt"])

        # Distributed setup: assume user launched with torch.distributed.launch or torchrun
        self.distributed = False
        if "WORLD_SIZE" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
            self.distributed = True
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.distributed.init_process_group(backend=cfg.get("dist_backend", "nccl"), init_method="env://")
            logger.info(f"{log_prefix} Distributed mode initialized (rank={self.local_rank}, world={os.environ.get('WORLD_SIZE')})")

        # move model to device (if distributed, wrap)
        if self.distributed:
            # move model to current device (LOCAL_RANK required)
            rank_device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
            self.model.to(rank_device)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank] if torch.cuda.is_available() else None)
            self.device = rank_device
        else:
            self.model.to(self.device)

        # deterministic dataloader defaults
        self.num_workers = cfg.get("num_workers", 0)  # deterministic when 0
        self.pin_memory = cfg.get("pin_memory", True) if self.device != "cpu" else False
        self.prefetch_factor = cfg.get("prefetch_factor", 2)

        logger.info(f"{log_prefix} initialized on device={self.device}, distributed={self.distributed}")

    # -------------------------
    # collate
    # -------------------------
    def collate(self, batch):
        texts = [b.get("text", "") for b in batch]
        toks = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.cfg.get("max_length", 2048),
            return_tensors="pt"
        )
        toks = {k: v.to(self.device, non_blocking=True) for k, v in toks.items()}
        toks["labels"] = toks["input_ids"].clone()
        return toks

    # -------------------------
    # step functions
    # -------------------------
    def _train_step(self, batch, grad_accum: int = 1):
        """
        One gradient accumulation step.
        Returns a detached float loss for logging,
        while maintaining a valid autograd graph.
        """

        # remove invalid keys that might break encoder-only models
        batch = {k: v for k, v in batch.items() if k in ["input_ids", "attention_mask", "labels"]}

        # === Forward pass ===
        if self.use_amp:
            with autocast(dtype=torch.float16):
                out = self.model(**batch)
                loss = getattr(out, "loss", out.get("loss") if isinstance(out, dict) else None)
                if loss is None:
                    raise ValueError("Model forward() returned no loss.")
                if not loss.requires_grad:
                    raise RuntimeError("Loss is detached from the graph (no grad_fn).")
                loss = loss / grad_accum
            # === Backward pass with scaling ===
            self.scaler.scale(loss).backward()
        else:
            out = self.model(**batch)
            loss = getattr(out, "loss", out.get("loss") if isinstance(out, dict) else None)
            if loss is None:
                raise ValueError("Model forward() returned no loss.")
            if not loss.requires_grad:
                raise RuntimeError("Loss is detached from the graph (no grad_fn).")
            loss = loss / grad_accum
            loss.backward()

        # === Optional gradient clipping ===
        if getattr(self, "max_grad_norm", None) is not None:
            if self.use_amp and self.scaler is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)


        # Return detached scalar for logging
        return float(loss.detach().item() * grad_accum)

    # -------------------------
    # optimizer step (safe AMP)
    # -------------------------
    def _optim_step(self):
        """
        Performs one optimizer step with optional AMP support,
        gradient clipping, and scheduler update.
        """

        if self.use_amp:
            # Unscale once here (not in _train_step)
            self.scaler.unscale_(self._opt)

            if getattr(self, "max_grad_norm", None) is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self._opt)
            self.scaler.update()
        else:
            if getattr(self, "max_grad_norm", None) is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self._opt.step()

        if self._sched is not None:
            try:
                self._sched.step()
            except Exception:
                pass

        # always clear gradients after each accumulation cycle
        self._opt.zero_grad(set_to_none=True)



    # -------------------------
    # training / evaluation loops
    # -------------------------
    def train_epoch(self, dataloader: DataLoader, epoch:int, grad_accum:int=1, log_interval:int=10):
        self.model.train()
        total_loss = 0.0
        steps = 0
        pbar = tqdm(dataloader, desc=f"{self.log_prefix} Train Ep{epoch}", disable=self.distributed and int(os.environ.get("LOCAL_RANK",0))!=0)
        for step, batch in enumerate(pbar):
            loss = self._train_step(batch, grad_accum)
            total_loss += loss
            steps += 1
            if (step + 1) % grad_accum == 0:
                self._optim_step()
            if step % log_interval == 0:
                stats = {"epoch": epoch, "step": step + 1, "loss": loss}
                self._emit_telemetry(stats)
                pbar.set_postfix({"loss": f"{loss:.4f}"})
        avg = total_loss / max(1, steps)
        return avg

    def evaluate(self, dataloader: DataLoader):
        self.model.eval()
        total_loss = 0.0
        steps = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"{self.log_prefix} Eval", disable=self.distributed and int(os.environ.get("LOCAL_RANK",0))!=0):
                out = self.model(**batch)
                # Support both dict and HF-style ModelOutput
        if isinstance(out, dict):
            loss_val = out.get("loss", None)
        else:
            loss_val = getattr(out, "loss", None)
 
        if loss_val is None:
            raise ValueError("Model forward() returned no loss during evaluation")

        total_loss += float(loss_val.item())

        steps += 1
        return total_loss / max(1, steps)

    # -------------------------
    # telemetry / callbacks
    # -------------------------
    def _emit_telemetry(self, payload: Dict):
        # add default fields
        payload = dict(payload)
        payload["_time"] = time.time()
        try:
            payload["_gpu_mem_gb"] = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        except Exception:
            payload["_gpu_mem_gb"] = None
        # send to user callback if provided
        if callable(self.telemetry_cb):
            try:
                self.telemetry_cb(payload)
            except Exception as e:
                logger.warning(f"{self.log_prefix} telemetry_cb failed: {e}")
        else:
            # default: log
            logger.info(f"{self.log_prefix} telemetry: {json.dumps(payload, default=str)}")

    # -------------------------
    # main fit
    # -------------------------
    def fit(self, epochs: int = 1, batch_size: int = 8, grad_accum: int = 1, resume_from: Optional[str] = None, save_every: int = DEFAULT_SAVE_INTERVAL, checkpoint_path: str = DEFAULT_CHECKPOINT):
        # Build dataloaders (DistributedSampler if needed)
        if self.distributed:
            from torch.utils.data.distributed import DistributedSampler
            train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            train_dl = DataLoader(self.train_dataset, batch_size=batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, pin_memory=self.pin_memory, collate_fn=self.collate,
                                  prefetch_factor=self.prefetch_factor)
            val_dl = DataLoader(self.val_dataset, batch_size=batch_size, sampler=DistributedSampler(self.val_dataset) if self.val_dataset else None,
                                num_workers=self.num_workers, pin_memory=self.pin_memory, collate_fn=self.collate,
                                prefetch_factor=self.prefetch_factor) if self.val_dataset else None
        else:
            # --- Safe DataLoader config ---
            # --- Safe DataLoader configuration (for Windows + CUDA) ---
            num_workers = self.num_workers
            prefetch_factor = self.prefetch_factor if num_workers > 0 else None

            use_pin_memory = self.pin_memory and (not str(self.device).startswith("cuda"))
              
            train_dl = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=use_pin_memory,
                collate_fn=self.collate,
                prefetch_factor=prefetch_factor,
            )

            val_dl = DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=use_pin_memory,
                collate_fn=self.collate,
                prefetch_factor=prefetch_factor,
            ) if self.val_dataset else None

        # optionally resume
        start_epoch = 1
        if resume_from is not None and os.path.exists(resume_from):
            load_checkpoint(resume_from, model=self.model, optimizer=self._opt, scheduler=self._sched, scaler=self.scaler, map_location=self.device)
            start_epoch = int(torch.load(resume_from).get("epoch", 1)) + 1

        for ep in range(start_epoch, epochs + 1):
            if self.distributed:
                train_dl.sampler.set_epoch(ep)

            train_loss = self.train_epoch(train_dl, ep, grad_accum=grad_accum)
            val_loss = None
            if val_dl:
                val_loss = self.evaluate(val_dl)

            # telemetry & logging
            metrics = {"epoch": ep, "train_loss": train_loss, "val_loss": val_loss}
            self._emit_telemetry(metrics)
            logger.info(f"{self.log_prefix} Epoch {ep}/{epochs} -> train={train_loss:.4f}" + (f" val={val_loss:.4f}" if val_loss is not None else ""))

            # checkpoint
            if (ep % save_every) == 0 or ep == epochs:
                save_checkpoint(checkpoint_path, self.model, self._opt, self._sched, self.scaler, ep, extra={"metrics": metrics})
                # if distributed, only rank 0 writes to disk
                if self.distributed:
                    torch.distributed.barrier()

        self.epochs_trained += (epochs - start_epoch + 1)
        logger.info(f"{self.log_prefix} Training complete (epochs_trained={self.epochs_trained})")
        return True

    # -------------------------
    # small helpers
    # -------------------------
    def generate(self, prompt: str, max_new_tokens: int = 128, **kwargs) -> str:
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.cfg.get("max_length", 2048))
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            with autocast() if self.use_amp else torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **kwargs)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

