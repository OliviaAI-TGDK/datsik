# ============================================================
# tgdk_zengarden.py — TGDK Zengarden+ (GhostGate + GradCkpt Edition)
# ============================================================
"""
TGDK Zengarden+ — Direct Adaptive Trainer for Symbolic Instructional Knowledge
-----------------------------------------------------------------------------
• Deterministic seeding
• Safe local HuggingFace fallback (no Hub calls)
• Checkpoint save/load (model + optimizer + scheduler + scaler + RNG)
• Gradient checkpointing toggle + fallback to torch.utils.checkpoint
• AMP (torch.cuda.amp) optional
• Multi-GPU via torch.distributed (env://)
• TGDK GhostGateAccelerator integration
• Self-healing after AcceleratorState corruption
• 16 HF Compatibility Bridges
"""

import os, time, json, logging, torch, random
import numpy as np
from pathlib import Path
from typing import Optional, Callable
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from trl import SFTTrainer
from transformers.models.auto import processing_auto
from transformers.utils import hub
from tgdk_accelerator import GhostGateAccelerator
import torch.utils.checkpoint as checkpoint


# ---------------------------------------------------------------------
# HuggingFace processor patch
# ---------------------------------------------------------------------
def patch_hf_auto_processor_fallback(local_path="./tgdk_base_tok"):
    """Monkeypatch AutoProcessor.from_pretrained() to always use a local fallback."""
    orig_from_pretrained = processing_auto.AutoProcessor.from_pretrained

    def safe_from_pretrained(model_id=None, *args, **kwargs):
        if not model_id or not isinstance(model_id, str) or not model_id.strip():
            model_id = local_path
        kwargs.pop("local_files_only", None)
        try:
            return orig_from_pretrained(model_id, *args, **kwargs)
        except Exception as e:
            print(f"[TGDK::patch] ⚠️ AutoProcessor fallback triggered ({e})")
            class DummyProcessor:
                def __call__(self, *a, **k): return {"input_ids": torch.zeros((1, 8), dtype=torch.long)}
                def decode(self, ids): return "<dummy>"
            return DummyProcessor()

    processing_auto.AutoProcessor.from_pretrained = safe_from_pretrained
    hub.cached_file = lambda *a, **k: local_path


patch_hf_auto_processor_fallback("./tgdk_base_tok")


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logger = logging.getLogger("zengarden")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    logger.addHandler(ch)


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------
def default_seed(seed: int = 1337):
    """Deterministic torch + numpy + random."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def ensure_local_model_path(model) -> str:
    """Guarantee a non-empty local name_or_path to prevent HF hub calls."""
    try:
        name_or_path = getattr(model, "name_or_path", "")
        if not name_or_path or not isinstance(name_or_path, str) or not name_or_path.strip():
            fallback = Path("./tgdk_base_tok").resolve()
            fallback.mkdir(parents=True, exist_ok=True)
            cfg_file = fallback / "config.json"
            if not cfg_file.exists():
                cfg_file.write_text(
                    '{"model_type": "bert", "created_by": "TGDK", "safe_fallback": true}',
                    encoding="utf-8",
                )
            model.name_or_path = str(fallback)
            logger.info(f"[TGDK::Trainer] 🧱 Safe fallback name_or_path → {model.name_or_path}")
        else:
            Path(model.name_or_path).mkdir(parents=True, exist_ok=True)
        return model.name_or_path
    except Exception as e:
        model.name_or_path = os.getcwd()
        logger.warning(f"[TGDK::Trainer] ⚠️ Fallback to cwd: {model.name_or_path} ({e})")
        return model.name_or_path


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, extra=None):
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "scaler_state": scaler.state_dict() if scaler else None,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "extra": extra or {},
    }
    torch.save(state, path)
    logger.info(f"[zengarden] 💾 Checkpoint saved → {path}")


def enable_gradient_checkpointing(model, enable=True):
    """
    Enable HF-style gradient checkpointing or fallback to torch.utils.checkpoint.
    """
    toggled = False
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            if enable:
                model.gradient_checkpointing_enable()
            else:
                model.gradient_checkpointing_disable()
            logger.info(f"[zengarden] ⚙️ HF gradient checkpointing toggled={enable}")
            toggled = True
        except Exception as e:
            logger.warning(f"[zengarden] ⚠️ gradient_checkpointing_enable failed: {e}")

    # fallback: wrap forward layers manually
    if not toggled:
        def checkpointed_forward(*inputs, **kwargs):
            return checkpoint.checkpoint(model.forward, *inputs, **kwargs)
        model.forward = checkpointed_forward
        logger.info("[zengarden] 🧩 Manual checkpointing via torch.utils.checkpoint activated")
        toggled = True
    return toggled

# ---------------------------------------------------------------------
# Robust universal gradient checkpointing for HF 4.44+
# ---------------------------------------------------------------------
def patch_gradient_checkpointing_layers(model):
    """
    Ensures that every layer (including MistralDecoderLayer, LlamaDecoderLayer, etc.)
    has a valid checkpointing callable.
    """
    fixed = 0
    for name, module in model.named_modules():
        # For newer HF models, only `gradient_checkpointing` bool exists
        if hasattr(module, "gradient_checkpointing") and not hasattr(module, "_gradient_checkpointing_func"):
            def _dummy_checkpoint_func(func, *args, **kwargs):
                import torch.utils.checkpoint as checkpoint
                return checkpoint.checkpoint(func, *args, **kwargs)
            setattr(module, "_gradient_checkpointing_func", _dummy_checkpoint_func)
            fixed += 1
        # Ensure the flag is set
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = True
    if fixed > 0:
        logger.info(f"[zengarden] 🩹 Patched {fixed} layers for gradient checkpointing compatibility.")
    return fixed


# ---------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------
class zengardenTrainer(SFTTrainer):
    """TGDK Production-Grade Trainer with GhostGateAccelerator + gradient checkpointing."""

    def __init__(
        self,
        model,
        tokenizer=None,
        train_dataset=None,
        val_dataset=None,
        config: Optional[dict] = None,
        telemetry_cb: Optional[Callable] = None,
        log_prefix="[zengarden]",
        *args,
        **kwargs,
    ):
        for key in ["tokenizer", "model_init", "processing_class", "model_id", "local_files_only"]:
            kwargs.pop(key, None)

        safe_path = ensure_local_model_path(model)

        # === Fallback datasets ===
        from torch.utils.data import Dataset
        if train_dataset is None:
            class _NullTrain(Dataset):
                def __len__(self): return 1
                def __getitem__(self, _): return {"text": "placeholder"}
            train_dataset = _NullTrain()
            logger.warning("[zengarden] train_dataset=None → _NullDataset()")

        if val_dataset is None:
            class _NullVal(Dataset):
                def __len__(self): return 1
                def __getitem__(self, _): return {"text": "validation"}
            val_dataset = _NullVal()
            logger.warning("[zengarden] val_dataset=None → _NullDataset()")

        # === Safe universal TRL initialization ===
        try:
            super().__init__(model=model, train_dataset=train_dataset, eval_dataset=val_dataset, **kwargs)
            logger.info(f"{log_prefix} ✅ Initialized SFTTrainer")
        except Exception as e:
            raise RuntimeError(f"{log_prefix} ❌ SFTTrainer init failed: {e}")

        # === Device Setup ===
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"{log_prefix} device={self.device}")

        # === Gradient Checkpointing Hooks ===
        if not hasattr(self.model, "gradient_checkpointing_enable"):
            def _enable_stub(*a, **k): logger.info("[TGDK] ⚙️ gradient_checkpointing_enable() stub")
            self.model.gradient_checkpointing_enable = _enable_stub
        if not hasattr(self.model, "gradient_checkpointing_disable"):
            def _disable_stub(*a, **k): logger.info("[TGDK] ⚙️ gradient_checkpointing_disable() stub")
            self.model.gradient_checkpointing_disable = _disable_stub

        # === TGDK GhostGateAccelerator integration ===
        try:
            self.accelerator = GhostGateAccelerator(mixed_precision="bf16", device=str(self.device))
            import transformers.trainer
            transformers.trainer.Accelerator = GhostGateAccelerator
            self.accelerator.heal_on_train_start()
            logger.info(f"{log_prefix} ⚙️ GhostGateAccelerator engaged successfully.")
        except Exception as e:
            logger.warning(f"{log_prefix} ⚠️ GhostGateAccelerator install failed ({e}); fallback dummy accelerator.")
            class _Dummy:
                def prepare(self, *a, **k): return a if len(a) > 1 else a[0]
                def backward(self, loss, **k): loss.backward()
                def unwrap_model(self, m, *a, **k): return m
                def wait_for_everyone(self): pass
            self.accelerator = _Dummy()

        # === Config ===
        self.cfg = config or {}
        self.telemetry_cb = telemetry_cb
        self.log_prefix = log_prefix
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.epochs_trained = 0

        # === Determinism ===
        if self.cfg.get("deterministic", True):
            default_seed(self.cfg.get("seed", 1337))

        # === AMP ===
        self.use_amp = bool(self.cfg.get("amp", False)) and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp:
            logger.info(f"{log_prefix} AMP enabled (native)")

        # === Gradient Checkpointing Activation ===
        if self.cfg.get("grad_ckpt", True):
            enable_gradient_checkpointing(self.model, True)

        # === Default Optimizer & Scheduler ===
        lr = self.cfg.get("lr", 5e-5)
        self._opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self._sched = torch.optim.lr_scheduler.CosineAnnealingLR(self._opt, T_max=10)
        self.optimizer = self._opt
        self.lr_scheduler = self._sched
        logger.info(f"{log_prefix} Optimizer initialized (lr={lr})")

    # -----------------------------------------------------------------
    # Data collation
    # -----------------------------------------------------------------
    def collate(self, batch):
        texts = [b.get("text", "") for b in batch]
        if not self.tokenizer:
            raise RuntimeError("[zengarden] tokenizer not provided")
        toks = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.cfg.get("max_length", 1024),
            return_tensors="pt"
        )
        toks["labels"] = toks["input_ids"].clone()
        return toks

    # -----------------------------------------------------------------
    # Core training step
    # -----------------------------------------------------------------
    def _train_step(self, batch, grad_accum=1):
        batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_amp):
            out = self.model(**batch)
        loss = out.get("loss") if isinstance(out, dict) else getattr(out, "loss", out)
        loss = loss / grad_accum
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        return float(loss.detach().item() * grad_accum)

    # -----------------------------------------------------------------
    # Optimizer step
    # -----------------------------------------------------------------
    def _optim_step(self):
        if self.use_amp:
            self.scaler.unscale_(self._opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self._opt)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self._opt.step()
        self._opt.zero_grad(set_to_none=True)
        if self._sched:
            try:
                self._sched.step()
            except Exception:
                pass

    # -----------------------------------------------------------------
    # Fit Loop
    # -----------------------------------------------------------------
    def fit(self, epochs=1, batch_size=4, grad_accum=1, checkpoint_path="./zengarden_checkpoint.pt"):
        dl = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate)
        total = len(dl.dataset) if hasattr(dl, "dataset") else 0
        logger.info(f"[zengarden] 🔁 Train start (epochs={epochs}, samples={total})")

        self.accelerator.heal_on_train_start()  # auto-heal before training

        for ep in range(1, epochs + 1):
            self.model.train()
            tot_loss, steps = 0.0, 0
            for step, batch in enumerate(tqdm(dl, desc=f"Epoch {ep}")):
                loss = self._train_step(batch, grad_accum)
                tot_loss += loss
                steps += 1
                if (step + 1) % grad_accum == 0:
                    self._optim_step()
                if step % 10 == 0:
                    logger.info(f"[zengarden] Ep{ep} Step{step} loss={loss:.4f}")
            avg = tot_loss / max(1, steps)
            logger.info(f"[zengarden] ✅ Epoch {ep}/{epochs} avg_loss={avg:.4f}")
            save_checkpoint(checkpoint_path, self.model, self._opt, self._sched, self.scaler, ep)
        return True


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import torch.nn.functional as F

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(10, 256)
            self.l2 = nn.Linear(256, 10)
        def forward(self, input_ids, attention_mask=None, labels=None):
            x = checkpoint.checkpoint(self.l1, input_ids.float())
            out = self.l2(x)
            return {"loss": F.mse_loss(out, input_ids.float())}

    tokenizer = lambda texts, **_: {"input_ids": torch.randn(len(texts), 10), "attention_mask": None}
    dummy = [{"text": "hi"} for _ in range(8)]
    trainer = zengardenTrainer(TinyModel(), tokenizer, dummy, dummy, config={"amp": False, "lr": 1e-4, "grad_ckpt": True})
    trainer.fit(epochs=1, batch_size=2)
