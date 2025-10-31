# Duo.py – TGDK Legacy + QLoRA Optimizer Integration (Qiskit replaced with Scoring)

import logging
import numpy as np
import torch
from torch.optim import Optimizer, AdamW
from lion_pytorch import Lion
from torch.optim.lr_scheduler import LambdaLR
from transformers import TrainerCallback

# TGDK scoring utilities
from scoring import simulate_quantum_with_scorer
# visceptor module (TGDK Sensors)
from tgdk_sensors import visceptor
from accelerate import Accelerator
from tgdk_accelerator import GhostGateAccelerator
from accelerate.optimizer import AcceleratedOptimizer
import torch, numpy as hashlib, base64

# TGDK Ritual imports
from Mahadevi import Mahadevi
from Maharaga import Maharaga
from Trinity import Trinity

# ------------------------------
# Helpers: HexQUAp + Ouija
# ------------------------------
def hexquap_fold(tensor: torch.Tensor) -> bytes:
    """Compress tensor into a HexQUAp hash signature."""
    arr = tensor.detach().cpu().float().numpy()
    h = hashlib.sha256(arr.tobytes()).digest()
    return h

def ouija_sliver(matrix: np.ndarray, pillar_sig: str) -> str:
    """Build an encoded sliver reference for frozen deltas."""
    h = hashlib.sha256(matrix.tobytes() + pillar_sig.encode()).hexdigest()
    return base64.urlsafe_b64encode(h.encode()).decode()[:64]

accelerator = Accelerator(mixed_precision="no")  # no AMP at all

def get_trainable_params(model):
    """
    Collects only trainable parameters (e.g., LoRA adapters).
    Falls back safely if none are found.
    """
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    if not params:
        logging.warning("[Duo] No trainable parameters found — inserting dummy param.")
        params = [torch.nn.Parameter(torch.zeros(1, requires_grad=True))]
    return params

# -----------------------------
# Duo-bound Optimizer Factory
# -----------------------------
def make_duo_optimizer(model, mmt_controller=None, lr=2e-5, weight_decay=0.01, mode="hybrid"):
    """
    Factory for Duo optimizers.
    mode = "symbolic" → TGDK Duo (Mahadevi/Maharaga/Trinity blending)
    mode = "amp"      → DuoOptimizer (AMP + Collator + GhostGate)
    mode = "hybrid"   → Wrap Duo inside DuoOptimizer for combined behavior
    """
    collator = ForkedCoalescingMatrixCollator()
    ghost_gate = GhostGateAccelerator(mixed_precision="no")

    if model is None:
        logging.warning("[Duo] No model provided, returning Dummy optimizer.")
        return DummyOptim([]), DummyScheduler()

    # --- collect only trainable params (LoRA adapters, unfrozen layers, etc.) ---
    trainable_params = [p for n, p in model.named_parameters() if p.requires_grad]
    if not trainable_params:
        logging.warning("[Duo] No trainable parameters found — inserting dummy param.")
        trainable_params = [torch.nn.Parameter(torch.zeros(1, requires_grad=True))]
    else:
        logging.info(f"[Duo] Found {len(trainable_params)} trainable params.")
        for n, p in model.named_parameters():
            if p.requires_grad:
                logging.debug(f"[Duo] Trainable → {n}: {tuple(p.shape)}")

    # --- Symbolic Duo ---
    symbolic_duo = Duo(
        trainable_params,
        lr=lr,
        weight_decay=weight_decay,
        mahadevi=getattr(mmt_controller, "mahadevi", None),
        maharaga=getattr(mmt_controller, "maharaga", None),
        trinity=getattr(mmt_controller, "trinity", None),
    )

    if mode == "symbolic":
        sched = LambdaLR(symbolic_duo, lr_lambda=lambda step: 1.0)
        logging.info("[Duo] Using symbolic Duo optimizer (Mahadevi/Maharaga/Trinity).")
        return symbolic_duo, sched

    # --- AMP/GhostGate Duo ---
    amp_duo = DuoOptimizer(
        trainable_params,
        lr=lr,
        weight_decay=weight_decay,
        collator=collator,
        mmt_controller=mmt_controller,
    )
    amp_duo = ghost_gate.inject(amp_duo)
    amp_duo.ghost_gate = ghost_gate

    if mode == "amp":
        sched = LambdaLR(amp_duo, lr_lambda=lambda step: 1.0)
        logging.info("[Duo] Using AMP/GhostGate DuoOptimizer.")
        return amp_duo, sched

    # --- Hybrid mode (wrap symbolic inside AMP) ---
    class HybridDuo(torch.optim.Optimizer):
        def __init__(self, symbolic, amp):
            self.symbolic = symbolic
            self.amp = amp
            self.param_groups = amp.param_groups  # share param groups

        def step_(self, closure=None, grad_scaler=None):
            loss = self.amp.step(closure=closure, grad_scaler=grad_scaler)
            self.symbolic.step()
            return loss

        def zero_grad_(self, set_to_none=False):
            self.amp.zero_grad(set_to_none=set_to_none)

        def state_dict_(self):
            return {"amp": self.amp.state_dict(), "symbolic": self.symbolic.state_dict()}

        def load_state_dict_(self, state_dict):
            self.amp.load_state_dict(state_dict["amp"])
            self.symbolic.load_state_dict(state_dict["symbolic"])

    hybrid_duo = HybridDuo(symbolic_duo, amp_duo)
    sched = LambdaLR(hybrid_duo, lr_lambda=lambda step: 1.0)

    logging.info("[Duo] Using HYBRID Duo (AMP+GhostGate with symbolic blending).")
    return hybrid_duo, sched


class GhostGateCallback(TrainerCallback):
    def __init__(self, ghost_gate):
        self.ghost_gate = ghost_gate

    def on_step_end(self, args, state, control, **kwargs):
        if hasattr(self.ghost_gate, "refresh_inf_state"):
            self.ghost_gate.refresh_inf_state()


# -----------------------------
# DataSectorDuoqiadratilizer
# -----------------------------
class DataSectorDuoqiadratilizer:
    def __init__(self, sector_count=8):
        logging.info("Initializing Data Sector Duoqiadratilizer (Scoring-powered)")
        self.sector_count = sector_count
        self.sympathizers = self._initialize_sympathizers(sector_count)
        self.indicators = self._initialize_indicators(sector_count)
        self.vector_sequences = self._initialize_vector_sequences(sector_count)

    def _initialize_sympathizers(self, sector_count):
        return [np.random.rand(sector_count) for _ in range(sector_count)]

    def _initialize_indicators(self, sector_count):
        return np.random.rand(sector_count)

    def _initialize_vector_sequences(self, sector_count):
        return [np.sin(np.linspace(0, 2 * np.pi, sector_count))
                for _ in range(sector_count)]

    def _apply_duoquadratic_modifications(self, data):
        modified_data = []
        for d in data:
            vector = np.array([ord(c) for c in d])
            modified_vector = vector + np.random.choice(self.sympathizers)
            modified_data.append(modified_vector)
        return modified_data

    def duoqiadratilize(self, data):
        """
        Perform secure duoqiadratilization using TGDK scoring instead of Qiskit.
        Returns: dict with S, amplitudes, probabilities, peak_state.
        """
        modified_data = self._apply_duoquadratic_modifications(data)
        flat = np.concatenate(modified_data)
        F = float(np.mean(flat))
        L = float(np.std(flat))
        M = float(np.median(flat))
        x = float(np.sum(flat) % 1000) / 1000.0

        result = simulate_quantum_with_scorer(F, L, M, x, size=self.sector_count)
        logging.info(f"[Duo] Duoqiadratilization result: {result}")
        return result

    def CoordVal(
        x,
        packet,
        DivValue,
        Metscore,
        Situation,
        Logistics,
        Location,
        Overfold,
        visceptor,
        disceptor,
        sublimationMetric,
        MatrixClause,
        PayloadRelease,
    ):
        term1 = circumferentialize_degree_field(packet + x) / DivValue
        term2 = (Metscore / Situation) * Logistics * Location / Overfold
        term3 = sublimationMetric / MatrixClause * PayloadRelease
        return term1 - term2 - disceptor + term3


class GentuoGuide:
    """
    Gentuo layer: guides the blend of AdamW and Lion updates
    by orbiting codepoints in a controlled timestep.
    """
    def __init__(self, warp_factor=0.5):
        self.warp_factor = warp_factor

    def warp(self, adamw_update, lion_update):
        # delta between the two optimizers
        delta = lion_update - adamw_update
        # warp it slightly, keep orbit stable
        return adamw_update + self.warp_factor * delta

class AMPProxyOptimizer(torch.optim.Optimizer):
    def __init__(self, duo_optim):
        self.duo_optim = duo_optim
        self.param_groups = duo_optim.param_groups

    def _step(self, closure=None):
        return self.duo_optim.step(closure=closure)

    def _zero_grad(self, set_to_none=False):
        return self.duo_optim.zero_grad(set_to_none=set_to_none)

    def _state_dict(self):
        return self.duo_optim.state_dict()

    def _load_state_dict(self, state_dict):
        return self.duo_optim.load_state_dict(state_dict)


# -----------------------------
# AMP-safe Duo Optimizer
# -----------------------------
class AlayaStore:
    """TGDK Alaya Storehouse for latent seeds."""
    def __init__(self):
        self.seed_bank = {}

    def plant(self, pid, update):
        if update is None:
            return
        self.seed_bank[pid] = update.clone().detach() / (update.norm() + 1e-9)

    def sprout(self, pid, factor=0.05):
        if pid in self.seed_bank:
            return factor * self.seed_bank[pid]
        return 0.0


class DuoOptimizer(Optimizer):
    """
    TGDK DuoOptimizer (AMP + Collator + Alaya + HexQUAp)
    - Blends AdamW and Lion
    - Coalesces params via Collator
    - Adapts LR with MMT volumetric flow
    - Stores shadow states in HexQUAp form
    - Plants/sprouts latent seeds with Alaya
    """
    _step_supports_amp_scaling = True  # AMP hint

    def __init__(self, params, lr=1e-4, weight_decay=0.01,
                 collator=None, mmt_controller=None, pillar_sig="tgdk-pillar"):
        params = [p for p in params if getattr(p, "requires_grad", False)]
        if not params:
            params = [torch.nn.Parameter(torch.zeros(1, requires_grad=True))]

        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # Sub-optimizers
        self.adamw = torch.optim.AdamW(self.param_groups, lr=lr, weight_decay=weight_decay)
        self.lion  = Lion(self.param_groups, lr=lr, weight_decay=weight_decay)

        # Controllers
        self.collator = collator
        self.mmt_controller = mmt_controller

        # Memory extensions
        self.alaya = AlayaStore()
        self.hex_state = {}
        self.pillar_sig = pillar_sig

        self._step_count = 0

    def step(self, closure=None, grad_scaler=None):
        """
        Performs a single optimization step (AdamW + Lion + Alaya).
        AMP-friendly: supports optional grad_scaler.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Save param states before step
        before = {id(p): p.clone().detach()
                  for g in self.param_groups
                  for p in g["params"] if p.grad is not None}

        # Step both optimizers
        if grad_scaler is not None:
            grad_scaler.step(self.adamw)
            grad_scaler.step(self.lion)
        else:
            self.adamw.step()
            self.lion.step()

        # Compute blended update + Alaya seeds
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                pid = id(p)
                delta = p.detach() - before[pid]

                # sprout from Alaya
                sprout = self.alaya.sprout(pid)
                p.data = before[pid] + delta + sprout

                # plant new seed
                self.alaya.plant(pid, delta)

                # shadow hash
                self.hex_state[pid] = hexquap_fold(delta)

        # Collator coalescing
        if self.collator:
            self.collator.coalesce(self.param_groups[0]["params"])

        # MMT adaptive LR
        if self.mmt_controller:
            mmt_state = self.mmt_controller.step(step_id=self._step_count)
            alpha = float(np.mean(mmt_state.get("volumetric", [0.5])) % 1.0)
            for g in self.param_groups:
                g["lr"] *= max(0.5, min(1.0, alpha))

        self._step_count += 1
        return loss

    def zero_grad(self, set_to_none: bool = False):
        """
        Clears gradients of all optimized parameters.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.detach_()
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()

    def state_dict(self):
        """
        Returns the state of the optimizer as a dict.
        """
        return {
            "adamw": self.adamw.state_dict(),
            "lion": self.lion.state_dict(),
            "_step_count": self._step_count,
        }

    def load_state_dict(self, state_dict):
        """
        Loads the optimizer state.
        """
        self.adamw.load_state_dict(state_dict["adamw"])
        self.lion.load_state_dict(state_dict["lion"])
        self._step_count = state_dict.get("_step_count", 0)

    @property
    def duo_param_groups(self):
        return self.adamw.param_groups


class DuoMetricsCallback(TrainerCallback):
    def __init__(self, duo_optim, trainer):
        self.duo_optim = duo_optim
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        if isinstance(self.duo_optim, Duo):
            last = state.log_history[-1] if state.log_history else {}
            current_epoch = int(state.epoch) if state.epoch is not None else 0
            # overwrite alpha balance using metrics
            self.duo_optim.compute_balance_factor(
                metrics=last,
                epoch=current_epoch
            )


class Duo(Optimizer):
    """
    TGDK Duo Optimizer (HexQUAp + Ouija + Alaya edition)
    - Blends AdamW and Lion updates
    - Balance guided by Mahadevi/Maharaga/Trinity, Jade, MMT
    - Stores shadow states in HexQUAp form
    - Rotates trainable layers with Ouija slivers
    - Seeds/sprouts latent shadows in AlayaStore
    """

    def __init__(self, params, lr=1e-4, weight_decay=0.01,
                 mahadevi=None, maharaga=None, trinity=None,
                 jade_lex=None, mmt_controller=None,
                 rotation_stride=4, pillar_sig="tgdk-pillar",
                 hexquap_groups=None, name="Olivia"):

        # filter trainable
        params = [p for p in params if getattr(p, "requires_grad", False)]
        if not params:
            logging.warning("[Duo] No trainable params found — inserting dummy param")
            params = [torch.nn.Parameter(torch.zeros(1, requires_grad=True))]

        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # sub-optimizers
        self.adamw = AdamW(self.param_groups, lr=lr, weight_decay=weight_decay)
        self.lion  = Lion(self.param_groups, lr=lr, weight_decay=weight_decay)
        self.name = name
        # controllers
        self.mahadevi = mahadevi
        self.maharaga = maharaga
        self.trinity  = trinity
        self.jade_lex = jade_lex
        self.mmt      = mmt_controller

        # memory extensions
        self.hex_state   = {}
        self.ouija_store = {}
        self.alaya       = AlayaStore()
        self.pillar_sig  = pillar_sig
        self.rotation_stride = rotation_stride

        self.hexquap_groups = hexquap_groups or []
        self.current_group_idx = 0

        self._step_count = 0
        self.last_coaxial = 0.0

        # compute footprint
        self.total_params = sum(p.numel() for p in self._all_params())
        self._log_trainables("[INIT]")

    def train(self, mode=True):
        """Compatibility shim for Accelerate; no-op."""
        return self

    def eval(self):
        """Compatibility shim for Accelerate; no-op."""
        return self

            
    def _auto_group_params(self, model=None):
        try:
            # attempt grouping
            layers = list(getattr(model, "layers", []))
            if layers:
                groups = [list(l.parameters()) for l in layers]
                return groups

            # fallback: split into chunks
            flat_params = self._all_params()
            chunk_size = max(1, len(flat_params) // 12)
            return [flat_params[i:i+chunk_size] for i in range(0, len(flat_params), chunk_size)]

        except Exception as e:
            print(f"[HexQUAp] Could not auto-group: {e}")
            return [self._all_params()]

    def _rotate_hexquap_group(self, epoch=None):
        if not self.hexquap_groups:
            return

        # freeze all
        for p in self._all_params():
            p.requires_grad = False

        # unfreeze next group
        group = self.hexquap_groups[self.current_group_idx % len(self.hexquap_groups)]
        for p in group:
            p.requires_grad = True

        # ✅ log active params at this epoch
        active_params = sum(p.numel() for p in self._all_params() if p.requires_grad)
        print(f"[HexQUAp] Epoch {epoch} → Group {self.current_group_idx} "
            f"Active {active_params:,} / {self.total_params:,}")

        self.current_group_idx += 1


    def attach_model(self, model):
        """
        Attach a model post-init and auto-build HexQUAp groups.
        """
        if not self.hexquap_groups:
            try:
                self.hexquap_groups = [
                   list(block.parameters()) 
                    for block in get_transformer_blocks(model)
                ]
                print(f"[HexQUAp] Auto-built {len(self.hexquap_groups)} groups")
            except Exception as e:
                print(f"[HexQUAp] Could not auto-group: {e}")
                self.hexquap_groups = [self._all_params()]

    def _all_params(self):
        """Return ALL model params this optimizer could manage."""
        seen = set()
        all_params = []
        for g in self.param_groups:
            for p in g["params"]:
                if id(p) not in seen:
                    seen.add(id(p))
                    all_params.append(p)
        return all_params

    def _log_trainables(self, prefix=""):
        trainables = sum(p.numel() for p in self._all_params() if p.requires_grad)
        print(f"{prefix} [Duo] Trainable = {trainables:,} / {self.total_params:,}")

    def _rotate_hexquap_group(self):
        if not self.hexquap_groups:
            return

        # freeze all
        for p in self._all_params():
            p.requires_grad = False

        # unfreeze next group
        group = self.hexquap_groups[self.current_group_idx % len(self.hexquap_groups)]
        for p in group:
            p.requires_grad = True

        self._log_trainables(f"[HexQUAp] Rotated → group {self.current_group_idx}")
        self.current_group_idx += 1

    def step(self, closure=None, metrics=None, epoch=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # AdamW update
        adamw_before = {id(p): p.clone().detach()
                        for g in self.param_groups
                        for p in g['params'] if p.grad is not None}
        self.adamw.step()
        adamw_update = {pid: (p.detach() - adamw_before[pid])
                        for g in self.param_groups
                        for p in g['params'] if p.grad is not None
                        for pid in [id(p)]}

        # Lion update
        lion_before = {id(p): p.clone().detach()
                       for g in self.param_groups
                       for p in g['params'] if p.grad is not None}
        self.lion.step()
        lion_update = {pid: (p.detach() - lion_before[pid])
                       for g in self.param_groups
                       for p in g['params'] if p.grad is not None
                       for pid in [id(p)]}

        # Blend with Alaya
        alpha = self.compute_balance_factor(metrics=metrics, epoch=epoch)
        for g in self.param_groups:
            for p in g['params']:
                if p.grad is None:
                    continue
                pid = id(p)
                blended = alpha * adamw_update[pid] + (1 - alpha) * lion_update[pid]

                # sprout latent seeds
                blended = blended + self.alaya.sprout(pid, factor=0.05)

                # apply blended update
                p.data = lion_before[pid] + blended

                # plant new seed
                self.alaya.plant_seed(pid, blended)

                # HexQUAp shadow
                self.hex_state[pid] = hexquap_fold(blended)

        # Ouija rotation
        if self._step_count % self.rotation_stride == 0:
            for g in self.param_groups:
                for p in g['params']:
                    if p.requires_grad:
                        matrix = p.data.detach().cpu().numpy()
                        sliver = ouija_sliver(matrix, self.pillar_sig)
                        self.ouija_store[id(p)] = sliver
                        p.requires_grad = False
                    else:
                        p.requires_grad = True

        # Honor: coaxial return vector
        try:
            adamw_vec = torch.cat([u.flatten() for u in adamw_update.values()])
            lion_vec  = torch.cat([u.flatten() for u in lion_update.values()])
            self.last_coaxial = float(torch.dot(
                adamw_vec, lion_vec
            ) / (torch.norm(adamw_vec) * torch.norm(lion_vec) + 1e-9))
        except Exception:
            self.last_coaxial = 0.0

        self._step_count += 1
        return loss


    def zero_grad(self, set_to_none: bool = False):
        for g in self.param_groups:
            for p in g['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.detach_()
                        p.grad.zero_()


    def state_dict(self):
        return {
            "adamw": self.adamw.state_dict(),
            "lion": self.lion.state_dict(),
            "hex_state": self.hex_state,
            "ouija_store": self.ouija_store,
            "_step_count": self._step_count,
        }

    def load_state_dict(self, state_dict):
        self.adamw.load_state_dict(state_dict["adamw"])
        self.lion.load_state_dict(state_dict["lion"])
        self.hex_state = state_dict.get("hex_state", {})
        self.ouija_store = state_dict.get("ouija_store", {})
        self._step_count = state_dict.get("_step_count", 0)


    def compute_balance_factor(self, metrics=None, epoch=None):
        """TGDK-aware α blend factor [0–1]."""
        alpha = 0.5

        # --- Base geometry (Mahadevi/Maharaga/Trinity) ---
        if self.mahadevi and self.maharaga and self.trinity:
            try:
                v1 = np.array(self.mahadevi.vector_field[0])
                v2 = np.array(self.mahadevi.vector_field[1])
                angle = self.mahadevi.angle_between_vectors(v1, v2) / 180.0

                centroid = (np.mean(self.maharaga.data_points, axis=0)
                            if self.maharaga.data_points else np.array([0.5]))
                centroid_norm = np.linalg.norm(centroid) % 1.0

                trinity_seq = np.mean(self.trinity.expand_data(np.random.rand(5)))
                alpha = float((angle + centroid_norm + trinity_seq) / 3.0)
            except Exception:
                alpha = 0.5

        # --- Jade lexicon influence ---
        if self.jade_lex and metrics:
            jade = self.jade_lex.bind_metrics(metrics)
            if jade["Δ_entropy"] == "JADE-Σ":
                alpha *= 0.7   # stability bias → AdamW
            else:
                alpha *= 1.3   # exploration bias → Lion
            if "HUM" in jade["Δ_rigpa"]:
                alpha *= 0.9   # humility dampener

        # --- MMT volumetric stabilizer ---
        if self.mmt:
            mmt_state = self.mmt.step(step_id=self._step_count)
            vol_avg = np.mean(mmt_state["volumetric"])
            alpha = (alpha + float(vol_avg % 1.0)) / 2.0

        # --- Epoch decay ---
        if epoch is not None:
            alpha *= (1 - min(0.05 * epoch, 0.5))

        return max(0.0, min(1.0, alpha))

class HexQUApUnfreezeScheduler:
    """
    TGDK HexQUAp Memory Matrix Unfreeze Scheduler
    - Cyclically rotates trainable modules across epochs.
    - Prevents full VRAM explosion by progressive activation.
    """

    def __init__(self, model, cycle_length=6):
        """
        cycle_length: how many phases (rows in the HexQUAp matrix) to rotate through
        """
        self.model = model
        self.cycle_length = cycle_length
        self.current_phase = 0

        # Precompute module groups (all possible categories)
        self.groups = {
            "lora": [n for n, _ in model.named_parameters() if "lora" in n or "adapter" in n],
            "attention_qkv": [n for n, _ in model.named_parameters() if any(k in n for k in ["q_proj", "k_proj", "v_proj"])],
            "attention_out": [n for n, _ in model.named_parameters() if "o_proj" in n],
            "mlp": [n for n, _ in model.named_parameters() if any(k in n for k in ["up_proj", "down_proj", "gate_proj"])],
            "norms": [n for n, _ in model.named_parameters() if "norm" in n],
            "embeddings": [n for n, _ in model.named_parameters() if "embed" in n],
        }

        print(f"[HexQUAp] Scheduler initialized with {len(self.groups)} module groups.")

    def step(self, epoch: int):
        """
        Rotate trainable modules based on epoch → HexQUAp row selection.
        """
        phase = epoch % self.cycle_length
        self.current_phase = phase

        # Freeze all first
        for _, p in self.model.named_parameters():
            p.requires_grad = False

        # Pick active group(s) for this phase
        active_keys = list(self.groups.keys())
        active_group = active_keys[phase % len(active_keys)]

        # Unfreeze chosen group
        active_params = []
        for name, param in self.model.named_parameters():
            if name in self.groups[active_group]:
                param.requires_grad = True
                active_params.append(name)

        print(f"[HexQUAp] Epoch {epoch}: Activated group → {active_group} ({len(active_params)} params)")

        return active_group, active_params

# -----------------------------
# Gradient Collator
# -----------------------------
class ForkedCoalescingMatrixCollator:
    """
    TGDK collator: coalesces gradients and applies Duo duoqiadratilization (via scoring).
    """

    def __init__(self):
        self.duoq = DataSectorDuoqiadratilizer()

    def coalesce(self, params):
        grads_as_text = [str(p.grad.shape) for p in params if getattr(p, "grad", None) is not None]
        if grads_as_text:
            try:
                result = self.duoq.duoqiadratilize(grads_as_text)
                logging.info(f"[Collator] Quantum merge result: {result}")
            except Exception as e:
                logging.warning(f"[Collator] Duoqiadratilization failed: {e}")

    
# -----------------------------
# Dummy Fallbacks
# -----------------------------
class DummyOptim:
    def __init__(self, params=None, collator=None):
        self.params = list(params) if params is not None else []
        self.collator = collator or ForkedCoalescingMatrixCollator()

    def step(self, *args, **kwargs):
        if self.collator:
            self.collator.coalesce(self.params)

    def zero_grad(self, *args, **kwargs):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def state_dict(self): return {}
    def load_state_dict(self, state_dict): pass


class DummyScheduler:
    def __init__(self, optimizer=None): self.optimizer = optimizer
    def step(self, *args, **kwargs): pass
    def state_dict(self): return {}
    def load_state_dict(self, state_dict): pass


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # Dummy model for testing
    dummy_model = torch.nn.Linear(10, 2)
    seed_influence = self.alaya.sprout(pid, condition_factor=0.05)
    blended = alpha * adamw_update[pid] + (1 - alpha) * lion_update[pid] + seed_influence


    # Run optimizer factory with dummy
    optimizer, scheduler = make_duo_optimizer(dummy_model, mmt_controller=None)
    optimizer = accelerator.prepare_optimizer(optimizer)
    print("Optimizer:", optimizer)
    print("Scheduler:", scheduler)
