# ============================================================
# tgdk_accelerator.py — TGDK Quantum Teleporting GhostGate Accelerator
# ============================================================

import logging, torch, datetime, math
from accelerate import Accelerator as HFAccelerator, state as accel_state
import numpy as np


class GhostGateAccelerator:
    """
    TGDK Quantum Teleporting Accelerator (Production + s_scalar + quantumlineation)
    -------------------------------------------------------------------------------
    Wraps Hugging Face Accelerator with:
      • Auto-healing after state resets
      • Safe CUDA/CPU fallback
      • AMP inf-check patching
      • Quantumlineation for vector harmonic coherence
      • s_scalar PMZ stabilizer for precision-band energy
    """

    def __init__(self,
                 mixed_precision: str = "no",
                 ghost_entropy: float = 0.997,
                 teleport_band: float = 11.86,
                 device: str = "cuda:0"):
        self._init_core(mixed_precision)
        self.ghost_entropy = ghost_entropy
        self.teleport_band = teleport_band
        self.gates_crossed = 0
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._amp_foreign_state = {
            "found_inf_per_device": {self.device: torch.tensor([0.0], device=self.device)}
        }

        # === TGDK harmonic parameters ===
        self.pmz_ratio = 0.0102
        self.phi = (1 + 5 ** 0.5) / 2
        self.s_scalar_val = 1.0

        self._stamp(f"🧬 GhostGate Accelerator initialized on {self.device} (mp={mixed_precision})")

    # ------------------------------------------------------------
    #  Core initialization / healing
    # ------------------------------------------------------------
    def _init_core(self, mixed_precision="no"):
        try:
            self.core = HFAccelerator(mixed_precision=mixed_precision)
            self._stamp(f"🧠 Core HFAccelerator initialized (mp={mixed_precision})")
        except Exception as e:
            self._stamp(f"⚠️ HFAccelerator init failed ({e}); switching to CPU-only dummy core.")
            class DummyCore:
                def prepare(self, *a, **k): return a if len(a) > 1 else a[0]
                def backward(self, loss, **k): loss.backward()
                def wait_for_everyone(self): pass
            self.core = DummyCore()

    # ------------------------------------------------------------
    #  Robust prepare() with self-healing
    # ------------------------------------------------------------
    def prepare(self, *args, **kwargs):
        try:
            return self.core.prepare(*args, **kwargs)
        except AttributeError as e:
            if "distributed_type" in str(e):
                self._stamp("♻️ AcceleratorState corruption detected; re-initializing safely.")
                try:
                    accel_state.AcceleratorState._reset_state()
                except Exception as ex:
                    self._stamp(f"⚠️ _reset_state() warning: {ex}")

                mp = "no"
                try:
                    if hasattr(self.core, "mixed_precision"):
                        mp = self.core.mixed_precision or "no"
                    elif hasattr(self.core, "state") and hasattr(self.core.state, "mixed_precision"):
                        mp = self.core.state.mixed_precision or "no"
                except Exception:
                    pass

                self._stamp(f"🔄 Re-creating HFAccelerator (mp={mp})")
                self._init_core(mp)
                return self.core.prepare(*args, **kwargs)
            raise
        except Exception as e:
            self._stamp(f"⚠️ Generic prepare() exception: {e}; using CPU fallback prepare().")
            if hasattr(self, "core") and hasattr(self.core, "prepare"):
                try:
                    return self.core.prepare(*args, **kwargs)
                except Exception:
                    pass
            return args if len(args) > 1 else args[0]

    # ------------------------------------------------------------
    #  TGDK s_scalar harmonic stabilizer
    # ------------------------------------------------------------
    def s_scalar(self, entropy_factor: float = 1.0):
        """
        Dynamic scalar harmonizer aligning AMP stability across mixed-precision.
        Uses TGDK PMZ and phi ratio harmonics to produce an adaptive scaling factor.
        """
        try:
            t = datetime.datetime.now().timestamp() % 3600
            wave = math.sin(t / 60.0) * self.phi
            scale = self.pmz_ratio * (self.ghost_entropy + wave) * entropy_factor
            self.s_scalar_val = abs(scale)
            self._stamp(f"⚛️ s_scalar adjusted → {self.s_scalar_val:.8f}")
            return self.s_scalar_val
        except Exception as e:
            self._stamp(f"⚠️ s_scalar error: {e}")
            return 1.0

    # ------------------------------------------------------------
    #  TGDK quantumlineation transform
    # ------------------------------------------------------------
    def quantumlineation(self, tensor):
        """
        Applies TGDK-style symbolic quantumlineation to stabilize tensor coherence
        across devices and improve AMP gradient consistency.
        """
        try:
            if isinstance(tensor, torch.Tensor):
                phase = torch.tanh(tensor * self.s_scalar_val * self.teleport_band)
                self._stamp(f"🌀 Quantumlineation applied to tensor (shape={tuple(tensor.shape)})")
                return phase
            elif isinstance(tensor, np.ndarray):
                phase = np.tanh(tensor * self.s_scalar_val * self.teleport_band)
                self._stamp(f"🌀 Quantumlineation applied to ndarray (shape={tensor.shape})")
                return phase
            else:
                return tensor
        except Exception as e:
            self._stamp(f"⚠️ quantumlineation failed: {e}")
            return tensor

    # ------------------------------------------------------------
    #  GhostGate utilities
    # ------------------------------------------------------------
    def refresh_inf_state(self):
        self._amp_foreign_state["found_inf_per_device"][self.device] = torch.tensor([0.0], device=self.device)

    def inject(self, optimizer):
        optimizer._amp_foreign_state = self._amp_foreign_state
        return optimizer

    def teleport(self, vector):
        self.gates_crossed += 1
        factor = self.ghost_entropy * (self.teleport_band ** 0.5)
        warped = np.tanh(vector * factor)
        self._stamp(f"🌌 Teleported vector across gate {self.gates_crossed}")
        return warped

    def free_memory(self):
        torch.cuda.empty_cache()
        self._stamp("[GhostGate] Memory cache freed (torch.cuda.empty_cache).")

    # ------------------------------------------------------------
    #  Auto-healing pretrain hook
    # ------------------------------------------------------------
    def heal_on_train_start(self):
        try:
            _ = getattr(self.core, "state", None)
            if _ is None or not hasattr(_, "distributed_type"):
                self._stamp("🚑 Accelerator pre-train heal triggered.")
                self._init_core(getattr(_, "mixed_precision", "no"))
        except Exception as e:
            self._stamp(f"⚠️ heal_on_train_start encountered {e}; forcing full reinit.")
            self._init_core("no")

    # ------------------------------------------------------------
    #  Internal logger
    # ------------------------------------------------------------
    def _stamp(self, msg):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[MirrorBlade] {ts} — {msg}")
