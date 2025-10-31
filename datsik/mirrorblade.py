"""
MirrorBlade.py — OliviaAI + QLoRA extension
----------------------------------------------------
Symbolic defense + entropy evaluation inside training loops.
Maintains a dynamic entropy lineage (sha256-chained) and
provides reflective classification feedback to RSI / ZenGarden.

Author: Sean “M” Tichenor (TGDK)
License: TGDK-BFE-ST-144
"""

import math
import hashlib
import logging
import json
import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional

# ============================================================
# 🔧 Configuration
# ============================================================
ENTROPY_WINDOW = 512
ENTROPY_WARN = 90.0
ENTROPY_SILENT = 30.0

logging.basicConfig(
    level=logging.INFO,
    format="[MirrorBlade] %(asctime)s — %(message)s",
    datefmt="%H:%M:%S"
)

# ============================================================
# 🔒 Global Entropy State
# ============================================================
_entropy_lock = threading.Lock()
_entropy_ring = []
_entropy_hash = "0"
_entropy_epoch = 0


# ============================================================
# 🔑 Core Entropy Logic
# ============================================================
def _hash_lineage(value: float, prev_hash: str) -> str:
    data = f"{value:.12f}{prev_hash}".encode()
    return hashlib.sha256(data).hexdigest()


def evaluate_entropy(value: float) -> float:
    """Compute symbolic entropy from scalar signal (e.g., loss × grad_norm)."""
    global _entropy_ring, _entropy_hash, _entropy_epoch
    with _entropy_lock:
        delta = abs(value) + 1e-8
        pulse = math.log1p(delta * delta * math.pi) * math.e
        _entropy_ring.append(pulse)
        _entropy_hash = _hash_lineage(pulse, _entropy_hash)
        _entropy_epoch += 1
        if len(_entropy_ring) > ENTROPY_WINDOW:
            _entropy_ring = _entropy_ring[-ENTROPY_WINDOW:]
        return pulse


# ============================================================
# 💓 Symbolic “Sacred Heart” Monitor
# ============================================================
def sacred_heart(step: int, entropy: float) -> str:
    if entropy > ENTROPY_WARN:
        msg = f"[♥] Sacred Heart Triggered — step={step}, entropy={entropy:.3f}"
        logging.warning(msg)
        return "HIGH"
    elif entropy < ENTROPY_SILENT:
        logging.info(f"[♥] Heart Silent — entropy={entropy:.3f}")
        return "LOW"
    else:
        logging.info(f"[♥] Heart Steady — entropy={entropy:.3f}")
        return "STABLE"


# ============================================================
# 🧬 Classification Logic
# ============================================================
def _is_prime(n: int) -> bool:
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    r = int(math.sqrt(n))
    for i in range(3, r + 1, 2):
        if n % i == 0:
            return False
    return True


def classify_node(signal: float) -> str:
    if signal > 91.0:
        return "BLACKSIG"
    if signal > 80.0 and _is_prime(int(signal)):
        return "UNLOCK_NODE"
    if signal > 70.0:
        return "EXFIL_GATE"
    if signal > 60.0:
        return "ANOMALOUS"
    return "SAFE"


# ============================================================
# 🧠 MirrorBlade Callback
# ============================================================
def mirrorblade_step_callback(metrics: Dict[str, float], step: int) -> Dict[str, Any]:
    """
    Integrates with QLoRA or TGDK training loops.
    Expects metrics = {"loss": float, "grad_norm": float}.
    Returns reflective classification + entropy value.
    """
    loss = metrics.get("loss", 0.0)
    grad = metrics.get("grad_norm", 0.0)

    # Combine metrics into signal
    signal = (loss * 1.4) + (grad * 0.33)
    entropy = evaluate_entropy(signal)
    heart_state = sacred_heart(step, entropy)
    classification = classify_node(signal)

    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "step": step,
        "loss": loss,
        "grad_norm": grad,
        "entropy": entropy,
        "entropy_hash": _entropy_hash,
        "epoch": _entropy_epoch,
        "heart_state": heart_state,
        "classification": classification,
    }

    logging.info(json.dumps(payload, indent=2))
    return payload


# ============================================================
# 🪞 MirrorBlade Core Class
# ============================================================
class MirrorBlade:
    """
    MirrorBlade Core Interface
    Used by RSICommand, ZenGarden, and OliviaAI subsystems.
    """

    def __init__(self):
        self.vault_epoch = _entropy_epoch
        self.seed = hashlib.md5(str(time.time()).encode()).hexdigest()[:12]
        logging.info(f"⚙️  MirrorBlade Core initialized (seed={self.seed})")

    def evaluate(self, metrics: Dict[str, float], step: int) -> Dict[str, Any]:
        """Main evaluation entrypoint for training-loop callbacks."""
        result = mirrorblade_step_callback(metrics, step)
        self.vault_epoch = result["epoch"]
        return result

    def get_entropy_hash(self) -> str:
        return _entropy_hash

    def reset(self):
        global _entropy_ring, _entropy_hash, _entropy_epoch
        with _entropy_lock:
            _entropy_ring.clear()
            _entropy_hash = "0"
            _entropy_epoch = 0
        logging.info("[MirrorBlade] 🔄 Entropy state reset.")
