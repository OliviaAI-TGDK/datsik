# compat_dummy.py
from typing import Any, Dict
import torch


class ForkedCoalescingMatrixCollator:
    """
    TGDK-style collator that coalesces parameter updates.
    Extend this with entropy-aware routing / symbolic backtrace if needed.
    """
    def coalesce(self, params):
        for p in params:
            if hasattr(p, "grad") and p.grad is not None:
                # Example: normalize or merge gradients
                p.grad = torch.clone(p.grad)  # placeholder


class DummyOptim:
    """
    Custom replacement for Accelerate DummyOptim.
    Behaves like an optimizer but integrates collator logic.
    """
    def __init__(self, params=None, collator=None):
        self.params = list(params) if params is not None else []
        self.collator = collator or ForkedCoalescingMatrixCollator()

    def step(self, *args, **kwargs):
        if self.collator is not None:
            self.collator.coalesce(self.params)

    def zero_grad(self, *args, **kwargs):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        pass


class DummyScheduler:
    """
    Custom replacement for Accelerate DummyScheduler.
    """
    def __init__(self, optimizer=None):
        self.optimizer = optimizer

    def step(self, *args, **kwargs):
        # No scheduling logic by default
        pass

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        pass
