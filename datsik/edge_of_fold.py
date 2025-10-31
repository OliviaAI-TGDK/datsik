# ───────────────────────────────────────────────────────────────
# edge_of_fold.py
# TGDK Sword Module — Edge of Fold (Q-LoRA Reflection Interface)
# Designed for OliviaAI & MahaToolkit — BFE Compliant
# ───────────────────────────────────────────────────────────────

from __future__ import annotations
import torch
import numpy as np
import uuid


class EdgeOfFold:
    def __init__(self, model):
        self.model = model
        self.id = f"Edge-{uuid.uuid4().hex[:8]}"

    def cut_low_gradients(self, threshold=1e-4):
        """Prune parameters with gradients below threshold."""
        cut_count = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                max_grad = torch.abs(param.grad).max().item()
                if max_grad < threshold:
                    param.requires_grad = False
                    cut_count += 1
        print(f"[Sword] ✂️ Cut {cut_count} params below grad threshold {threshold}.")

    def reflect_tensor(self, tensor: torch.Tensor, axis: str = 'x'):
        """Reflect tensor across axis (symbolic mirror slice)."""
        if axis == 'x':
            return tensor.flip(0)
        elif axis == 'y':
            return tensor.flip(1)
        return tensor

    def bind_to_gate(self, tensor: torch.Tensor, gate_id: str = None):
        """Symbolic binding of tensor to gate for logging/debug."""
        gate = gate_id or f"Gate-{uuid.uuid4().hex[:4]}"
        signature = f"SWORD::BIND::{gate}::{tensor.shape}"
        print(f"[Sword] 🧷 Bound tensor to {gate} with signature: {signature}")
        return signature

    def slice_layer(self, layer_name: str):
        """Extract and return a named layer from the model."""
        for name, module in self.model.named_modules():
            if layer_name in name:
                print(f"[Sword] ⚔️ Sliced layer: {name}")
                return module
        print(f"[Sword] Layer '{layer_name}' not found.")
        return None

    def engrave_signature(self, config: dict):
        """Embed sword signature into LoRA config or checkpoint metadata."""
        config['sword_signature'] = self.id
        print(f"[Sword] 🧬 Engraved signature into config: {self.id}")
        return config

    def invoke_edge_alignment(self):
        """Log aligned parameters (e.g., matching direction & gradient intent)."""
        aligned = 0
        for name, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                dot = torch.dot(p.view(-1), p.grad.view(-1))
                if dot > 0:
                    aligned += 1
        print(f"[Sword] 🧭 {aligned} parameters aligned with training intent.")
        return aligned

    def sever_overfit(self, head_names: list):
        """Zero out attention heads or named layers suspected of overfitting."""
        severed = 0
        for name, module in self.model.named_modules():
            if any(h in name for h in head_names):
                if hasattr(module, 'weight'):
                    with torch.no_grad():
                        module.weight.zero_()
                        severed += 1
        print(f"[Sword] 🔥 Severed {severed} overfitting modules.")

    def trace_cut(self, tensor: torch.Tensor):
        """Visualize cut trajectory (e.g., flattened vector shape)."""
        flat = tensor.detach().cpu().numpy().flatten()
        print(f"[Sword] ✴️ Trace Cut Shape: {tensor.shape} → Length: {len(flat)}")
        print(f"[Trace] Preview: {flat[:10]}...")

# ───────────────────────────────────────────────
# Symbolic Kata Execution (External Function)
# ───────────────────────────────────────────────
def perform_kata(blade: EdgeOfFold, model, config):
    """
    Executes symbolic reflection routine using the EdgeOfFold blade.
    Designed for QLoRA pre/post-processing phases.
    """
    print("\n🥋 KATA OF THE EDGE — INITIATED\n")

    blade.cut_low_gradients()
    blade.invoke_edge_alignment()
    blade.engrave_signature(config)

    layer = blade.slice_layer("lm_head")
    if layer is not None and hasattr(layer, "weight"):
        blade.trace_cut(layer.weight)

    print("\n🥋 KATA COMPLETE — THE EDGE IS STILL\n")
    return config
