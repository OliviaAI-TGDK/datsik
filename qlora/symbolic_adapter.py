# ─────────────────────────────────────────────────────────────────────────────
# SYMBOLIC ADAPTER — From Recursive Symbolic Systems to QLoRA-Compatible Inputs
# Designed for: Olivia-Aware Architectures, MahaToolkit, HIJQUAp Protocols
# License: BFE-Compliant | φ-Fold Ready | QLoRA Compatible
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import math
import uuid
import torch
from typing import List, Tuple
from torch.utils.data import Dataset

PHI = 1.61803398875  # Golden Ratio


# ───────────────────────────────────────────────
# Core Vector Class
# ───────────────────────────────────────────────
class Vector3:
    def __init__(self, x: float, y: float, z: float, intent: str = "undefined"):
        self.x = x
        self.y = y
        self.z = z
        self.intent = intent

    def __repr__(self):
        return f"⟨{self.x}, {self.y}, {self.z}⟩ :: {self.intent}"

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def magnitude(self) -> float:
        return round(math.sqrt(self.x**2 + self.y**2 + self.z**2), 5)

    def normalize(self) -> Vector3:
        mag = self.magnitude()
        if mag == 0:
            return Vector3(0, 0, 0, self.intent)
        return Vector3(self.x / mag, self.y / mag, self.z / mag, self.intent)

    def scale(self, factor: float) -> Vector3:
        return Vector3(self.x * factor, self.y * factor, self.z * factor, self.intent)

    @staticmethod
    def distance(a: Vector3, b: Vector3) -> float:
        return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)

    @staticmethod
    def dot_product(a: Vector3, b: Vector3) -> float:
        return a.x * b.x + a.y * b.y + a.z * b.z

    @staticmethod
    def cross_product(a: Vector3, b: Vector3) -> Vector3:
        return Vector3(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x,
            intent="cross"
        )

    @staticmethod
    def phi_scale(vector: Vector3) -> Vector3:
        return vector.scale(PHI)

    @staticmethod
    def average_vector(vectors: List[Vector3]) -> Vector3:
        x = sum(v.x for v in vectors) / len(vectors)
        y = sum(v.y for v in vectors) / len(vectors)
        z = sum(v.z for v in vectors) / len(vectors)
        return Vector3(x, y, z, intent="average")

    @staticmethod
    def is_orthogonal(a: Vector3, b: Vector3) -> bool:
        return abs(Vector3.dot_product(a, b)) < 1e-6

    @staticmethod
    def angle_between(a: Vector3, b: Vector3) -> float:
        denom = (a.magnitude() * b.magnitude()) + 1e-8
        return math.acos(Vector3.dot_product(a, b) / denom)

    @staticmethod
    def rotate_vector(vector: Vector3, axis: str, degrees: float) -> Vector3:
        radians = math.radians(degrees)
        cos = math.cos(radians)
        sin = math.sin(radians)
        if axis == 'z':
            x = vector.x * cos - vector.y * sin
            y = vector.x * sin + vector.y * cos
            return Vector3(x, y, vector.z, intent="rotated")
        elif axis == 'x':
            y = vector.y * cos - vector.z * sin
            z = vector.y * sin + vector.z * cos
            return Vector3(vector.x, y, z, intent="rotated")
        elif axis == 'y':
            x = vector.x * cos + vector.z * sin
            z = -vector.x * sin + vector.z * cos
            return Vector3(x, vector.y, z, intent="rotated")
        return vector

    @staticmethod
    def reflect_phi(vector):
        return Vector3.reflect_phi(vector)

    @staticmethod
    def twist_vector(vector: Vector3, factor: float) -> Vector3:
        return Vector3(vector.x + math.sin(vector.y)*factor, vector.y, vector.z, intent="twisted")

    @staticmethod
    def loop_fold(vector: Vector3) -> Vector3:
        return Vector3(-vector.x, -vector.y, -vector.z, intent="looped")

    @staticmethod
    def dual_fold(vector: Vector3) -> List[Vector3]:
        return [vector, Vector3.loop_fold(vector)]

    @staticmethod
    def vibrate(vector: Vector3, frequency: float = 0.1) -> Vector3:
        offset = math.sin(vector.x * frequency)
        return Vector3(vector.x, vector.y + offset, vector.z, intent="vibrated")

    @staticmethod
    def reflect_phi(vector: Vector3) -> Vector3:
        return Vector3(-vector.x * PHI, vector.y * PHI, -vector.z * PHI, intent="phi_reflected")

    @staticmethod
    def collapse_to_origin(vector: Vector3) -> Vector3:
        return Vector3(0, 0, 0, intent="collapsed")

    @staticmethod
    def pulse_stack(vectors: List[Vector3]) -> Vector3:
        return Vector3.average_vector(vectors).scale(PHI)


# ───────────────────────────────────────────────
# Scooty Packet — Symbolic Signal Carrier
# ───────────────────────────────────────────────
class Scooty:
    def __init__(self, event: str, axis: str, source_vector: Vector3, result_vector: Vector3):
        self.event = event
        self.axis = axis
        self.source_vector = source_vector
        self.result_vector = result_vector
        self.intent_carried = source_vector.intent
        self.intent_result = result_vector.intent
        self.signature = f"SCOOTY::{event}::{axis}::{source_vector.intent}"

    def emit(self):
        print(f"[Scooty] 🚀 {self.event}")
        print(f" ↳ Axis: {self.axis}")
        print(f" ↳ From: {self.source_vector}")
        print(f" ↳ To  : {self.result_vector}")
        print(f" ↳ Signature: {self.signature}")

    def as_dict(self):
        return {
            "event": self.event,
            "axis": self.axis,
            "origin": self.source_vector.as_tuple(),
            "reflected": self.result_vector.as_tuple(),
            "intent_in": self.intent_carried,
            "intent_out": self.intent_result,
            "signature": self.signature
        }


# ───────────────────────────────────────────────
# Reflection & Vector Utilities
# ───────────────────────────────────────────────
def reflect_through_fold(vector: Vector3, axis: str = 'x') -> Scooty:
    if axis == 'x':
        reflected = Vector3(-vector.x, vector.y, vector.z, intent="reflected")
    elif axis == 'y':
        reflected = Vector3(vector.x, -vector.y, vector.z, intent="reflected")
    elif axis == 'z':
        reflected = Vector3(vector.x, vector.y, -vector.z, intent="reflected")
    else:
        reflected = vector

    return Scooty(
        event="vector_reflection",
        axis=axis,
        source_vector=vector,
        result_vector=reflected
    )


def scooty_to_vector(scooty: Scooty) -> List[float]:
    origin = list(scooty.source_vector.as_tuple())
    reflected = list(scooty.result_vector.as_tuple())
    intent_hash = hash(scooty.intent_carried) % 1000 / 1000.0
    return origin + reflected + [intent_hash]


# ───────────────────────────────────────────────
# QLoRA-Compatible Dataset from Scooties
# ───────────────────────────────────────────────
class SymbolicSignalDataset(Dataset):
    def __init__(self, scooties: List[Scooty]):
        self.vectors = [torch.tensor(scooty_to_vector(s), dtype=torch.float32) for s in scooties]

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, idx):
        return self.vectors[idx]
