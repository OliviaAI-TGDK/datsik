# ─────────────────────────────────────────────────────────────────────────────
# NEWVECTOR MATH KIT — Symbolic Vector Architecture for Recursive Environments
# Designed for integration with DuolineatedSynthesizer, MahaToolkit, HIJQUAp
# License: BFE-Compliant / Overfold-Ready / Olivia-Aware
# ─────────────────────────────────────────────────────────────────────────────

import math
from typing import Tuple

PHI = 1.61803398875  # Golden ratio

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
        # Emits to console or telemetry (can be extended)
        print(f"[Scooty] 🚀 Event: {self.event}")
        print(f"         ↳ Axis: {self.axis}")
        print(f"         ↳ From: {self.source_vector}")
        print(f"         ↳ To  : {self.result_vector}")
        print(f"         ↳ Signature: {self.signature}")

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


class Vector3:
    def __init__(self, x: float, y: float, z: float, intent: str = None):
        self.x = x
        self.y = y
        self.z = z
        self.intent = intent or "undefined"

    def __repr__(self):
        return f"⟨{self.x}, {self.y}, {self.z}⟩ :: {self.intent}"

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def magnitude(self) -> float:
        return round(math.sqrt(self.x**2 + self.y**2 + self.z**2), 5)

    def normalize(self):
        mag = self.magnitude()
        if mag == 0:
            return Vector3(0, 0, 0, self.intent)
        return Vector3(self.x / mag, self.y / mag, self.z / mag, self.intent)

    def scale(self, factor: float):
        return Vector3(self.x * factor, self.y * factor, self.z * factor, self.intent)

    def add(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z, self.intent)

    def subtract(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z, self.intent)


# 🌀 Snellineated path between two vectors (phi-based curve)
def snell_curve(start: Vector3, end: Vector3) -> Tuple[Vector3, Vector3, Vector3]:
    mid_x = (start.x + end.x) / 2
    mid_y = (start.y + end.y) / 2 + math.sin(start.x + end.x) * PHI
    mid_z = (start.z + end.z) / 2
    mid = Vector3(mid_x, mid_y, mid_z, intent="snell_mid")
    return (start, mid, end)

# 🔁 Reflect through a fold (mirror axis)
def reflect_through_fold(vector: Vector3, axis: str = 'x') -> Scooty:
    """
    Reflects a vector through a specified fold axis.
    Returns a Scooty object (symbolic telemetry packet).
    """
    if axis == 'x':
        reflected = Vector3(-vector.x, vector.y, vector.z, intent="reflected")
    elif axis == 'y':
        reflected = Vector3(vector.x, -vector.y, vector.z, intent="reflected")
    elif axis == 'z':
        reflected = Vector3(vector.x, vector.y, -vector.z, intent="reflected")
    else:
        reflected = vector  # No-op

    scooty = Scooty(
        event="vector_reflection",
        axis=axis,
        source_vector=vector,
        result_vector=reflected
    )

    scooty.emit()  # Optional: announce its movement
    return scooty
    return reflected

# ✴️ Quantum dot (contextual overlap)
def quantum_dot(a: Vector3, b: Vector3) -> float:
    dot = a.x * b.x + a.y * b.y + a.z * b.z
    overlap = dot / (a.magnitude() * b.magnitude() + 1e-8)
    return round(overlap * PHI, 5)  # Tuned by golden ratio

# 🪞 Echo-mirror pulse through Ouroboros-style return gate
def echo_mirror(vector: Vector3) -> Vector3:
    return vector.scale(-1).normalize()

# 🔗 Bind vector to a gate (symbolic modulation)
def bind_to_gate(vector: Vector3, gate_id: str) -> str:
    return f"Vector {vector} bound to Gate[{gate_id}] with intent: {vector.intent}"

# 📡 Emit pulse (for visualization, symbolic echo)
def pulse(vector: Vector3) -> str:
    return f"::Pulse:: {vector} ⟶ magnitude: {vector.magnitude()} | intent: {vector.intent}"
