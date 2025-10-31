# mainmethods.py
# Bridge for rotate_vector and reflect_phi from symbolic_adapter.Vector3

from symbolic_adapter import Vector3


def rotate_vector(vector: Vector3, axis: str = "z", degrees: float = 0.0) -> Vector3:
    """Rotate a Vector3 around x, y or z axis by degrees."""
    return Vector3.rotate_vector(vector, axis, degrees)


def reflect_phi(vector: Vector3) -> Vector3:
    """Apply φ-based reflection to a Vector3."""
    return Vector3.reflect_phi(vector)


def dual_operation(vector: Vector3, axis: str = "z", degrees: float = 90.0) -> tuple[Vector3, Vector3]:
    """Perform both a rotation and a φ-reflection and return both results."""
    rotated = Vector3.rotate_vector(vector, axis, degrees)
    reflected = Vector3.reflect_phi(rotated)
    return rotated, reflected


def phi_chain(vectors: list[Vector3], axis: str = "z", degrees: float = 45.0) -> list[Vector3]:
    """Apply rotation + φ-reflection sequentially to a list of vectors."""
    out = []
    for v in vectors:
        r = Vector3.rotate_vector(v, axis, degrees)
        out.append(Vector3.reflect_phi(r))
    return out


def verify_vector(v: Vector3):
    """Simple diagnostic printout."""
    print(f"[Vector3] {v.intent}  ⟨{v.x:.4f}, {v.y:.4f}, {v.z:.4f}⟩  | |v| = {v.magnitude()}")
