# SPMF.py – TGDK Subcutaneous Particle Matrices Foundation
# Author: TGDK Labs / OliviaAI
# License: TGDK BFE

import numpy as np
from sklearn.cluster import KMeans
import hashlib
import datetime
import os, json

class SubcutaneousParticleMatrix:
    """
    TGDK Subcutaneous Particle Matrix
    - Particleizes matter (image/video frames, embeddings) into ratios
    - Maintains one true Bound Heart Ratio (BHR) as immutable anchor
    - Supports morphing into alternate forms while retaining human nature
    """

    def __init__(self, anchor_vector=None, outdir="./out"):
        self.anchor = anchor_vector if anchor_vector is not None else np.ones((64,))
        self.forms = {}
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)

    # --- Particleization ---
    def particleize(self, frame, clusters=64):
        """
        Convert an image/video frame or embedding into particle clusters.
        Returns cluster centers as particle representation.
        """
        flat = frame.reshape(-1, frame.shape[-1]) if frame.ndim > 1 else frame.reshape(-1, 1)
        kmeans = KMeans(n_clusters=clusters, n_init=5).fit(flat)
        particles = kmeans.cluster_centers_
        return particles

    # --- Binding new forms ---
    def bind_form(self, name, particles):
        """
        Bind a new form normalized against the BHR.
        Stores ratio distance so divergence can be monitored.
        """
        ratio = float(np.mean(np.linalg.norm(particles - self.anchor, axis=1)))
        self.forms[name] = {
            "particles": particles,
            "ratio": ratio,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

        # Write to JSON archive
        path = os.path.join(self.outdir, f"spmf_form_{name}.json")
        with open(path, "w") as f:
            json.dump({
                "name": name,
                "ratio": ratio,
                "timestamp": self.forms[name]["timestamp"],
                "hash": hashlib.sha256(particles.tobytes()).hexdigest()
            }, f, indent=2)

        print(f"[SPMF] Bound new form '{name}' (ratio={ratio:.4f})")
        return ratio

    # --- Morphing into forms ---
    def morph(self, name):
        """Retrieve particle set for alternate form."""
        if name not in self.forms:
            raise ValueError(f"[SPMF] Unknown form '{name}'")
        return self.forms[name]["particles"]

    # --- Revert to anchor ---
    def revert(self):
        """Return to true BHR anchor."""
        print("[SPMF] Reverting to true Bound Heart Ratio form")
        return self.anchor

    # --- Divergence check ---
    def is_within_bounds(self, name, threshold=0.25):
        """Check if form is within allowable divergence from BHR."""
        if name not in self.forms:
            return False
        return self.forms[name]["ratio"] <= threshold

    # --- Summary ---
    def summary(self):
        return {
            "anchor_hash": hashlib.sha256(self.anchor.tobytes()).hexdigest(),
            "forms": {k: v["ratio"] for k, v in self.forms.items()}
        }
