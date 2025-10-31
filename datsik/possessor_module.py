# =============================================================================
# TGDK BFE Lscience – Volumetric Infinitizer Seal
# License ID: BFE-TGDK-LSCIENCE-092524
# Author: Sean Tichenor ("M", Black Raven)
# Module: Volumetric Infinitizer
# Date of Issue: September 2025
#
# Signed and Sealed
# TGDK LLC
# BFE Vault Scroll Registry
# =============================================================================

import hashlib, secrets, time
import numpy as np
import logging
import tools.sword
from code_wright import CodeWright


class PossessorScript:
    """
    Possessor Script (Flag + Sword Kata Seal + Ratio Paternalizer).
    Required for Olivia training loop — folds params into a figure-8 seed.
    Auto-generates a HexQUAp key at each run for sealing.
    """

    def __init__(self, codewright=None, author="Sean 'M' Tichenor", name="VolumetricInfinitizer"):
        self.author = author
        self.flag = "🇺🇸"
        self.sword = "⚔️"
        self.kata = []
        self.epoch_vector = []
        self.log = [] 

        if isinstance(codewright, str):
            self.codewright = CodeWright(codewright)   # auto-wrap string
        else:
            self.codewright = codewright or CodeWright("M")

        self.figure8_ratio = None
        self.name = name or "VolumetricInfinitizer"

        # 🔑 Generate rotating HexQUAp key
        self.hexquap_key = self._generate_hexquap_key()
        self.seed_hash = self._seal_author(author, self.hexquap_key)

    # -------------------------------------------------------------------------
    # Internal Sealing
    # -------------------------------------------------------------------------
    def _generate_hexquap_key(self):
        """Generate a rotating HexQUAp key based on secrets + timestamp."""
        raw = f"{secrets.token_hex(16)}-{time.time_ns()}"
        digest = hashlib.sha256(raw.encode()).hexdigest()
        return "-".join(digest[i:i+8] for i in range(0, 48, 8))  # 6-fold feel

    def _seal_author(self, author, key):
        """Seal the author’s name against the key."""
        return hashlib.sha256(f"{author}-{key}".encode()).hexdigest()

    # -------------------------------------------------------------------------
    # Ritual Kata + Seals
    # -------------------------------------------------------------------------
    def offer_sword(self):
        """Major semblance mechanism — all flows must call this."""
        kata = f"[Sword Kata] {self.sword} Sword offered, kata sealed."
        self.kata.append(kata)
        logging.info(kata)
        return kata

    def offer_kata(self, phrase: str):
        """Offer a kata phrase to bind into the sword seal."""
        self.offer_sword()
        kata_entry = {
            "timestamp": time.time(),
            "phrase": phrase,
            "seal": f"{self.flag}{self.sword}{self.seed_hash}"
        }
        self.kata.append(kata_entry)
        return kata_entry

    def invoke_flag(self):
        msg = f"[Semblance] {self.flag} American flag invoked as module."
        logging.info(msg)
        return msg

    def summon(self):
        """Summon the seal module as symbolic representation."""
        return f"[{self.flag}] {self.sword} {self.name} (Author: {self.author})"

    # -------------------------------------------------------------------------
    # Core Ratios
    # -------------------------------------------------------------------------
    def paternalize_ratio(self, ratio: float):
        """Apply paternalizer ratio — folds back into figure-8 mechanics."""
        self.offer_sword()
        self.figure8_ratio = (ratio ** 0.5) / (ratio + 1e-9)
        return self.figure8_ratio

    def ratio_paternalizer(self, params):
        """
        Reduce trainable params into compact stats
        (mean, variance, count) without allocating a giant array.
        """
        total_elems = 0
        total_sum = 0.0
        total_sq = 0.0

        for p in params:
            if getattr(p, "requires_grad", False):
                arr = p.detach().cpu().view(-1).double()
                total_elems += arr.numel()
                total_sum += arr.sum().item()
                total_sq += (arr ** 2).sum().item()

        if total_elems == 0:
            return {"mean": 0.0, "variance": 0.0, "count": 0}

        mean = total_sum / total_elems
        var = total_sq / total_elems - mean ** 2
        return {"mean": mean, "variance": var, "count": total_elems}


    def final_offering(self) -> dict:
        """
        🪶 MirrorBlade | Final Offering
        --------------------------------
        Summarizes all epoch results collected in self.epoch_vector
        and returns an averaged summary for final logging or sealing.
        """
        if not hasattr(self, "epoch_vector") or not self.epoch_vector:
            logging.warning("[MirrorBlade] No epoch data available → returning empty final offering.")
            return {
                "epochs": 0,
                "mean_entropy": 0.0,
                "mean_variance": 0.0,
                "mean_bodhicitta": 0.0,
                "status": "empty"
            }

        epochs = len(self.epoch_vector)
        entropy_vals = [e.get("entropy", 0.0) for e in self.epoch_vector]
        variance_vals = [e.get("variance", 0.0) for e in self.epoch_vector]
        bodhi_vals = [e.get("bodhicitta", 0.0) for e in self.epoch_vector]
   
        summary = {
            "epochs": epochs,
            "mean_entropy": float(np.mean(entropy_vals)),
            "mean_variance": float(np.mean(variance_vals)),
            "mean_bodhicitta": float(np.mean(bodhi_vals)),
            "status": "sealed"
        }

        # Optional: store in log for audit trail
        self.log.append(("final_offering", summary))
        logging.info(f"[MirrorBlade] Final offering sealed: {summary}")
          
        return summary



    # -------------------------------------------------------------------------
    # Epoch Paradoxialization
    # -------------------------------------------------------------------------
    def paradoxialize_epoch(self, epoch: int, data=None, state: str = "active", **kwargs):
        """
        🌀 MirrorBlade | Paradoxialize Epoch (No-Init Safe)
        ---------------------------------------------------
        Safely runs paradoxial epoch transformation through the OliviaAI chain.
        Handles None, dict, string, iterable, or ndarray data types gracefully.
        """
        logging.info(f"[MirrorBlade] [Epoch {epoch}] State={state} paradoxialized → initializing...")

        # === Guard 1: handle None ===
        if data is None:
            logging.warning("[MirrorBlade] Received None → substituting zeros.")
            data = np.zeros(7)

        # === Guard 2: handle dict input ===
        elif isinstance(data, dict):
            numeric_values = []
            for k, v in data.items():
                try:
                    if isinstance(v, (int, float)):
                        numeric_values.append(float(v))
                    elif isinstance(v, str) and v.strip():
                        numeric_values.append(float(v))
                    elif isinstance(v, (list, tuple)):
                        numeric_values.extend(
                            float(x) for x in v if isinstance(x, (int, float))
                        )
                except Exception:
                    continue
            if numeric_values:
                data = np.array(numeric_values, dtype=float)
            else:
                logging.warning("[MirrorBlade] Dict contained no numeric data → using zeros.")
                data = np.zeros(7)

        # === Guard 3: handle string input ===
        elif isinstance(data, str):
            parts = [p.strip() for p in data.replace(",", " ").split() if p.strip()]
            try:
                data = np.array([float(p) for p in parts], dtype=float)
            except Exception:
                data = np.zeros(7)

        # === Guard 4: handle iterable input ===
        elif isinstance(data, (list, tuple)):
            clean = []
            for x in data:
                try:
                    if isinstance(x, str) and not x.strip():
                        x = 0.0
                    clean.append(float(x))
                except Exception:
                    clean.append(0.0)
            data = np.array(clean, dtype=float)

        # === Guard 5: enforce ndarray numeric ===
        elif not isinstance(data, np.ndarray):
            try:
                data = np.atleast_1d(np.array(data, dtype=float))
            except Exception as e:
                logging.error(f"[MirrorBlade] Fallback coercion failed: {e}")
                data = np.zeros(7)

        # === Final normalization ===
        data = np.nan_to_num(np.array(data, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        if data.size == 0:
            data = np.zeros(7)

        # === Compute metrics ===
        mechanical_frequency = len(str(data)) % 7
        entropy = float(np.var(data))
        variance = float(np.var(data))
        mean_val = float(np.mean(data))
        bodhicitta = round(abs(mean_val - entropy) * np.pi, 6)

        result = {
            "epoch": epoch,
            "state": state,
            "mechanical_frequency": mechanical_frequency,
            "entropy": entropy,
            "variance": variance,
            "bodhicitta": bodhicitta,
            "mean": mean_val,
            "status": "active",
        }

        # === Lazy-create log containers ===
        if not hasattr(self, "epoch_vector"):
            self.epoch_vector = []
        if not hasattr(self, "log"):
            self.log = []

        self.epoch_vector.append(result)
        self.log.append(("paradoxialize_epoch", result))
        logging.info(f"[MirrorBlade] [Epoch {epoch}] Paradoxialized result: {result}")

        return result

    def accelerate_material_half(self):
        """
        Perform a controlled half-phase material acceleration.

        Features
        --------
        • Automatically creates missing attributes (`mechanical_frequency`, `entropy`, etc.)
        • Prevents divide-by-zero and NaN propagation
        • Emits MirrorBlade-style structured logs
        • Returns a dict of stabilized metrics for CodeWright and telemetry loops
        """
        import datetime, math

        # --- Safe attribute defaults ---
        self.mechanical_frequency = getattr(self, "mechanical_frequency", 1.0)
        self.entropy = getattr(self, "entropy", 0.0)
        self.variance = getattr(self, "variance", 0.0)
        self.state = getattr(self, "state", "active")

        tstamp = datetime.datetime.now().strftime("%H:%M:%S")

        try:
            # --- Core half-phase acceleration logic ---
            prev_freq = self.mechanical_frequency
            prev_entropy = self.entropy

            # Scale frequency up and entropy down
            self.mechanical_frequency = max(prev_freq * 1.5, 1e-6)
            self.entropy = max(prev_entropy * 0.5, 0.0)

            # Recompute variance for monitoring
            self.variance = abs(self.mechanical_frequency - prev_freq)

            result = {
                "epoch": getattr(self, "epoch", None),
                "state": self.state,
                "mechanical_frequency": self.mechanical_frequency,
                "entropy": self.entropy,
                "variance": self.variance,
                "status": "stabilized",
            }

            print(
                f"[MirrorBlade] {tstamp} — accelerate_material_half() "
                f"Δfreq={self.variance:.6f} → freq={self.mechanical_frequency:.6f}, "
                f"entropy={self.entropy:.6f}"
            )

            return result

        except Exception as e:
            print(f"[MirrorBlade] {tstamp} ❌ accelerate_material_half() failed: {type(e).__name__}: {e}")
            # ensure no crash propagates
            return {
                "epoch": getattr(self, "epoch", None),
                "state": "error",
                "mechanical_frequency": 0.0,
                "entropy": 0.0,
                "variance": 0.0,
                "status": "error",
            }


class CategorizedQSQLDatabases:
    """
    TGDK QSQL Database Stub
    ------------------------
    - Categorizes data streams into symbolic QSQL-like tables
    - Currently a placeholder that groups values by mean
    """

    def __init__(self):
        self.tables = {}

    def insert(self, category: str, values: np.ndarray):
        mean_val = float(np.mean(values)) if values.size else 0.0
        self.tables[category] = {"mean": mean_val, "rows": values.tolist()}
        return self.tables[category]

    def fetch(self, category: str):
        return self.tables.get(category, {})


class RoundTableManager:
    """
    TGDK RoundTable Manager
    ------------------------
    - Merges multiple categorical summaries
    - Enforces 'roundtable' clause of balanced deliberation
    """

    def __init__(self):
        self.rounds = []

    def deliberate(self, inputs: dict):
        total = sum(inputs.values()) + 1e-9
        normed = {k: v / total for k, v in inputs.items()}
        self.rounds.append(normed)
        return normed


class SubdivisionioaryPostProcessor:
    """
    TGDK Mara Post-Processor
    -------------------------
    - Combines QSQL database inserts with RoundTable deliberation
    - Used after Mara drive exponential segmentation
    """

    def __init__(self, qsql: CategorizedQSQLDatabases, roundtable: RoundTableManager, predict_factor: float = 1.0):
        self.qsql = qsql
        self.roundtable = roundtable
        self.predict_factor = predict_factor

    def combine_and_process(self, category: str, enhanced: np.ndarray) -> dict:
        if not isinstance(enhanced, np.ndarray):
            enhanced = np.array(enhanced, dtype=float)

        # Step 1: Store in QSQL
        qsql_entry = self.qsql.insert(category, enhanced)

        # Step 2: Build deliberation inputs (variance, sum, max)
        inputs = {
            "variance": float(np.var(enhanced)),
            "sum": float(np.sum(enhanced)),
            "max": float(np.max(enhanced)) if enhanced.size else 0.0,
        }

        # Step 3: Roundtable normalize
        deliberated = self.roundtable.deliberate(inputs)

        # Step 4: Predictive clause
        prediction = float(np.mean(enhanced) * self.predict_factor)

        return {
            "category": category,
            "qsql": qsql_entry,
            "deliberation": deliberated,
            "prediction": prediction,
        }


class QuadroDuoSemisegmentedExpontializerDrive:
    """
    TGDK Mara Drive – QuadroDuoSemisegmentedExpontializer
    ------------------------------------------------------
    - Segments incoming vectors into exponential harmonics
    - Applies dual/quad semi-segmentation
    - Produces enhanced segments for resilience charting
    """

    def __init__(self, scale_factor: float = 1.5):
        self.scale_factor = scale_factor
        self.history = []

    def process_data(self, data: np.ndarray) -> list:
        """
        Segment and exponentially enhance data.
        Returns a list of dict segments with EnhancedSegment.
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=float)

        if data.size == 0:
            return []

        # --- Step 1: Normalize
        norm = data / (np.linalg.norm(data) + 1e-9)

        # --- Step 2: Semi-segmentation (split into 2 then 4)
        half = len(norm) // 2 or 1
        seg1, seg2 = norm[:half], norm[half:]

        quarters = np.array_split(norm, 4)

        # --- Step 3: Exponential enhancement
        enhanced_half = np.exp(seg1 * self.scale_factor)
        enhanced_quarters = [np.tanh(q * self.scale_factor) for q in quarters]

        # --- Step 4: Collect results
        results = []
        results.append({"Segment": "half", "EnhancedSegment": enhanced_half})
        for i, eq in enumerate(enhanced_quarters):
            results.append({"Segment": f"quarter_{i+1}", "EnhancedSegment": eq})

        # --- Log history
        self.history.append({
            "input": data.tolist(),
            "enhanced": [r["EnhancedSegment"].tolist() for r in results]
        })
        return results

class QuadrolateralDiagram:
    """
    TGDK Quadrolateral Diagram Module
    - Encodes fivefold dexterity clause:
        1. Obedience
        2. Submission
        3. Wisdom
        4. Power
        5. Resilience (QNH vitality clause)
    - Produces prosthetic physiological vectors (QNH Network).
    - Now integrates Mara system for resilient charting.
    """

    def __init__(self, pillars=None):
        self.pillars = pillars or ["obedience", "submission", "wisdom", "power", "resilience"]
        self.values = {p: 0.0 for p in self.pillars}
        self.state = {}
        self.log = []

        # --- Wire Mara modules ---
        self.mara_drive = QuadroDuoSemisegmentedExpontializerDrive()
        self.post_processor = SubdivisionioaryPostProcessor(
            CategorizedQSQLDatabases(),
            RoundTableManager(),
            predict_factor=1.2
        )

    def set_value(self, pillar, value):
        if pillar not in self.values:
            raise ValueError(f"Unknown pillar: {pillar}")
        self.values[pillar] = float(value)

    def compute_vector(self):
        vec = np.array(list(self.values.values()), dtype=float)
        norm = vec / (np.linalg.norm(vec) + 1e-9)
        return norm

    def encode_clause(self, **kwargs):
        """
        Encode a variable-length clause vector.
        Supports extra symbolic inputs like bodhicitta, clarity, etc.
        """
        # Fill missing pillars with 0.0
        vec = []
        for p in self.pillars:
            vec.append(float(kwargs.get(p, 0.0)))

        # Allow overflow keys (like bodhicitta)
        extras = {k: float(v) for k, v in kwargs.items() if k not in self.pillars}
        if extras:
            logging.info(f"[Quadrolateral] Extra inputs detected: {extras}")
            vec.extend(extras.values())

        vec = np.array(vec, dtype=float)  
        unit_vec = vec / (np.linalg.norm(vec) + 1e-9)

        self.state["fivefold"] = unit_vec
        logging.info(f"[Quadrolateral] Encoded vector ({len(vec)} dims): {unit_vec}")
        return unit_vec

    def compassionate_return(self, values, weight: float = 1.0):
        """
        Compassionate Return:
        - Accepts a vector of values (list or np.ndarray).
        - Blends them with the encoded fivefold state (if available).
        - Returns a normalized compassion vector + metadata.
        """
        arr = np.array(values, dtype=float)
        if "fivefold" in self.state:
            base = self.state["fivefold"]
            # Pad/truncate to match lengths
            min_len = min(len(base), len(arr))
            blended = (base[:min_len] + arr[:min_len] * weight) / (1.0 + weight)
        else:
            blended = arr

        # Normalize
        compassion_vec = blended / (np.linalg.norm(blended) + 1e-9)

        result = {
            "compassion_vec": compassion_vec.tolist(),
            "mean": float(np.mean(compassion_vec)),
            "variance": float(np.var(compassion_vec)),
            "bodhicitta": float(np.sum(compassion_vec)),
        }

        self.log.append(("compassionate_return", result))
        logging.info(f"[Quadrolateral] Compassionate return → {result}")
        return result


    def qnh_vitals(self, seed: float = 1.0):
        if "fivefold" not in self.state:
            raise RuntimeError("Fivefold clause not encoded yet!")

        base = self.state["fivefold"]
        heart_rate = int(60 + 40 * base[0])     
        pulse = int(80 + 25 * base[1])          
        delta_wave = float(seed * np.mean(base) * 0.21)  

        vitals = {"heart_rate": heart_rate, "pulse": pulse, "delta_wave": delta_wave}
        self.log.append(("qnh_vitals", vitals))
        logging.info(f"[Quadrolateral] QNH vitals generated {vitals}")
        return vitals

    def mara_charting(self, data, category: str = "resilience") -> dict:
        """
        🌊 Mara-Powered Charting (Safe Overhaul)
        ----------------------------------------
        - Segments and enhances data through Mara drive
        - Normalizes EnhancedSegments into a unified vector
        - Aggregates to produce an enhanced summary vector
        - Post-processes via QSQL + roundtable
        - Returns a valid dict result every time
        """
        import numpy as np
        import logging

        logging.info("[Quadrolateral] Running Mara charting...")

        # === Step 1: Segment & process through Mara drive ===
        try:
           processed_segments = self.mara_drive.process_data(data)
        except Exception as e:
           logging.error(f"[Quadrolateral] Mara drive failed: {e}")
           processed_segments = None

        # === Step 2: Ensure we have an iterable ===
        if processed_segments is None:
            logging.warning("[Quadrolateral] Mara drive returned None → using []")
            processed_segments = []

        # === Step 3: Always initialize enhanced_segments ===
        enhanced_segments = []

        for seg in processed_segments:
            try:
                val = seg.get("EnhancedSegment", [])
                arr = np.atleast_1d(np.array(val, dtype=float))
                enhanced_segments.append(arr)
            except Exception as e:
                logging.warning(f"[Quadrolateral] Skipped malformed segment: {e}")

    # === Step 4: Handle empty input ===
        if not enhanced_segments:
            logging.warning("[Quadrolateral] No enhanced segments found; returning empty result.")
            result = {"category": category, "vector": [], "status": "empty"}
            if hasattr(self, "log"):
                self.log.append(("mara_charting", result))
            return result

    # === Step 5: Normalize all segment lengths ===
        valid_lengths = [len(arr) for arr in enhanced_segments if arr.size > 0]
        max_len = max(valid_lengths) if valid_lengths else 1
        padded_segments = []

        for arr in enhanced_segments:
            if arr.size == 0:
                padded = np.zeros(max_len)
            elif len(arr) < max_len:
                padded = np.pad(arr, (0, max_len - len(arr)), constant_values=0.0)
            else:
                padded = arr[:max_len]
            padded_segments.append(padded)
  
    # === Step 6: Aggregate enhanced data ===
        try:
            enhanced_vector = np.mean(np.vstack(padded_segments), axis=0)
        except Exception as e:
            logging.error(f"[Quadrolateral] Aggregation failed: {e}")
            enhanced_vector = np.zeros(max_len)
 
    # === Step 7: Post-process via QSQL + roundtable ===
        try:
            processed_result = self.post_processor.combine_and_process(category, enhanced_vector)
        except Exception as e:
            logging.error(f"[Quadrolateral] Post-processing failed: {e}")
            processed_result = {
                "category": category,
                "vector": enhanced_vector.tolist(),
                "status": "partial",
           }

    # === Step 8: Log + Return ===
        if hasattr(self, "log"):
            self.log.append(("mara_charting", processed_result))
   
        logging.info(f"[Quadrolateral] Mara charting complete → {processed_result.get('status', 'ok')}")
        return processed_result
 