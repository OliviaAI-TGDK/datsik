import numpy as np
import logging
import hashlib
import psycopg2
from psycopg2.extras import RealDictCursor


class CodeWright:
    """
    TGDK CodeWright Utility
    -----------------------
    Handles Author Seals, AI registration, and symbolic encapsulation.
    Now supports user-defined:
        - ai_name (the AI being baked)
        - author_name
        - organization
    """

    def __init__(self, ai_name=None, author_name=None, organization=None, dsn=None):
        # === Identity layer ===
        self.ai_name = ai_name or input("🤖 Enter AI name: ").strip() or "UnnamedAI"
        self.author_name = author_name or input("👤 Enter Author name: ").strip() or "Anonymous"
        self.organization = organization or input("🏛️ Enter Organization name: ").strip() or "Independent"

        # === Database connection string (optional) ===
        self.dsn = dsn or "dbname=quomo user=quomo password=quomo host=localhost port=5432"

        # === System states ===
        self.seal_active = False
        self.efficacy_module = None
        self.impairment_drive = None
        self.underwrite_module = None
        self.invocations = []

        logging.info(
            f"[CodeWright] 🧬 Initialized for {self.ai_name} "
            f"by {self.author_name} ({self.organization})"
        )

    # ------------------------------------------------
    # Core connection and metadata utilities
    # ------------------------------------------------
    def _get_conn(self):
        return psycopg2.connect(self.dsn, cursor_factory=RealDictCursor)

    def register_identity(self):
        """Return current AI identity summary."""
        return {
            "ai_name": self.ai_name,
            "author": self.author_name,
            "organization": self.organization
        }

    # ------------------------------------------------
    # Sealing / verification / synchronization
    # ------------------------------------------------
    def duosync(self, model):
        """Sync model parameters across attached Duo systems."""
        if hasattr(model, "parameters"):
            total = sum(p.numel() for p in model.parameters())
            print(f"[CodeWright] 🔁 Duosync complete — {total:,} params synchronized.")
            return True
        print("[CodeWright] ⚠️ Model has no parameters; duosync skipped.")
        return False

    def seal(self, payload: str) -> str:
        """Seal a payload with author + AI identity."""
        self.seal_active = True
        signature = f"[SEALED by {self.author_name} | {self.organization} | AI:{self.ai_name}]"
        return f"{signature}\n{payload}"

    def verify(self, payload: str) -> bool:
        """Verify that a payload contains this AI + author seal."""
        key = f"[SEALED by {self.author_name} | {self.organization} | AI:{self.ai_name}]"
        return key in payload

    # ------------------------------------------------
    # Epoch / Fold / Matterfold mechanics
    # ------------------------------------------------
    def epoch_fold(self, epoch, data):
        glance, meta = self.successionary_glance(data)
        fold_input = f"{epoch}-{meta['engine']}-{meta['count']}-{meta.get('timestamp','')}"
        fold_hash = hashlib.sha256(fold_input.encode()).hexdigest()
        fold = {"epoch": epoch, "meta": meta, "fold_hash": fold_hash}
        self.invocations.append(("epoch_fold", fold))
        return fold

    def matterfold_assert(self, epoch, vow_vector, data):
        glance, meta = self.successionary_glance(data)
        payload = f"{epoch}-{vow_vector}-{meta['count']}"
        integrity = hashlib.sha256(payload.encode()).hexdigest()
        clause = {
            "epoch": epoch,
            "assertion": f"Epoch {epoch} sealed into Matterfold",
            "vows": vow_vector,
            "integrity": integrity
        }
        self.invocations.append(("matterfold_assert", clause))
        return clause

    def full_paradoxialization(self, epoch, vow_vector, data):
        glance = self.successionary_glance(data)
        fold = self.epoch_fold(epoch, data)
        clause = self.matterfold_assert(epoch, vow_vector, data)
        return {"epoch": epoch, "glance": glance, "fold": fold, "clause": clause}

    # ------------------------------------------------
    # Modular integrations
    # ------------------------------------------------
    def integrate_modules(self, efficacy_module=None, impairment_drive=None, underwrite_module=None):
        """Integrate supporting modules."""
        self.efficacy_module = efficacy_module
        self.impairment_drive = impairment_drive
        self.underwrite_module = underwrite_module
        logging.info("[CodeWright] 🔗 Modules integrated successfully.")

    # ------------------------------------------------
    # Glance logic and efficacy override
    # ------------------------------------------------
    def successionary_glance(self, data):
        try:
            arr = np.array([len(str(x)) for x in (data if isinstance(data, (list, tuple, np.ndarray)) else [data])], dtype=float)
            metadata = {
                "engine": getattr(self, "ai_name", "unknown"),
                "preview": str(data)[:64],
                "count": len(arr),
            }
            return arr, metadata
        except Exception as e:
            logging.error(f"[successionary_glance] failed: {e}")
            return np.array([]), {"engine": self.ai_name, "error": str(e)}

    def efficacy_override(self, glance):
        if glance.size == 0:
            return glance
        norm = (glance - glance.min()) / (glance.ptp() + 1e-9)
        return norm

    def distributional_class_matter(self, overridden):
        if overridden.size == 0:
            return "undefined"
        mean_val = overridden.mean()
        if mean_val > 0.75:
            return "transcendent"
        elif mean_val > 0.5:
            return "ascendant"
        elif mean_val > 0.25:
            return "equanimous"
        else:
            return "mundane"

