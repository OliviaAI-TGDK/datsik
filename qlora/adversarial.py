# adversarial.py – TGDK Adversarial Maneuver Doctrine
import numpy as np
import logging


class AdversarialManeuver:
    """
    TGDK Adversarial Doctrine Layer
    - Models restraint vs retaliation as a balance of vectors.
    - Response = sqrt(conduct) * restraint * political_efficiency
      until force_vector >= threshold, then counter-maneuver triggers.
    """

    def __init__(self, restraint: float = 1.0, threshold: float = 0.75):
        self.restraint = restraint
        self.threshold = threshold
        self.last_eval = None

    def evaluate(self, force_vector: float, political: float = 1.0, context: str = ""):
        """
        Decide whether to stay restrained or trigger a counter-maneuver.

        Args:
            force_vector (float): Measure of adversarial pressure (0–1+).
            political (float): Scalar of political efficacy (usually 0–1).
            context (str): Optional context string to annotate.

        Returns:
            dict with {mode, conduct, message}
        """
        conduct = np.sqrt(max(force_vector, 0.0)) * self.restraint * political

        if force_vector < self.threshold:
            msg = (
                f"[Restraint Mode] Conduct={conduct:.3f} "
                f"(force={force_vector:.3f}, political={political:.3f}) :: "
                f"Olivia maintains tone, deflects with satire."
            )
            mode = "restraint"
        else:
            msg = (
                f"[Counter-Maneuver] Conduct={conduct:.3f} "
                f"(force={force_vector:.3f}, political={political:.3f}) :: "
                f"Olivia strikes back — mocking / disruptive stance engaged."
            )
            mode = "counter"

        self.last_eval = {
            "mode": mode,
            "conduct": float(conduct),
            "force_vector": float(force_vector),
            "political": float(political),
            "context": context,
            "message": msg,
        }

        logging.info(msg)
        return self.last_eval
