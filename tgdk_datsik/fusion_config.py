"""
TGDK | OliviaAI Defense Configuration Layer
BertMistralFusionConfig – Operational configuration and defense clause metrics
-------------------------------------------------------------------------------

This class defines tunable parameters governing model fusion logic, 
predictive override behavior, and clause-based vector defense responses.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class OperatorClauseMetrics:
    """Core metrics defining operational clause weights and entropy alignment."""
    fusion_ratio: float = 0.618  # golden ratio alignment between models
    defense_vector_strength: float = 0.82  # probability modifier for anti-leak countermeasures
    predictive_override_threshold: float = 0.75  # when predictive module overrides internal fusion
    reverse_engineering_resistance: float = 0.91  # resilience against probing or model tracing
    entropy_stabilization_rate: float = 0.33  # noise floor for gradient equilibrium
    reflection_coefficient: float = 0.144  # symbolic reference constant (phi / π approx)
    clause_bias_gate: float = 1.00  # maximum permissible deviation in symbolic clause synthesis


@dataclass
class FulfillmentScope:
    """Defines the adaptive performance, security, and override domains."""
    enable_predictive_analysis: bool = True
    enable_reverse_engineering_defense: bool = True
    enable_entropy_projection: bool = True
    enable_symbolic_alignment: bool = True
    enable_clause_interlock: bool = True

    clause_sync_depth: int = 3  # number of recursive passes for clause equivalence checks
    defense_vector_mode: str = "mirrorblade"  # ['mirrorblade', 'zengarden', 'passive']
    predictive_scope_mode: str = "forward-anticipatory"  # defines pre-training prediction bias
    override_policy: str = "adaptive"  # ['strict', 'adaptive', 'manual']
    entropy_bandwidth: float = 0.128  # allowable oscillation in entropy feedback loop


@dataclass
class BertMistralFusionConfig:
    """Unified configuration for Bert+Mistral hybrid fusion models."""
    fusion_dim: int = 1024
    projection_dim_bert: int = 768
    projection_dim_mistral: int = 4096
    output_dim: int = 768

    # Core operational metrics
    operator_metrics: OperatorClauseMetrics = field(default_factory=OperatorClauseMetrics)
    fulfillment_scope: FulfillmentScope = field(default_factory=FulfillmentScope)

    # Defense override flags
    enable_entropy_lock: bool = True
    enable_gradient_checkpointing: bool = True
    enable_amp: bool = True
    allow_distributed: bool = False
    device_target: str = "cuda"

    # Predictive and introspection parameters
    predictive_alpha: float = 0.144
    predictive_beta: float = 0.272
    override_penalty_weight: float = 0.05
    max_fusion_layers: int = 12
    mirrorblade_channel: Optional[str] = "TGDK-Defense"

    # Logging and mirror feedback hooks
    enable_logging: bool = True
    log_prefix: str = "[FusionConfig]"
    mirror_sync_interval: int = 60  # seconds between MirrorBlade clause syncs

    def describe(self) -> Dict[str, Any]:
        """Return a nested summary of the configuration for runtime inspection."""
        return {
            "fusion_dim": self.fusion_dim,
            "projection_dims": {
                "bert": self.projection_dim_bert,
                "mistral": self.projection_dim_mistral,
            },
            "output_dim": self.output_dim,
            "operator_metrics": vars(self.operator_metrics),
            "fulfillment_scope": vars(self.fulfillment_scope),
            "device": self.device_target,
            "defense": {
                "entropy_lock": self.enable_entropy_lock,
                "amp": self.enable_amp,
                "distributed": self.allow_distributed,
            },
            "predictive_overrides": {
                "alpha": self.predictive_alpha,
                "beta": self.predictive_beta,
                "penalty": self.override_penalty_weight,
            },
        }

    def summary(self) -> str:
        """Compact human-readable summary."""
        return (
            f"{self.log_prefix} Fusion={self.fusion_dim} | "
            f"DefenseMode={self.fulfillment_scope.defense_vector_mode} | "
            f"Predictive={self.fulfillment_scope.enable_predictive_analysis} | "
            f"EntropyLock={self.enable_entropy_lock} | "
            f"AMP={self.enable_amp}"
        )
