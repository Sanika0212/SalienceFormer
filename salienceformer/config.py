"""
SalienceFormer configuration.

All hyperparameters are centralized here, following the pattern from brain/config.py.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class SalienceFormerConfig:
    """Configuration for SalienceFormer model."""

    # Base model settings
    base_model_name: str = "google/gemma-2b"
    hidden_dim: int = 2048  # Will be overridden by base model config
    freeze_base: bool = True
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = ("q_proj", "v_proj")

    # Salience Gate settings (SPW-R analogue)
    salience_n_heads: int = 4
    salience_min_duration: int = 2  # Temporal smoothing kernel size
    salience_threshold_init: float = 0.0  # Learnable threshold initialization
    importance_weight_range: Tuple[float, float] = (2.0, 5.0)  # [lo, hi] for tagged
    untagged_weight: float = 1.0  # Weight for non-salient tokens

    # Memory Consolidator settings (ReplayEngine analogue)
    buffer_size: int = 2048
    decay_rate: float = 0.9  # From brain/config.py
    max_replay_rounds: int = 10
    priority_threshold: float = 1.0  # Only store/replay if weight > this
    soft_topk_temperature: float = 1.0

    # Drift Calibrator settings
    n_anchors: int = 64
    drift_threshold: float = 0.3
    drift_update_momentum: float = 0.99

    # Output fusion settings
    fusion_n_heads: int = 8
    fusion_dropout: float = 0.1

    # Training settings
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # Loss weights
    lm_loss_weight: float = 1.0
    salience_weighted_loss_weight: float = 0.5
    sparsity_loss_weight: float = 0.1
    memory_util_loss_weight: float = 0.01

    # Target metrics (from evaluation/tier2_llm.py)
    consolidation_r_target: float = 0.86

    def __post_init__(self):
        """Validate configuration."""
        assert 0.0 < self.decay_rate < 1.0, "decay_rate must be in (0, 1)"
        assert self.buffer_size > 0, "buffer_size must be positive"
        assert self.max_replay_rounds > 0, "max_replay_rounds must be positive"
        assert self.importance_weight_range[0] < self.importance_weight_range[1], \
            "importance_weight_range[0] must be less than [1]"
        assert self.priority_threshold >= 0, "priority_threshold must be non-negative"
