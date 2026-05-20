"""
Ablation Study Framework for SalienceFormer

Systematic component isolation and comparative experiments.
"""

import copy
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn as nn

from salienceformer.config import SalienceFormerConfig


class AblationType(Enum):
    """Types of ablation experiments."""

    REMOVE_SALIENCE = "no_salience"
    REMOVE_MEMORY = "no_memory"
    REMOVE_DRIFT = "no_drift"
    RANDOM_SALIENCE = "random_salience"
    FIXED_SALIENCE = "fixed_salience"
    VARY_BUFFER_SIZE = "vary_buffer"
    VARY_DECAY_RATE = "vary_decay"
    VARY_IMPORTANCE_RANGE = "vary_importance"
    BASE_MODEL_ONLY = "base_only"
    LORA_ONLY = "lora_only"


@dataclass
class AblationConfig:
    """Configuration for an ablation experiment."""

    name: str
    ablation_type: AblationType
    description: str

    # Component disabling
    disable_salience_gate: bool = False
    disable_memory_buffer: bool = False
    disable_drift_calibrator: bool = False

    # Salience modifications
    use_random_salience: bool = False
    fixed_salience_value: Optional[float] = None

    # Hyperparameter variations
    buffer_size: Optional[int] = None
    decay_rate: Optional[float] = None
    importance_range: Optional[tuple] = None

    # Model modifications
    freeze_hippo_components: bool = False
    use_base_model_only: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "ablation_type": self.ablation_type.value,
            "description": self.description,
            "disable_salience_gate": self.disable_salience_gate,
            "disable_memory_buffer": self.disable_memory_buffer,
            "disable_drift_calibrator": self.disable_drift_calibrator,
            "use_random_salience": self.use_random_salience,
            "fixed_salience_value": self.fixed_salience_value,
            "buffer_size": self.buffer_size,
            "decay_rate": self.decay_rate,
            "importance_range": self.importance_range,
        }


def create_ablation_variants() -> List[AblationConfig]:
    """
    Create standard set of ablation configurations for paper.

    Returns:
        List of AblationConfig for each experiment variant
    """
    variants = [
        # Full model (baseline)
        AblationConfig(
            name="full_salienceformer",
            ablation_type=AblationType.REMOVE_SALIENCE,  # placeholder
            description="Full SalienceFormer with all components",
            disable_salience_gate=False,
            disable_memory_buffer=False,
            disable_drift_calibrator=False,
        ),

        # Component removal ablations
        AblationConfig(
            name="no_salience_gate",
            ablation_type=AblationType.REMOVE_SALIENCE,
            description="SalienceFormer without salience gate (uniform importance)",
            disable_salience_gate=True,
        ),

        AblationConfig(
            name="no_memory_buffer",
            ablation_type=AblationType.REMOVE_MEMORY,
            description="SalienceFormer without memory consolidation buffer",
            disable_memory_buffer=True,
        ),

        AblationConfig(
            name="no_drift_calibrator",
            ablation_type=AblationType.REMOVE_DRIFT,
            description="SalienceFormer without embedding drift calibration",
            disable_drift_calibrator=True,
        ),

        # Salience ablations
        AblationConfig(
            name="random_salience",
            ablation_type=AblationType.RANDOM_SALIENCE,
            description="Random salience scores instead of learned",
            use_random_salience=True,
        ),

        AblationConfig(
            name="fixed_salience_0.5",
            ablation_type=AblationType.FIXED_SALIENCE,
            description="Fixed salience score of 0.5 for all tokens",
            fixed_salience_value=0.5,
        ),

        # Buffer size ablations
        AblationConfig(
            name="buffer_512",
            ablation_type=AblationType.VARY_BUFFER_SIZE,
            description="Small memory buffer (512 entries)",
            buffer_size=512,
        ),

        AblationConfig(
            name="buffer_1024",
            ablation_type=AblationType.VARY_BUFFER_SIZE,
            description="Medium memory buffer (1024 entries)",
            buffer_size=1024,
        ),

        AblationConfig(
            name="buffer_4096",
            ablation_type=AblationType.VARY_BUFFER_SIZE,
            description="Large memory buffer (4096 entries)",
            buffer_size=4096,
        ),

        # Decay rate ablations
        AblationConfig(
            name="decay_0.8",
            ablation_type=AblationType.VARY_DECAY_RATE,
            description="Fast decay rate (0.8)",
            decay_rate=0.8,
        ),

        AblationConfig(
            name="decay_0.95",
            ablation_type=AblationType.VARY_DECAY_RATE,
            description="Slow decay rate (0.95)",
            decay_rate=0.95,
        ),

        AblationConfig(
            name="decay_0.99",
            ablation_type=AblationType.VARY_DECAY_RATE,
            description="Very slow decay rate (0.99)",
            decay_rate=0.99,
        ),

        # Importance range ablations
        AblationConfig(
            name="importance_1_3",
            ablation_type=AblationType.VARY_IMPORTANCE_RANGE,
            description="Narrow importance range [1, 3]",
            importance_range=(1.0, 3.0),
        ),

        AblationConfig(
            name="importance_1_10",
            ablation_type=AblationType.VARY_IMPORTANCE_RANGE,
            description="Wide importance range [1, 10]",
            importance_range=(1.0, 10.0),
        ),

        # Base model comparison
        AblationConfig(
            name="base_model_only",
            ablation_type=AblationType.BASE_MODEL_ONLY,
            description="Base Gemma model without any SalienceFormer components",
            use_base_model_only=True,
        ),

        AblationConfig(
            name="base_with_lora",
            ablation_type=AblationType.LORA_ONLY,
            description="Base model with LoRA but no SalienceFormer components",
            use_base_model_only=True,
            disable_salience_gate=True,
            disable_memory_buffer=True,
            disable_drift_calibrator=True,
        ),
    ]

    return variants


class DisabledModule(nn.Module):
    """Placeholder module that passes input through unchanged."""

    def __init__(self, return_format: str = "tensor"):
        super().__init__()
        self.return_format = return_format

    def forward(self, x, *args, **kwargs):
        if self.return_format == "tuple":
            # For salience gate: return (salience, importance_weights)
            batch, seq = x.shape[:2]
            ones = torch.ones(batch, seq, device=x.device)
            return ones * 0.5, ones  # Uniform salience and weights
        elif self.return_format == "dict":
            return {"output": x}
        else:
            return x


class RandomSalienceModule(nn.Module):
    """Module that returns random salience scores."""

    def forward(self, x, *args, **kwargs):
        batch, seq = x.shape[:2]
        salience = torch.rand(batch, seq, device=x.device)
        importance = torch.ones(batch, seq, device=x.device)
        return salience, importance


class FixedSalienceModule(nn.Module):
    """Module that returns fixed salience scores."""

    def __init__(self, value: float = 0.5):
        super().__init__()
        self.value = value

    def forward(self, x, *args, **kwargs):
        batch, seq = x.shape[:2]
        salience = torch.full((batch, seq), self.value, device=x.device)
        importance = torch.ones(batch, seq, device=x.device)
        return salience, importance


def apply_ablation(model, ablation_config: AblationConfig):
    """
    Apply ablation configuration to a SalienceFormer model.

    Modifies the model in-place by replacing or disabling components.

    Args:
        model: SalienceFormer model instance
        ablation_config: Configuration specifying what to ablate

    Returns:
        Modified model
    """
    # Disable salience gate
    if ablation_config.disable_salience_gate:
        if hasattr(model, "salience_gate"):
            model.salience_gate = DisabledModule(return_format="tuple")

    # Disable memory buffer
    if ablation_config.disable_memory_buffer:
        if hasattr(model, "memory_buffer"):
            # Replace with pass-through
            original_forward = model.forward

            def no_memory_forward(self, *args, **kwargs):
                kwargs["use_memory"] = False
                return original_forward(*args, **kwargs)

            model._ablation_no_memory = True

    # Disable drift calibrator
    if ablation_config.disable_drift_calibrator:
        if hasattr(model, "drift_calibrator"):
            model.drift_calibrator = DisabledModule()

    # Random salience
    if ablation_config.use_random_salience:
        if hasattr(model, "salience_gate"):
            model.salience_gate = RandomSalienceModule()

    # Fixed salience
    if ablation_config.fixed_salience_value is not None:
        if hasattr(model, "salience_gate"):
            model.salience_gate = FixedSalienceModule(
                ablation_config.fixed_salience_value
            )

    return model


def create_ablated_config(
    base_config: SalienceFormerConfig,
    ablation_config: AblationConfig,
) -> SalienceFormerConfig:
    """
    Create a new SalienceFormerConfig with ablation modifications.

    Args:
        base_config: Original configuration
        ablation_config: Ablation to apply

    Returns:
        Modified configuration
    """
    # Deep copy the config
    new_config = copy.deepcopy(base_config)

    # Apply hyperparameter modifications
    if ablation_config.buffer_size is not None:
        new_config.buffer_size = ablation_config.buffer_size

    if ablation_config.decay_rate is not None:
        new_config.decay_rate = ablation_config.decay_rate

    if ablation_config.importance_range is not None:
        new_config.importance_weight_range = ablation_config.importance_range

    return new_config


@dataclass
class AblationResult:
    """Results from a single ablation experiment."""

    config: AblationConfig
    metrics: Dict[str, float]
    seed: int
    training_history: Optional[Dict[str, List[float]]] = None


class AblationRunner:
    """
    Runner for systematic ablation experiments.

    Manages multiple ablation variants, training, and evaluation.
    """

    def __init__(
        self,
        base_config: SalienceFormerConfig,
        model_factory: Callable[[SalienceFormerConfig], nn.Module],
        train_fn: Callable,
        eval_fn: Callable,
        seeds: List[int] = None,
        output_dir: str = "./ablation_results",
    ):
        """
        Initialize ablation runner.

        Args:
            base_config: Base SalienceFormer configuration
            model_factory: Function that creates model from config
            train_fn: Function to train model: (model, config) -> history
            eval_fn: Function to evaluate model: (model) -> metrics dict
            seeds: Random seeds for multiple runs
            output_dir: Directory to save results
        """
        self.base_config = base_config
        self.model_factory = model_factory
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.seeds = seeds or [42, 123, 456]
        self.output_dir = output_dir

        self.results: List[AblationResult] = []

    def run_single(
        self,
        ablation_config: AblationConfig,
        seed: int,
    ) -> AblationResult:
        """
        Run a single ablation experiment.

        Args:
            ablation_config: Ablation configuration
            seed: Random seed

        Returns:
            AblationResult with metrics
        """
        # Set seed
        torch.manual_seed(seed)

        # Create modified config
        modified_config = create_ablated_config(self.base_config, ablation_config)

        # Create model
        if ablation_config.use_base_model_only:
            # Just use base model without SalienceFormer wrapper
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                modified_config.base_model_name
            )
        else:
            model = self.model_factory(modified_config)
            model = apply_ablation(model, ablation_config)

        # Train
        history = self.train_fn(model, modified_config)

        # Evaluate
        metrics = self.eval_fn(model)

        result = AblationResult(
            config=ablation_config,
            metrics=metrics,
            seed=seed,
            training_history=history,
        )

        self.results.append(result)
        return result

    def run_all(
        self,
        ablation_configs: Optional[List[AblationConfig]] = None,
        verbose: bool = True,
    ) -> List[AblationResult]:
        """
        Run all ablation experiments with multiple seeds.

        Args:
            ablation_configs: List of ablations (default: standard set)
            verbose: Print progress

        Returns:
            List of all results
        """
        if ablation_configs is None:
            ablation_configs = create_ablation_variants()

        total = len(ablation_configs) * len(self.seeds)
        current = 0

        for ablation in ablation_configs:
            for seed in self.seeds:
                current += 1
                if verbose:
                    print(f"[{current}/{total}] Running {ablation.name} (seed={seed})")

                result = self.run_single(ablation, seed)

                if verbose:
                    print(f"  Perplexity: {result.metrics.get('perplexity', 'N/A'):.2f}")

        return self.results

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics across seeds.

        Returns:
            Dictionary mapping ablation name to aggregated metrics
        """
        import numpy as np

        # Group by ablation name
        grouped = {}
        for result in self.results:
            name = result.config.name
            if name not in grouped:
                grouped[name] = []
            grouped[name].append(result.metrics)

        # Aggregate
        summary = {}
        for name, metrics_list in grouped.items():
            summary[name] = {}

            # Get all metric keys
            all_keys = set()
            for m in metrics_list:
                all_keys.update(m.keys())

            for key in all_keys:
                values = [m.get(key) for m in metrics_list if m.get(key) is not None]
                if values:
                    summary[name][f"{key}_mean"] = np.mean(values)
                    summary[name][f"{key}_std"] = np.std(values)

        return summary

    def save_results(self, filename: str = "ablation_results.json"):
        """Save results to JSON file."""
        import json
        import os

        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, filename)

        data = {
            "results": [
                {
                    "config": r.config.to_dict(),
                    "metrics": r.metrics,
                    "seed": r.seed,
                }
                for r in self.results
            ],
            "summary": self.get_summary(),
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Saved ablation results to {path}")
