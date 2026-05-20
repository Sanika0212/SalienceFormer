"""
SalienceFormer Evaluation Suite

Comprehensive evaluation infrastructure for research paper publication.
"""

from evaluation.metrics import (
    compute_perplexity,
    compute_bleu,
    compute_rouge,
    compute_f1,
    compute_exact_match,
    EvaluationMetrics,
)
from evaluation.datasets import (
    load_wikitext103,
    load_pg19,
    load_narrativeqa,
    create_eval_dataloader,
    DatasetConfig,
)
from evaluation.ablation import (
    AblationConfig,
    AblationRunner,
    create_ablation_variants,
)
from evaluation.visualization import (
    plot_salience_heatmap,
    plot_memory_dynamics,
    plot_training_curves,
    plot_ablation_comparison,
)
from evaluation.statistics import (
    compute_confidence_interval,
    paired_significance_test,
    aggregate_seeds,
)
from evaluation.runner import (
    EvaluationRunner,
    EvaluationConfig,
)

__all__ = [
    # Metrics
    "compute_perplexity",
    "compute_bleu",
    "compute_rouge",
    "compute_f1",
    "compute_exact_match",
    "EvaluationMetrics",
    # Datasets
    "load_wikitext103",
    "load_pg19",
    "load_narrativeqa",
    "create_eval_dataloader",
    "DatasetConfig",
    # Ablation
    "AblationConfig",
    "AblationRunner",
    "create_ablation_variants",
    # Visualization
    "plot_salience_heatmap",
    "plot_memory_dynamics",
    "plot_training_curves",
    "plot_ablation_comparison",
    # Statistics
    "compute_confidence_interval",
    "paired_significance_test",
    "aggregate_seeds",
    # Runner
    "EvaluationRunner",
    "EvaluationConfig",
]
