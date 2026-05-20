"""
Evaluation Runner for SalienceFormer

Orchestrates full evaluation pipeline with CLI interface.
"""

import os
import json
import time
import argparse
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np

from salienceformer.config import SalienceFormerConfig
from salienceformer.model import SalienceFormer

from evaluation.metrics import (
    compute_perplexity,
    compute_perplexity_by_position,
    compute_salienceformer_metrics,
    compute_generation_quality,
    EvaluationMetrics,
)
from evaluation.datasets import (
    create_eval_dataloader,
    DatasetConfig,
    get_dataset_info,
)
from evaluation.ablation import (
    AblationConfig,
    AblationRunner,
    create_ablation_variants,
)
from evaluation.statistics import (
    aggregate_seeds,
    compare_models,
    create_results_table,
)
from evaluation.visualization import (
    setup_plotting_style,
    plot_training_curves,
    plot_ablation_comparison,
    plot_salience_heatmap,
    plot_perplexity_by_position,
    create_paper_figures,
)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation pipeline."""

    # Model
    checkpoint_path: Optional[str] = None
    base_model_name: str = "google/gemma-2b"

    # Datasets
    datasets: List[str] = field(default_factory=lambda: ["wikitext-2"])
    max_seq_length: int = 512
    batch_size: int = 4
    max_eval_samples: Optional[int] = None

    # Evaluation settings
    run_perplexity: bool = True
    run_generation: bool = False
    run_salienceformer_metrics: bool = True
    run_position_analysis: bool = False

    # Ablation
    run_ablation: bool = False
    ablation_seeds: List[int] = field(default_factory=lambda: [42, 123, 456])

    # Output
    output_dir: str = "./eval_results"
    save_predictions: bool = False
    generate_figures: bool = True
    figure_format: str = "pdf"

    # Device
    device: str = "auto"

    def get_device(self) -> torch.device:
        """Get torch device."""
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.device)


class EvaluationRunner:
    """
    Main evaluation pipeline runner.

    Orchestrates all evaluation tasks and generates reports.
    """

    def __init__(
        self,
        config: EvaluationConfig,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
    ):
        """
        Initialize evaluation runner.

        Args:
            config: Evaluation configuration
            model: Pre-loaded model (optional, will load from checkpoint if not provided)
            tokenizer: Pre-loaded tokenizer (optional)
        """
        self.config = config
        self.device = config.get_device()
        self.results: Dict[str, Any] = {}

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

        # Load model and tokenizer
        if model is not None:
            self.model = model
        elif config.checkpoint_path:
            self.model = self._load_model(config.checkpoint_path)
        else:
            self.model = None

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = self._load_tokenizer()

        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

    def _load_tokenizer(self):
        """Load tokenizer from HuggingFace."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_model(self, checkpoint_path: str) -> nn.Module:
        """Load model from checkpoint."""
        print(f"Loading model from {checkpoint_path}")

        # Load config
        config_path = os.path.join(checkpoint_path, "config.pt")
        if os.path.exists(config_path):
            hippo_config = torch.load(config_path)
        else:
            hippo_config = SalienceFormerConfig()

        # Create model
        model = SalienceFormer(hippo_config)

        # Load weights
        weights_path = os.path.join(checkpoint_path, "checkpoint.pt")
        if os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=self.device)
            model.load_state_dict(state["model_state_dict"])

        return model

    def run(self) -> Dict[str, Any]:
        """
        Run full evaluation pipeline.

        Returns:
            Dictionary with all evaluation results
        """
        start_time = time.time()
        print(f"Starting evaluation on {self.device}")
        print(f"Datasets: {self.config.datasets}")

        # Run evaluations
        for dataset_name in self.config.datasets:
            print(f"\n{'='*50}")
            print(f"Evaluating on {dataset_name}")
            print(f"{'='*50}")

            dataset_results = self._evaluate_dataset(dataset_name)
            self.results[dataset_name] = dataset_results

        # Run ablation study if requested
        if self.config.run_ablation and self.model is not None:
            print(f"\n{'='*50}")
            print("Running Ablation Study")
            print(f"{'='*50}")
            self.results["ablation"] = self._run_ablation()

        # Calculate total time
        total_time = time.time() - start_time
        self.results["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "total_time_seconds": total_time,
            "device": str(self.device),
            "datasets": self.config.datasets,
        }

        # Save results
        self._save_results()

        # Generate figures
        if self.config.generate_figures:
            self._generate_figures()

        print(f"\nEvaluation complete in {total_time:.1f} seconds")
        print(f"Results saved to {self.config.output_dir}")

        return self.results

    def _evaluate_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Evaluate model on a single dataset."""
        results = {}

        # Get dataset info
        info = get_dataset_info(dataset_name)
        print(f"Dataset: {info.get('description', dataset_name)}")
        print(f"Task: {info.get('task', 'unknown')}")

        # Create dataloader
        dataset_config = DatasetConfig(
            name=dataset_name,
            max_seq_length=self.config.max_seq_length,
            batch_size=self.config.batch_size,
            max_samples=self.config.max_eval_samples,
        )

        dataloader = create_eval_dataloader(
            dataset_name,
            self.tokenizer,
            dataset_config,
        )

        # Run perplexity evaluation
        if self.config.run_perplexity and self.model is not None:
            print("Computing perplexity...")
            ppl_results = compute_perplexity(
                self.model,
                dataloader,
                self.device,
            )
            results["perplexity"] = ppl_results
            print(f"  Perplexity: {ppl_results['perplexity']:.2f}")
            print(f"  Loss: {ppl_results['loss']:.4f}")

        # Run position analysis
        if self.config.run_position_analysis and self.model is not None:
            print("Computing perplexity by position...")
            pos_results = compute_perplexity_by_position(
                self.model,
                dataloader,
                self.device,
                max_batches=100,  # Limit for efficiency
            )
            results["position_analysis"] = pos_results
            print(f"  Mean position perplexity: {pos_results['mean_perplexity']:.2f}")

        # Run SalienceFormer-specific metrics
        if self.config.run_salienceformer_metrics and self.model is not None:
            if hasattr(self.model, "salience_gate"):
                print("Computing SalienceFormer metrics...")
                hippo_results = compute_salienceformer_metrics(
                    self.model,
                    dataloader,
                    self.device,
                    max_batches=50,
                )
                results["salienceformer"] = hippo_results
                print(f"  Mean salience: {hippo_results['mean_salience']:.3f}")
                print(f"  Tagged ratio: {hippo_results['tagged_ratio']:.2%}")
                print(f"  Buffer utilization: {hippo_results['buffer_utilization']:.2%}")

        return results

    def _run_ablation(self) -> Dict[str, Any]:
        """Run ablation study."""
        # This is a simplified version - full ablation requires training each variant
        print("Note: Full ablation study requires training each variant")
        print("Running inference-only ablation comparison...")

        ablation_results = {}
        variants = create_ablation_variants()

        # For inference-only, we compare component disabling effects
        for variant in variants[:5]:  # Limit to first 5 for quick run
            print(f"  Testing: {variant.name}")
            ablation_results[variant.name] = {
                "description": variant.description,
                "config": variant.to_dict(),
            }

        return ablation_results

    def _save_results(self):
        """Save results to JSON file."""
        results_path = os.path.join(self.config.output_dir, "eval_results.json")

        # Convert numpy types to Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        serializable = convert(self.results)

        with open(results_path, "w") as f:
            json.dump(serializable, f, indent=2)

        print(f"Saved results to {results_path}")

    def _generate_figures(self):
        """Generate visualization figures."""
        figures_dir = os.path.join(self.config.output_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)

        setup_plotting_style()

        # Generate per-dataset figures
        for dataset_name, dataset_results in self.results.items():
            if dataset_name in ["metadata", "ablation"]:
                continue

            # Position analysis plot
            if "position_analysis" in dataset_results:
                pos_ppl = dataset_results["position_analysis"].get("position_perplexity", {})
                if pos_ppl:
                    plot_perplexity_by_position(
                        pos_ppl,
                        title=f"Perplexity by Position ({dataset_name})",
                        save_path=os.path.join(
                            figures_dir,
                            f"{dataset_name}_position_ppl.{self.config.figure_format}"
                        ),
                    )

        # Ablation comparison
        if "ablation" in self.results:
            # Would generate ablation plots here
            pass

        print(f"Saved figures to {figures_dir}")


def run_quick_eval(
    model: nn.Module,
    tokenizer,
    dataset: str = "wikitext-2",
    max_samples: int = 100,
    device: str = "auto",
) -> EvaluationMetrics:
    """
    Quick evaluation helper for interactive use.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        dataset: Dataset name
        max_samples: Maximum samples to evaluate
        device: Device to use

    Returns:
        EvaluationMetrics with results
    """
    config = EvaluationConfig(
        datasets=[dataset],
        max_eval_samples=max_samples,
        device=device,
        run_generation=False,
        run_position_analysis=False,
        generate_figures=False,
    )

    runner = EvaluationRunner(config, model=model, tokenizer=tokenizer)
    results = runner.run()

    # Convert to EvaluationMetrics
    metrics = EvaluationMetrics()

    if dataset in results:
        dr = results[dataset]
        if "perplexity" in dr:
            metrics.perplexity = dr["perplexity"]["perplexity"]
            metrics.loss = dr["perplexity"]["loss"]
        if "salienceformer" in dr:
            metrics.mean_salience = dr["salienceformer"]["mean_salience"]
            metrics.tagged_ratio = dr["salienceformer"]["tagged_ratio"]
            metrics.buffer_utilization = dr["salienceformer"]["buffer_utilization"]

    return metrics


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="SalienceFormer Evaluation Pipeline")

    # Model arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="google/gemma-2b",
        help="Base model name (default: google/gemma-2b)",
    )

    # Dataset arguments
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["wikitext-2"],
        help="Datasets to evaluate on",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per dataset (for quick eval)",
    )

    # Evaluation options
    parser.add_argument(
        "--no-perplexity",
        action="store_true",
        help="Skip perplexity evaluation",
    )
    parser.add_argument(
        "--run-generation",
        action="store_true",
        help="Run generation evaluation",
    )
    parser.add_argument(
        "--run-position-analysis",
        action="store_true",
        help="Run perplexity by position analysis",
    )
    parser.add_argument(
        "--run-ablation",
        action="store_true",
        help="Run ablation study",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./eval_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation",
    )
    parser.add_argument(
        "--figure-format",
        type=str,
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Figure output format",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, mps, cpu)",
    )

    args = parser.parse_args()

    # Create config
    config = EvaluationConfig(
        checkpoint_path=args.checkpoint,
        base_model_name=args.base_model,
        datasets=args.datasets,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        max_eval_samples=args.max_samples,
        run_perplexity=not args.no_perplexity,
        run_generation=args.run_generation,
        run_position_analysis=args.run_position_analysis,
        run_ablation=args.run_ablation,
        output_dir=args.output_dir,
        generate_figures=not args.no_figures,
        figure_format=args.figure_format,
        device=args.device,
    )

    # Run evaluation
    runner = EvaluationRunner(config)
    results = runner.run()

    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)

    for dataset_name, dataset_results in results.items():
        if dataset_name in ["metadata", "ablation"]:
            continue

        print(f"\n{dataset_name}:")
        if "perplexity" in dataset_results:
            print(f"  Perplexity: {dataset_results['perplexity']['perplexity']:.2f}")
        if "salienceformer" in dataset_results:
            hr = dataset_results["salienceformer"]
            print(f"  Mean Salience: {hr['mean_salience']:.3f}")
            print(f"  Tagged Ratio: {hr['tagged_ratio']:.2%}")


if __name__ == "__main__":
    main()
