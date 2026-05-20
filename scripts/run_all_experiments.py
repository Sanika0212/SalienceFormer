#!/usr/bin/env python3
"""
SalienceFormer: Full Experiment Suite for NeurIPS/AAAI Submission

This script runs all experiments needed for a top-venue submission:
1. Multiple datasets (WikiText-2, WikiText-103, PG-19)
2. Multiple seeds (3-5 runs)
3. Baseline comparisons (Longformer, base model)
4. Ablation studies
5. Long-context evaluation
6. Downstream tasks (NarrativeQA)

Usage:
    python scripts/run_all_experiments.py --checkpoint <path> --output-dir ./results

    # Quick test
    python scripts/run_all_experiments.py --checkpoint <path> --quick-test

    # Full NeurIPS submission
    python scripts/run_all_experiments.py --checkpoint <path> --full --seeds 5
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_perplexity_evaluation(model, tokenizer, dataset_name, config, device):
    """Run perplexity evaluation on a dataset."""
    from evaluation.datasets import create_eval_dataloader
    from evaluation.metrics import compute_perplexity

    print(f"\n{'='*60}")
    print(f"Evaluating on {dataset_name}")
    print(f"{'='*60}")

    dataloader = create_eval_dataloader(
        dataset_name,
        tokenizer,
        max_samples=config.get("max_samples"),
        batch_size=config.get("batch_size", 4),
        max_seq_length=config.get("max_seq_length", 512),
    )

    ppl = compute_perplexity(model, dataloader, device=device)
    print(f"Perplexity on {dataset_name}: {ppl:.2f}")

    return {"dataset": dataset_name, "perplexity": ppl}


def run_ablation_study(checkpoint_path, output_dir, seeds, device, max_samples=None):
    """Run ablation study with multiple seeds."""
    from evaluation.ablation import AblationRunner

    print(f"\n{'='*60}")
    print(f"Running Ablation Study ({len(seeds)} seeds)")
    print(f"{'='*60}")

    all_results = {}
    variants = [
        "full_model",
        "no_salience",
        "no_memory",
        "random_salience",
        "fixed_salience",
        "buffer_512",
        "buffer_1024",
        "buffer_4096",
        "decay_0.8",
        "decay_0.95",
    ]

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        runner = AblationRunner(
            base_checkpoint=checkpoint_path,
            device=device,
            max_samples=max_samples,
        )

        results = runner.run_ablation_suite(variants=variants)

        for variant, metrics in results.items():
            if variant not in all_results:
                all_results[variant] = []
            all_results[variant].append(metrics)

    # Aggregate results
    summary = {}
    for variant, runs in all_results.items():
        ppls = [r.get("perplexity", float("inf")) for r in runs]
        summary[variant] = {
            "perplexity_mean": np.mean(ppls),
            "perplexity_std": np.std(ppls),
            "perplexity_runs": ppls,
        }

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "ablation_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nAblation Results Summary:")
    print("-" * 50)
    for variant, metrics in sorted(summary.items(), key=lambda x: x[1]["perplexity_mean"]):
        print(f"{variant:20s}: {metrics['perplexity_mean']:.2f} ± {metrics['perplexity_std']:.2f}")

    return summary


def run_baseline_comparison(checkpoint_path, output_dir, datasets, device, max_samples=None):
    """Compare SalienceFormer against baselines."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from salienceformer import SalienceFormer, SalienceFormerConfig
    from evaluation.datasets import create_eval_dataloader
    from evaluation.metrics import compute_perplexity

    print(f"\n{'='*60}")
    print(f"Running Baseline Comparison")
    print(f"{'='*60}")

    results = {}

    # 1. Load SalienceFormer
    print("\nLoading SalienceFormer...")
    config = SalienceFormerConfig(
        base_model_name="google/gemma-2b",
        freeze_base=True,
        use_lora=True,
    )
    salienceformer = SalienceFormer(config)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    salienceformer.load_state_dict(ckpt["model_state_dict"], strict=False)
    salienceformer = salienceformer.to(device)
    salienceformer.eval()

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Baselines to compare
    baselines = {
        "SalienceFormer": salienceformer,
    }

    # Try to load Gemma base
    try:
        print("\nLoading Gemma-2B (base)...")
        gemma_base = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2b",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        gemma_base.eval()
        baselines["Gemma-2B (base)"] = gemma_base
    except Exception as e:
        print(f"Could not load Gemma-2B: {e}")

    # Try to load Mamba (state space model baseline)
    try:
        print("\nLoading Mamba-2.8B...")
        from transformers import AutoModelForCausalLM as MambaLoader
        mamba = MambaLoader.from_pretrained(
            "state-spaces/mamba-2.8b-hf",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(device)
        mamba.eval()
        baselines["Mamba-2.8B"] = mamba
        print("Mamba-2.8B loaded successfully")
    except Exception as e:
        print(f"Could not load Mamba: {e}")
        print("Install with: pip install mamba-ssm")

    # Try to load Transformer-XL
    try:
        print("\nLoading Transformer-XL...")
        from transformers import TransfoXLLMHeadModel, TransfoXLTokenizer
        # Note: Transformer-XL uses its own tokenizer
        # We'll evaluate it separately with its own tokenizer
        txl_model = TransfoXLLMHeadModel.from_pretrained("transfo-xl-wt103")
        txl_model = txl_model.to(device)
        txl_model.eval()
        baselines["Transformer-XL"] = txl_model
        print("Transformer-XL loaded successfully")
    except Exception as e:
        print(f"Could not load Transformer-XL: {e}")

    # Try to load GPT-2 (common baseline)
    try:
        print("\nLoading GPT-2...")
        gpt2 = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            torch_dtype=torch.float32,
        ).to(device)
        gpt2.eval()
        baselines["GPT-2 (124M)"] = gpt2
        print("GPT-2 loaded successfully")
    except Exception as e:
        print(f"Could not load GPT-2: {e}")

    # Try to load GPT-2 Medium
    try:
        print("\nLoading GPT-2 Medium...")
        gpt2_med = AutoModelForCausalLM.from_pretrained(
            "gpt2-medium",
            torch_dtype=torch.float32,
        ).to(device)
        gpt2_med.eval()
        baselines["GPT-2 Medium (355M)"] = gpt2_med
        print("GPT-2 Medium loaded successfully")
    except Exception as e:
        print(f"Could not load GPT-2 Medium: {e}")

    # Try to load Pythia (EleutherAI)
    try:
        print("\nLoading Pythia-2.8B...")
        pythia = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/pythia-2.8b",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        pythia.eval()
        baselines["Pythia-2.8B"] = pythia
        print("Pythia-2.8B loaded successfully")
    except Exception as e:
        print(f"Could not load Pythia: {e}")

    print(f"\n{len(baselines)} models loaded for comparison")

    # 3. Evaluate on each dataset
    for dataset in datasets:
        print(f"\n--- {dataset} ---")
        results[dataset] = {}

        dataloader = create_eval_dataloader(
            dataset,
            tokenizer,
            max_samples=max_samples,
            batch_size=4,
            max_seq_length=512,
        )

        for name, model in baselines.items():
            try:
                ppl = compute_perplexity(model, dataloader, device=device)
                results[dataset][name] = ppl
                print(f"{name:25s}: {ppl:.2f}")
            except Exception as e:
                print(f"{name:25s}: Error - {e}")
                results[dataset][name] = None

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "baseline_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_long_context_evaluation(checkpoint_path, output_dir, device, max_samples=100):
    """Evaluate on long-context benchmarks."""
    from transformers import AutoTokenizer
    from salienceformer import SalienceFormer, SalienceFormerConfig
    from evaluation.datasets import create_eval_dataloader, load_pg19, load_narrativeqa
    from evaluation.metrics import compute_perplexity

    print(f"\n{'='*60}")
    print(f"Running Long-Context Evaluation")
    print(f"{'='*60}")

    # Load model
    config = SalienceFormerConfig(
        base_model_name="google/gemma-2b",
        freeze_base=True,
        use_lora=True,
    )
    model = SalienceFormer(config)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}

    # 1. PG-19 (long books)
    print("\n--- PG-19 (Long-Form Books) ---")
    try:
        from evaluation.datasets import DatasetConfig
        pg19_config = DatasetConfig(
            name="pg19",
            max_seq_length=2048,
            stride=512,
            max_samples=max_samples,
            batch_size=2,
        )
        pg19_loader = load_pg19(tokenizer, pg19_config)
        ppl = compute_perplexity(model, pg19_loader, device=device)
        results["pg19"] = {"perplexity": ppl}
        print(f"PG-19 Perplexity: {ppl:.2f}")
    except Exception as e:
        print(f"PG-19 Error: {e}")
        results["pg19"] = {"error": str(e)}

    # 2. NarrativeQA
    print("\n--- NarrativeQA (Story QA) ---")
    try:
        narrativeqa_config = DatasetConfig(
            name="narrativeqa",
            max_seq_length=2048,
            max_samples=max_samples,
            batch_size=2,
        )
        nqa_data = load_narrativeqa(tokenizer, narrativeqa_config)
        ppl = compute_perplexity(model, nqa_data["dataloader"], device=device)
        results["narrativeqa"] = {"perplexity": ppl, "num_samples": len(nqa_data["questions"])}
        print(f"NarrativeQA Perplexity: {ppl:.2f}")
    except Exception as e:
        print(f"NarrativeQA Error: {e}")
        results["narrativeqa"] = {"error": str(e)}

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "long_context_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_perplexity_by_position(checkpoint_path, output_dir, device, max_samples=500):
    """Analyze perplexity as a function of position in sequence."""
    from transformers import AutoTokenizer
    from salienceformer import SalienceFormer, SalienceFormerConfig
    from evaluation.datasets import create_eval_dataloader

    print(f"\n{'='*60}")
    print(f"Running Perplexity-by-Position Analysis")
    print(f"{'='*60}")

    # Load model
    config = SalienceFormerConfig(
        base_model_name="google/gemma-2b",
        freeze_base=True,
        use_lora=True,
    )
    model = SalienceFormer(config)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use WikiText-2 for position analysis
    dataloader = create_eval_dataloader(
        "wikitext-2",
        tokenizer,
        max_samples=max_samples,
        batch_size=1,
        max_seq_length=512,
    )

    position_losses = {}  # position -> list of losses

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            # Calculate per-position loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            losses = losses.view(shift_labels.size())

            # Accumulate by position
            for pos in range(losses.size(1)):
                if pos not in position_losses:
                    position_losses[pos] = []
                position_losses[pos].append(losses[0, pos].item())

    # Average by position
    position_ppl = {}
    for pos, losses in position_losses.items():
        avg_loss = np.mean(losses)
        position_ppl[pos] = np.exp(avg_loss)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "position_analysis.json"), "w") as f:
        json.dump(position_ppl, f, indent=2)

    # Summary stats
    early_ppl = np.mean([position_ppl[p] for p in range(50)])
    late_ppl = np.mean([position_ppl[p] for p in range(450, 500) if p in position_ppl])

    print(f"\nPosition Analysis:")
    print(f"  Early positions (0-50):   {early_ppl:.2f}")
    print(f"  Late positions (450-500): {late_ppl:.2f}")
    print(f"  Difference:               {late_ppl - early_ppl:+.2f}")

    return position_ppl


def generate_results_tables(output_dir):
    """Generate markdown tables from results."""
    print(f"\n{'='*60}")
    print(f"Generating Results Tables")
    print(f"{'='*60}")

    tables = []

    # Ablation table
    ablation_path = os.path.join(output_dir, "ablation_results.json")
    if os.path.exists(ablation_path):
        with open(ablation_path) as f:
            ablation = json.load(f)

        table = "### Ablation Study Results\n\n"
        table += "| Variant | PPL (mean ± std) | Δ PPL |\n"
        table += "|---------|------------------|-------|\n"

        baseline_ppl = ablation.get("full_model", {}).get("perplexity_mean", 0)

        for variant, metrics in sorted(ablation.items(), key=lambda x: x[1].get("perplexity_mean", float("inf"))):
            mean = metrics.get("perplexity_mean", 0)
            std = metrics.get("perplexity_std", 0)
            delta = mean - baseline_ppl
            delta_str = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
            table += f"| {variant} | {mean:.2f} ± {std:.2f} | {delta_str} |\n"

        tables.append(table)

    # Baseline comparison table
    baseline_path = os.path.join(output_dir, "baseline_comparison.json")
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baselines = json.load(f)

        table = "### Baseline Comparison\n\n"
        table += "| Model | " + " | ".join(baselines.keys()) + " |\n"
        table += "|-------|" + "|".join(["---"] * len(baselines)) + "|\n"

        all_models = set()
        for dataset, models in baselines.items():
            all_models.update(models.keys())

        for model in all_models:
            row = f"| {model} |"
            for dataset in baselines.keys():
                ppl = baselines[dataset].get(model, "N/A")
                if isinstance(ppl, (int, float)):
                    row += f" {ppl:.2f} |"
                else:
                    row += f" {ppl} |"
            table += row + "\n"

        tables.append(table)

    # Write combined tables
    with open(os.path.join(output_dir, "results_tables.md"), "w") as f:
        f.write("# SalienceFormer Experiment Results\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        for table in tables:
            f.write(table + "\n\n")

    print(f"Tables saved to {output_dir}/results_tables.md")


def main():
    parser = argparse.ArgumentParser(description="Run SalienceFormer experiments")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output-dir", type=str, default="./experiment_results", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/mps/auto)")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per dataset")

    # Experiment selection
    parser.add_argument("--quick-test", action="store_true", help="Quick test with minimal samples")
    parser.add_argument("--full", action="store_true", help="Full experiment suite")
    parser.add_argument("--ablation-only", action="store_true", help="Only run ablation")
    parser.add_argument("--baselines-only", action="store_true", help="Only run baselines")
    parser.add_argument("--long-context-only", action="store_true", help="Only run long-context")

    args = parser.parse_args()

    # Auto-detect device
    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    print(f"Device: {args.device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output_dir}")

    # Quick test config
    if args.quick_test:
        args.max_samples = 50
        args.seeds = 1
        datasets = ["wikitext-2"]
    elif args.full:
        args.max_samples = None
        datasets = ["wikitext-2", "wikitext-103"]
    else:
        datasets = ["wikitext-2", "wikitext-103"]
        if args.max_samples is None:
            args.max_samples = 500

    os.makedirs(args.output_dir, exist_ok=True)

    # Run experiments
    start_time = time.time()

    if args.ablation_only:
        run_ablation_study(
            args.checkpoint,
            args.output_dir,
            seeds=list(range(args.seeds)),
            device=args.device,
            max_samples=args.max_samples,
        )
    elif args.baselines_only:
        run_baseline_comparison(
            args.checkpoint,
            args.output_dir,
            datasets=datasets,
            device=args.device,
            max_samples=args.max_samples,
        )
    elif args.long_context_only:
        run_long_context_evaluation(
            args.checkpoint,
            args.output_dir,
            device=args.device,
            max_samples=args.max_samples or 100,
        )
    else:
        # Full suite
        print("\n" + "="*60)
        print("RUNNING FULL EXPERIMENT SUITE")
        print("="*60)

        # 1. Baseline comparison
        run_baseline_comparison(
            args.checkpoint,
            args.output_dir,
            datasets=datasets,
            device=args.device,
            max_samples=args.max_samples,
        )

        # 2. Ablation study
        run_ablation_study(
            args.checkpoint,
            args.output_dir,
            seeds=list(range(args.seeds)),
            device=args.device,
            max_samples=args.max_samples,
        )

        # 3. Long-context evaluation
        run_long_context_evaluation(
            args.checkpoint,
            args.output_dir,
            device=args.device,
            max_samples=args.max_samples or 100,
        )

        # 4. Position analysis
        run_perplexity_by_position(
            args.checkpoint,
            args.output_dir,
            device=args.device,
            max_samples=args.max_samples or 500,
        )

        # 5. Generate tables
        generate_results_tables(args.output_dir)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"All experiments completed in {elapsed/60:.1f} minutes")
    print(f"Results saved to {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
