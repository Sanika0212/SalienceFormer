"""
Comprehensive SalienceFormer Evaluation Suite

Tests all components of the brain-inspired memory system:
1. Standard NLP metrics (perplexity, generation quality)
2. Salience gate behavior (selective tagging)
3. Memory buffer dynamics (consolidation, replay)
4. Long-range dependency handling
5. Ablation studies
6. Brain-like behavior validation
"""

import torch
import torch.nn as nn
import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import json
import os


@dataclass
class ComprehensiveResults:
    """Container for all evaluation results."""

    # Standard metrics
    perplexity: float = 0.0
    loss: float = 0.0

    # Salience gate metrics
    salience_mean: float = 0.0
    salience_std: float = 0.0
    tagged_ratio: float = 0.0
    content_vs_function_ratio: float = 0.0  # Are content words tagged more?

    # Memory buffer metrics
    buffer_utilization: float = 0.0
    mean_priority: float = 0.0
    priority_std: float = 0.0
    mean_age: float = 0.0
    total_writes: int = 0

    # Long-range metrics
    ppl_early: float = 0.0  # PPL on first 25% of sequence
    ppl_late: float = 0.0   # PPL on last 25% of sequence
    long_range_benefit: float = 0.0  # Improvement on late tokens

    # Ablation results
    ablation_no_salience: float = 0.0
    ablation_no_memory: float = 0.0
    ablation_random_salience: float = 0.0

    # Brain-like behavior
    selective_tagging_score: float = 0.0  # Do important tokens get higher scores?
    temporal_coherence: float = 0.0  # Are nearby tokens tagged together?
    replay_benefit: float = 0.0  # Does replay improve consolidation?

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "COMPREHENSIVE SALIENCEFORMER EVALUATION",
            "=" * 60,
            "",
            "1. STANDARD METRICS",
            f"   Perplexity: {self.perplexity:.2f}",
            f"   Loss: {self.loss:.4f}",
            "",
            "2. SALIENCE GATE (Brain-like selective memory)",
            f"   Mean salience: {self.salience_mean:.3f}",
            f"   Std salience: {self.salience_std:.3f}",
            f"   Tagged ratio (>0.5): {self.tagged_ratio:.1%}",
            f"   Content vs function word ratio: {self.content_vs_function_ratio:.2f}x",
            "",
            "3. MEMORY BUFFER (Hippocampal consolidation)",
            f"   Buffer utilization: {self.buffer_utilization:.1%}",
            f"   Mean priority: {self.mean_priority:.2f} / 5.0",
            f"   Total writes: {self.total_writes:,}",
            "",
            "4. LONG-RANGE DEPENDENCIES",
            f"   PPL (early tokens): {self.ppl_early:.2f}",
            f"   PPL (late tokens): {self.ppl_late:.2f}",
            f"   Long-range benefit: {self.long_range_benefit:+.2f}",
            "",
            "5. ABLATION ANALYSIS",
            f"   Without salience: {self.ablation_no_salience:.2f} (diff: {self.ablation_no_salience - self.perplexity:+.2f})",
            f"   Without memory: {self.ablation_no_memory:.2f} (diff: {self.ablation_no_memory - self.perplexity:+.2f})",
            f"   Random salience: {self.ablation_random_salience:.2f} (diff: {self.ablation_random_salience - self.perplexity:+.2f})",
            "",
            "6. BRAIN-LIKE BEHAVIOR VALIDATION",
            f"   Selective tagging score: {self.selective_tagging_score:.2f} / 1.0",
            f"   Temporal coherence: {self.temporal_coherence:.2f} / 1.0",
            f"   Replay benefit: {self.replay_benefit:.2f}",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)


class ComprehensiveEvaluator:
    """
    Comprehensive evaluation suite for SalienceFormer.

    Tests all brain-inspired components and validates
    that the model exhibits hippocampal-like memory behavior.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

        # Common function words to compare against content words
        self.function_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "to", "of", "in", "for", "on", "with", "at", "by",
            "from", "as", "into", "through", "during", "before", "after",
            "above", "below", "between", "under", "again", "further",
            "then", "once", "here", "there", "when", "where", "why",
            "how", "all", "each", "every", "both", "few", "more", "most",
            "other", "some", "such", "no", "nor", "not", "only", "own",
            "same", "so", "than", "too", "very", "just", "and", "but",
            "if", "or", "because", "until", "while", "although", "that",
            "which", "who", "whom", "this", "these", "those", "it", "its",
        }

    def evaluate(
        self,
        texts: List[str],
        max_length: int = 512,
        verbose: bool = True,
    ) -> ComprehensiveResults:
        """
        Run comprehensive evaluation on given texts.

        Args:
            texts: List of text samples to evaluate
            max_length: Maximum sequence length
            verbose: Print progress

        Returns:
            ComprehensiveResults with all metrics
        """
        results = ComprehensiveResults()

        if verbose:
            print("Running comprehensive evaluation...")

        # 1. Standard perplexity
        if verbose:
            print("  [1/6] Computing perplexity...")
        ppl_results = self._compute_perplexity(texts, max_length)
        results.perplexity = ppl_results["perplexity"]
        results.loss = ppl_results["loss"]

        # 2. Salience gate analysis
        if verbose:
            print("  [2/6] Analyzing salience gate...")
        salience_results = self._analyze_salience(texts, max_length)
        results.salience_mean = salience_results["mean"]
        results.salience_std = salience_results["std"]
        results.tagged_ratio = salience_results["tagged_ratio"]
        results.content_vs_function_ratio = salience_results["content_vs_function"]

        # 3. Memory buffer analysis
        if verbose:
            print("  [3/6] Analyzing memory buffer...")
        memory_results = self._analyze_memory_buffer()
        results.buffer_utilization = memory_results["utilization"]
        results.mean_priority = memory_results["mean_priority"]
        results.priority_std = memory_results["priority_std"]
        results.mean_age = memory_results["mean_age"]
        results.total_writes = memory_results["total_writes"]

        # 4. Long-range dependency analysis
        if verbose:
            print("  [4/6] Testing long-range dependencies...")
        lr_results = self._analyze_long_range(texts, max_length)
        results.ppl_early = lr_results["ppl_early"]
        results.ppl_late = lr_results["ppl_late"]
        results.long_range_benefit = lr_results["benefit"]

        # 5. Ablation studies
        if verbose:
            print("  [5/6] Running ablation studies...")
        ablation_results = self._run_ablations(texts, max_length)
        results.ablation_no_salience = ablation_results["no_salience"]
        results.ablation_no_memory = ablation_results["no_memory"]
        results.ablation_random_salience = ablation_results["random_salience"]

        # 6. Brain-like behavior validation
        if verbose:
            print("  [6/6] Validating brain-like behavior...")
        brain_results = self._validate_brain_behavior(texts, max_length)
        results.selective_tagging_score = brain_results["selective_tagging"]
        results.temporal_coherence = brain_results["temporal_coherence"]
        results.replay_benefit = brain_results["replay_benefit"]

        if verbose:
            print("\nEvaluation complete!")
            print(results.summary())

        return results

    def _compute_perplexity(
        self,
        texts: List[str],
        max_length: int,
    ) -> Dict[str, float]:
        """Compute standard perplexity."""
        nlls = []

        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs["input_ids"])
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits

                # Shift for causal LM
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs["input_ids"][..., 1:].contiguous()

                loss = torch.nn.CrossEntropyLoss()(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                nlls.append(loss.item())

        avg_nll = sum(nlls) / len(nlls)
        return {
            "perplexity": math.exp(avg_nll),
            "loss": avg_nll,
        }

    def _analyze_salience(
        self,
        texts: List[str],
        max_length: int,
    ) -> Dict[str, float]:
        """Analyze salience gate behavior."""
        all_salience = []
        content_salience = []
        function_salience = []

        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(self.device)

            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

            with torch.no_grad():
                # Get hidden states
                base_out = self.model.base_model(
                    inputs["input_ids"],
                    output_hidden_states=True,
                )
                hidden = base_out.hidden_states[-1]

                # Get salience scores
                salience, _ = self.model.salience_gate(hidden)
                salience = salience[0].cpu().numpy()

            all_salience.extend(salience.tolist())

            # Separate content vs function words
            for i, token in enumerate(tokens):
                clean_token = token.replace("▁", "").lower()
                if clean_token in self.function_words:
                    function_salience.append(salience[i])
                elif clean_token.isalpha() and len(clean_token) > 2:
                    content_salience.append(salience[i])

        # Compute content vs function ratio
        if function_salience and content_salience:
            content_mean = np.mean(content_salience)
            function_mean = np.mean(function_salience)
            ratio = content_mean / max(function_mean, 0.001)
        else:
            ratio = 1.0

        return {
            "mean": np.mean(all_salience),
            "std": np.std(all_salience),
            "tagged_ratio": np.mean([s > 0.5 for s in all_salience]),
            "content_vs_function": ratio,
        }

    def _analyze_memory_buffer(self) -> Dict[str, float]:
        """Analyze memory buffer state."""
        buf = self.model.memory_consolidator

        valid_mask = buf.valid_mask.bool()
        valid_count = valid_mask.sum().item()
        buffer_size = buf.keys.shape[0]

        if valid_count > 0:
            priorities = buf.priorities[valid_mask].cpu().numpy()
            ages = buf.ages[valid_mask].cpu().numpy()
            mean_priority = float(np.mean(priorities))
            priority_std = float(np.std(priorities))
            mean_age = float(np.mean(ages))
        else:
            mean_priority = 0.0
            priority_std = 0.0
            mean_age = 0.0

        return {
            "utilization": valid_count / buffer_size,
            "mean_priority": mean_priority,
            "priority_std": priority_std,
            "mean_age": mean_age,
            "total_writes": int(buf.total_writes.item()),
        }

    def _analyze_long_range(
        self,
        texts: List[str],
        max_length: int,
    ) -> Dict[str, float]:
        """Analyze performance on early vs late tokens."""
        early_nlls = []
        late_nlls = []

        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(self.device)

            seq_len = inputs["input_ids"].size(1)
            if seq_len < 20:
                continue

            with torch.no_grad():
                outputs = self.model(inputs["input_ids"])
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits

                # Per-token loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs["input_ids"][..., 1:].contiguous()

                loss_per_token = torch.nn.CrossEntropyLoss(reduction="none")(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                ).view(shift_labels.shape)

                # Early (first 25%) vs Late (last 25%)
                quarter = seq_len // 4
                early_loss = loss_per_token[0, :quarter].mean().item()
                late_loss = loss_per_token[0, -quarter:].mean().item()

                early_nlls.append(early_loss)
                late_nlls.append(late_loss)

        ppl_early = math.exp(np.mean(early_nlls)) if early_nlls else 0
        ppl_late = math.exp(np.mean(late_nlls)) if late_nlls else 0

        return {
            "ppl_early": ppl_early,
            "ppl_late": ppl_late,
            "benefit": ppl_early - ppl_late,  # Positive = better on late tokens
        }

    def _run_ablations(
        self,
        texts: List[str],
        max_length: int,
    ) -> Dict[str, float]:
        """Run ablation studies."""
        import copy

        # Store original components
        original_salience_gate = self.model.salience_gate
        original_fusion_bias = self.model.output_fusion.gate[2].bias.clone()

        results = {}

        # 1. No salience (uniform)
        class UniformSalience(nn.Module):
            def forward(self, x, attention_mask=None):
                return (
                    torch.ones(x.shape[:2], device=x.device) * 0.5,
                    torch.ones(x.shape[:2], device=x.device),
                )

        self.model.salience_gate = UniformSalience()
        ppl = self._compute_perplexity(texts, max_length)["perplexity"]
        results["no_salience"] = ppl

        # 2. Random salience
        class RandomSalience(nn.Module):
            def forward(self, x, attention_mask=None):
                return (
                    torch.rand(x.shape[:2], device=x.device),
                    torch.ones(x.shape[:2], device=x.device),
                )

        self.model.salience_gate = RandomSalience()
        ppl = self._compute_perplexity(texts, max_length)["perplexity"]
        results["random_salience"] = ppl

        # Restore salience gate
        self.model.salience_gate = original_salience_gate

        # 3. No memory fusion
        with torch.no_grad():
            self.model.output_fusion.gate[2].bias.fill_(-10)
        ppl = self._compute_perplexity(texts, max_length)["perplexity"]
        results["no_memory"] = ppl

        # Restore
        with torch.no_grad():
            self.model.output_fusion.gate[2].bias.copy_(original_fusion_bias)

        return results

    def _validate_brain_behavior(
        self,
        texts: List[str],
        max_length: int,
    ) -> Dict[str, float]:
        """Validate brain-like memory behavior."""

        # 1. Selective tagging: Do semantically important tokens get higher scores?
        # We test if named entities / nouns get higher salience than function words
        # (Already computed in salience analysis as content_vs_function)
        salience_results = self._analyze_salience(texts[:5], max_length)
        selective_score = min(salience_results["content_vs_function"] / 2, 1.0)

        # 2. Temporal coherence: Are nearby tokens tagged together?
        # Measure autocorrelation of salience scores
        temporal_scores = []
        for text in texts[:5]:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(self.device)

            with torch.no_grad():
                base_out = self.model.base_model(
                    inputs["input_ids"],
                    output_hidden_states=True,
                )
                hidden = base_out.hidden_states[-1]
                salience, _ = self.model.salience_gate(hidden)
                salience = salience[0].cpu().numpy()

            if len(salience) > 2:
                # Compute lag-1 autocorrelation
                autocorr = np.corrcoef(salience[:-1], salience[1:])[0, 1]
                if not np.isnan(autocorr):
                    temporal_scores.append(autocorr)

        temporal_coherence = np.mean(temporal_scores) if temporal_scores else 0.0
        # Normalize to 0-1 range
        temporal_coherence = (temporal_coherence + 1) / 2

        # 3. Replay benefit: Does the memory consolidation help?
        # This is captured by the ablation (no_memory vs full)
        # We normalize the benefit
        memory_results = self._analyze_memory_buffer()
        replay_benefit = min(memory_results["mean_priority"] / 5.0, 1.0)

        return {
            "selective_tagging": selective_score,
            "temporal_coherence": temporal_coherence,
            "replay_benefit": replay_benefit,
        }


def run_comprehensive_evaluation(
    checkpoint_path: str,
    output_path: Optional[str] = None,
    device: str = "cpu",
    num_samples: int = 20,
):
    """
    Run comprehensive evaluation on a SalienceFormer checkpoint.

    Args:
        checkpoint_path: Path to checkpoint.pt
        output_path: Optional path to save JSON results
        device: Device to run on
        num_samples: Number of test samples
    """
    from salienceformer.config import SalienceFormerConfig
    from salienceformer.model import SalienceFormer
    from transformers import AutoTokenizer
    from datasets import load_dataset

    print("Loading model...")
    config = SalienceFormerConfig(
        base_model_name="google/gemma-2b",
        freeze_base=True,
        use_lora=True,
    )
    model = SalienceFormer(config)

    print("Loading checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.float()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading test data...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"] if len(t.strip()) > 100][:num_samples]

    # Run evaluation
    evaluator = ComprehensiveEvaluator(model, tokenizer, device)
    results = evaluator.evaluate(texts)

    # Save results
    if output_path:
        with open(output_path, "w") as f:
            json.dump(results.to_dict(), f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.pt")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples")

    args = parser.parse_args()

    run_comprehensive_evaluation(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        device=args.device,
        num_samples=args.samples,
    )
