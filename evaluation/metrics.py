"""
Evaluation Metrics for SalienceFormer

Standard NLP metrics for language modeling and downstream tasks.
"""

import math
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@dataclass
class EvaluationMetrics:
    """Container for evaluation results."""

    perplexity: Optional[float] = None
    loss: Optional[float] = None
    bleu: Optional[float] = None
    rouge_1: Optional[float] = None
    rouge_2: Optional[float] = None
    rouge_l: Optional[float] = None
    f1: Optional[float] = None
    exact_match: Optional[float] = None

    # SalienceFormer-specific metrics
    mean_salience: Optional[float] = None
    tagged_ratio: Optional[float] = None
    buffer_utilization: Optional[float] = None
    consolidation_r: Optional[float] = None

    # Timing
    inference_time_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def __str__(self) -> str:
        """Pretty print metrics."""
        lines = []
        for k, v in self.to_dict().items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")
        return "EvaluationMetrics:\n" + "\n".join(lines)


def compute_perplexity(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute perplexity on a dataset.

    Args:
        model: SalienceFormer or any causal LM
        dataloader: DataLoader with input_ids and attention_mask
        device: Device to run on
        max_batches: Limit number of batches (for quick eval)

    Returns:
        Dictionary with perplexity and loss
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # Shift for causal LM: predict next token
            labels = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()
            if attention_mask is not None:
                attention_mask = attention_mask[:, :-1].contiguous()

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # Get logits - handle both SalienceFormer and HF model outputs
            if isinstance(outputs, dict):
                logits = outputs.get("logits", outputs.get("output"))
            elif hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs

            # Compute cross-entropy loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction="sum",
                ignore_index=-100,
            )

            # Count non-padding tokens
            if attention_mask is not None:
                n_tokens = attention_mask.sum().item()
            else:
                n_tokens = labels.numel()

            total_loss += loss.item()
            total_tokens += n_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")

    return {
        "perplexity": perplexity,
        "loss": avg_loss,
        "total_tokens": total_tokens,
    }


def compute_perplexity_by_position(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute perplexity broken down by position in sequence.

    Useful for analyzing how well the model handles long-range dependencies.

    Returns:
        Dictionary with per-position losses and overall perplexity
    """
    model.eval()
    position_losses = {}  # position -> (total_loss, count)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            labels = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            if isinstance(outputs, dict):
                logits = outputs.get("logits", outputs.get("output"))
            elif hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs

            # Per-token loss
            loss_per_token = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction="none",
            ).view(labels.shape)

            # Aggregate by position
            for pos in range(loss_per_token.size(1)):
                if pos not in position_losses:
                    position_losses[pos] = [0.0, 0]

                if attention_mask is not None:
                    mask = attention_mask[:, pos]
                    position_losses[pos][0] += (loss_per_token[:, pos] * mask).sum().item()
                    position_losses[pos][1] += mask.sum().item()
                else:
                    position_losses[pos][0] += loss_per_token[:, pos].sum().item()
                    position_losses[pos][1] += loss_per_token.size(0)

    # Compute per-position perplexity
    position_ppl = {}
    for pos, (total, count) in position_losses.items():
        if count > 0:
            avg_loss = total / count
            position_ppl[pos] = math.exp(avg_loss) if avg_loss < 100 else float("inf")

    return {
        "position_perplexity": position_ppl,
        "mean_perplexity": np.mean(list(position_ppl.values())),
    }


def _get_ngrams(tokens: List[str], n: int) -> Counter:
    """Extract n-grams from token list."""
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def compute_bleu(
    predictions: List[str],
    references: List[List[str]],
    max_n: int = 4,
    smoothing: bool = True,
) -> Dict[str, float]:
    """
    Compute BLEU score.

    Args:
        predictions: List of predicted strings
        references: List of reference string lists (multiple refs per pred)
        max_n: Maximum n-gram order
        smoothing: Apply smoothing for zero counts

    Returns:
        Dictionary with BLEU-1 through BLEU-n and overall BLEU
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have same length")

    # Tokenize (simple whitespace tokenization)
    pred_tokens = [p.lower().split() for p in predictions]
    ref_tokens = [[r.lower().split() for r in refs] for refs in references]

    precisions = []

    for n in range(1, max_n + 1):
        matches = 0
        total = 0

        for pred, refs in zip(pred_tokens, ref_tokens):
            pred_ngrams = _get_ngrams(pred, n)

            # Max count from any reference
            max_ref_counts = Counter()
            for ref in refs:
                ref_ngrams = _get_ngrams(ref, n)
                for ngram, count in ref_ngrams.items():
                    max_ref_counts[ngram] = max(max_ref_counts[ngram], count)

            # Clipped counts
            for ngram, count in pred_ngrams.items():
                matches += min(count, max_ref_counts[ngram])
            total += sum(pred_ngrams.values())

        if total == 0:
            precision = 0.0
        else:
            precision = matches / total
            if smoothing and precision == 0:
                precision = 1e-10  # Smoothing for zero precision

        precisions.append(precision)

    # Brevity penalty
    pred_len = sum(len(p) for p in pred_tokens)
    ref_len = sum(
        min(len(r) for r in refs)
        for refs in ref_tokens
    )

    if pred_len >= ref_len:
        bp = 1.0
    elif pred_len == 0:
        bp = 0.0
    else:
        bp = math.exp(1 - ref_len / pred_len)

    # Geometric mean of precisions
    if all(p > 0 for p in precisions):
        log_precisions = [math.log(p) for p in precisions]
        bleu = bp * math.exp(sum(log_precisions) / len(log_precisions))
    else:
        bleu = 0.0

    result = {"bleu": bleu, "brevity_penalty": bp}
    for i, p in enumerate(precisions, 1):
        result[f"bleu_{i}"] = p

    return result


def _lcs_length(x: List[str], y: List[str]) -> int:
    """Compute longest common subsequence length."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i-1] == y[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]


def compute_rouge(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).

    Args:
        predictions: List of predicted strings
        references: List of reference strings

    Returns:
        Dictionary with ROUGE-1, ROUGE-2, ROUGE-L F1 scores
    """
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []

    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()

        # ROUGE-1 (unigrams)
        pred_1 = _get_ngrams(pred_tokens, 1)
        ref_1 = _get_ngrams(ref_tokens, 1)
        common_1 = sum((pred_1 & ref_1).values())

        if sum(pred_1.values()) > 0 and sum(ref_1.values()) > 0:
            p1 = common_1 / sum(pred_1.values())
            r1 = common_1 / sum(ref_1.values())
            f1_1 = 2 * p1 * r1 / (p1 + r1) if (p1 + r1) > 0 else 0
        else:
            f1_1 = 0
        rouge_1_scores.append(f1_1)

        # ROUGE-2 (bigrams)
        if len(pred_tokens) >= 2 and len(ref_tokens) >= 2:
            pred_2 = _get_ngrams(pred_tokens, 2)
            ref_2 = _get_ngrams(ref_tokens, 2)
            common_2 = sum((pred_2 & ref_2).values())

            if sum(pred_2.values()) > 0 and sum(ref_2.values()) > 0:
                p2 = common_2 / sum(pred_2.values())
                r2 = common_2 / sum(ref_2.values())
                f1_2 = 2 * p2 * r2 / (p2 + r2) if (p2 + r2) > 0 else 0
            else:
                f1_2 = 0
        else:
            f1_2 = 0
        rouge_2_scores.append(f1_2)

        # ROUGE-L (LCS)
        lcs = _lcs_length(pred_tokens, ref_tokens)
        if len(pred_tokens) > 0 and len(ref_tokens) > 0:
            p_l = lcs / len(pred_tokens)
            r_l = lcs / len(ref_tokens)
            f1_l = 2 * p_l * r_l / (p_l + r_l) if (p_l + r_l) > 0 else 0
        else:
            f1_l = 0
        rouge_l_scores.append(f1_l)

    return {
        "rouge_1": np.mean(rouge_1_scores),
        "rouge_2": np.mean(rouge_2_scores),
        "rouge_l": np.mean(rouge_l_scores),
    }


def compute_f1(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute token-level F1 score (used for QA tasks).

    Args:
        predictions: List of predicted answer strings
        references: List of reference answer strings

    Returns:
        Dictionary with precision, recall, and F1
    """
    f1_scores = []
    precision_scores = []
    recall_scores = []

    for pred, ref in zip(predictions, references):
        pred_tokens = set(pred.lower().split())
        ref_tokens = set(ref.lower().split())

        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            f1_scores.append(0.0)
            precision_scores.append(0.0)
            recall_scores.append(0.0)
            continue

        common = pred_tokens & ref_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)

    return {
        "f1": np.mean(f1_scores),
        "precision": np.mean(precision_scores),
        "recall": np.mean(recall_scores),
    }


def compute_exact_match(
    predictions: List[str],
    references: List[str],
    normalize: bool = True,
) -> Dict[str, float]:
    """
    Compute exact match accuracy.

    Args:
        predictions: List of predicted strings
        references: List of reference strings
        normalize: Lowercase and strip whitespace

    Returns:
        Dictionary with exact match score
    """
    matches = 0

    for pred, ref in zip(predictions, references):
        if normalize:
            pred = pred.lower().strip()
            ref = ref.lower().strip()

        if pred == ref:
            matches += 1

    return {
        "exact_match": matches / len(predictions) if predictions else 0.0,
        "total": len(predictions),
        "matches": matches,
    }


def compute_salienceformer_metrics(
    model,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute SalienceFormer-specific metrics.

    Args:
        model: SalienceFormer model
        dataloader: Evaluation dataloader
        device: Device to run on
        max_batches: Limit batches for quick eval

    Returns:
        Dictionary with salience, memory, and consolidation metrics
    """
    model.eval()

    all_salience = []
    all_tagged_ratio = []
    all_buffer_util = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_salience=True,
                return_memory_stats=True,
            )

            if "salience_stats" in outputs:
                stats = outputs["salience_stats"]
                all_salience.append(stats.get("mean_salience", 0))
                all_tagged_ratio.append(stats.get("tagged_ratio", 0))

            if "memory_stats" in outputs:
                stats = outputs["memory_stats"]
                all_buffer_util.append(stats.get("buffer_utilization", 0))

    return {
        "mean_salience": np.mean(all_salience) if all_salience else 0.0,
        "tagged_ratio": np.mean(all_tagged_ratio) if all_tagged_ratio else 0.0,
        "buffer_utilization": np.mean(all_buffer_util) if all_buffer_util else 0.0,
    }


def compute_generation_quality(
    model,
    tokenizer,
    prompts: List[str],
    references: List[str],
    device: torch.device,
    max_new_tokens: int = 100,
) -> Dict[str, Any]:
    """
    Evaluate generation quality on a set of prompts.

    Args:
        model: SalienceFormer or causal LM
        tokenizer: Tokenizer for encoding/decoding
        prompts: List of input prompts
        references: List of expected continuations
        device: Device to run on
        max_new_tokens: Maximum tokens to generate

    Returns:
        Dictionary with generation metrics and samples
    """
    model.eval()
    generations = []

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # Generate
            if hasattr(model, "generate"):
                output_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Greedy for reproducibility
                    pad_token_id=tokenizer.pad_token_id,
                )
            else:
                # Fallback to base model generate
                output_ids = model.base_model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Decode only the new tokens
            new_tokens = output_ids[0, inputs["input_ids"].size(1):]
            generated = tokenizer.decode(new_tokens, skip_special_tokens=True)
            generations.append(generated)

    # Compute metrics
    rouge_scores = compute_rouge(generations, references)
    bleu_scores = compute_bleu(generations, [[r] for r in references])

    return {
        "generations": generations,
        **rouge_scores,
        **bleu_scores,
    }
