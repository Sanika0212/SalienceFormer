"""
SalienceFormer Training Losses

Multi-objective loss functions for training SalienceFormer, including:
1. Language modeling loss
2. Salience-weighted loss
3. Sparsity regularization
4. Consolidation R-score auxiliary loss
"""

from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def pearson_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute Pearson correlation coefficient between two tensors.

    Args:
        x: (N,) first tensor
        y: (N,) second tensor

    Returns:
        Scalar correlation coefficient
    """
    x_mean = x - x.mean()
    y_mean = y - y.mean()

    num = (x_mean * y_mean).sum()
    den = torch.sqrt((x_mean ** 2).sum() * (y_mean ** 2).sum() + 1e-8)

    return num / den


class SalienceFormerLoss(nn.Module):
    """
    Multi-objective loss for SalienceFormer training.

    Combines multiple objectives to train the hippocampal memory mechanisms:
    1. LM Loss: Standard next-token prediction
    2. Salience-Weighted LM: Higher weight for salient tokens
    3. Sparsity: Encourage selective tagging (not everything important)
    4. Memory Utilization: Encourage using the memory
    """

    def __init__(
        self,
        lm_weight: float = 1.0,
        salience_weighted_weight: float = 0.5,
        sparsity_weight: float = 0.1,
        memory_util_weight: float = 0.01,
        r_score_weight: float = 0.0,  # Enable after initial training
        target_r_score: float = 0.86,
    ):
        super().__init__()
        self.lm_weight = lm_weight
        self.salience_weighted_weight = salience_weighted_weight
        self.sparsity_weight = sparsity_weight
        self.memory_util_weight = memory_util_weight
        self.r_score_weight = r_score_weight
        self.target_r_score = target_r_score

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        salience_scores: torch.Tensor,
        importance_weights: torch.Tensor,
        retrieval_quality: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-objective loss.

        Args:
            logits: (batch, seq_len, vocab_size) model predictions
            labels: (batch, seq_len) target labels
            salience_scores: (batch, seq_len) from SalienceGate
            importance_weights: (batch, seq_len) from SalienceGate
            retrieval_quality: (batch, seq_len) optional per-token retrieval scores
            attention_mask: (batch, seq_len) optional mask

        Returns:
            Dictionary with total loss and component losses
        """
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_salience = salience_scores[..., :-1].contiguous()
        shift_weights = importance_weights[..., :-1].contiguous()

        if attention_mask is not None:
            shift_mask = attention_mask[..., :-1].contiguous()
        else:
            shift_mask = torch.ones_like(shift_labels, dtype=torch.float)

        # 1. Standard LM loss
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction='mean',
        )

        # 2. Salience-weighted LM loss
        token_losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction='none',
        ).view_as(shift_labels)

        # Apply mask
        valid_mask = (shift_labels != -100).float() * shift_mask

        # Weighted loss (normalize to maintain scale)
        weighted_losses = token_losses * shift_weights * valid_mask
        weighted_lm_loss = weighted_losses.sum() / (valid_mask.sum() + 1e-8)

        # 3. Sparsity loss (binary entropy)
        # Low entropy = salience is decisive (near 0 or 1)
        # We want to maximize entropy penalty, so we add negative entropy
        eps = 1e-8
        entropy = -(
            shift_salience * torch.log(shift_salience + eps) +
            (1 - shift_salience) * torch.log(1 - shift_salience + eps)
        )
        sparsity_loss = entropy.mean()  # We want low entropy (decisive salience)

        # 4. Memory utilization loss
        # Encourage some selection (not all zeros)
        mean_salience = (shift_salience * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        # Penalty if mean is too low (< 0.1) or too high (> 0.5)
        target_mean = 0.2
        memory_util_loss = (mean_salience - target_mean) ** 2

        # 5. R-score loss (optional, for later training stages)
        r_score_loss = torch.tensor(0.0, device=logits.device)
        if self.r_score_weight > 0 and retrieval_quality is not None:
            shift_retrieval = retrieval_quality[..., :-1].contiguous()
            flat_salience = shift_salience.view(-1)
            flat_retrieval = shift_retrieval.view(-1)
            flat_mask = valid_mask.view(-1).bool()

            if flat_mask.sum() > 10:  # Need enough samples
                r = pearson_correlation(
                    flat_salience[flat_mask],
                    flat_retrieval[flat_mask]
                )
                # Loss: penalize if R < target
                r_score_loss = F.relu(self.target_r_score - r)

        # Combine losses
        total_loss = (
            self.lm_weight * lm_loss +
            self.salience_weighted_weight * weighted_lm_loss +
            self.sparsity_weight * sparsity_loss +
            self.memory_util_weight * memory_util_loss +
            self.r_score_weight * r_score_loss
        )

        return {
            "loss": total_loss,
            "lm_loss": lm_loss,
            "weighted_lm_loss": weighted_lm_loss,
            "sparsity_loss": sparsity_loss,
            "memory_util_loss": memory_util_loss,
            "r_score_loss": r_score_loss,
            "mean_salience": mean_salience,
        }


class ConsolidationRScore(nn.Module):
    """
    Consolidation R-Score metric from evaluation/tier2_llm.py.

    Measures Pearson correlation between salience (importance_weight)
    and retrieval quality (how well memories are retrieved).

    Target: R >= 0.86 (from the original paper)
    """

    def __init__(self, target: float = 0.86):
        super().__init__()
        self.target = target

    def forward(
        self,
        salience: torch.Tensor,
        retrieval_quality: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute consolidation R-score.

        Args:
            salience: (N,) or (batch, seq) salience scores
            retrieval_quality: (N,) or (batch, seq) retrieval accuracy
            mask: Optional mask for valid entries

        Returns:
            Dictionary with R-score and whether target is met
        """
        flat_salience = salience.view(-1)
        flat_retrieval = retrieval_quality.view(-1)

        if mask is not None:
            flat_mask = mask.view(-1).bool()
            flat_salience = flat_salience[flat_mask]
            flat_retrieval = flat_retrieval[flat_mask]

        r = pearson_correlation(flat_salience, flat_retrieval)

        return {
            "r_score": r,
            "target": torch.tensor(self.target, device=r.device),
            "meets_target": r >= self.target,
        }
