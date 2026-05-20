"""
Salience Gate Module

Dual-pathway salience scoring inspired by hippocampal SPW-R (Sharp Wave Ripple) detection.
This is the learned equivalent of the fixed bandpass + Hilbert + threshold detection
in brain/tagger.py.

Key innovations vs. prior work:
- Titans uses single "surprise" metric; we separate local (token-intrinsic) and
  global (contextual) importance, mirroring how SPW-Rs emerge from both cellular
  and population-level dynamics.
- Temporal smoothing via causal convolution mimics duration constraints
  (MIN_RIPPLE_DURATION_MS / MAX_RIPPLE_DURATION_MS from original).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SalienceGate(nn.Module):
    """
    Dual-pathway salience scoring module.

    Computes importance scores for each token position using:
    1. Local pathway: Token-intrinsic importance via MLP (like single-electrode ripple detection)
    2. Global pathway: Contextual importance via cross-attention (like population synchrony)
    3. Temporal smoothing: Prevents isolated spikes, mimics duration constraints

    Output:
    - salience_scores: [0, 1] probability of being "tagged"
    - importance_weights: [1.0, 5.0] importance weight (1.0 for untagged, 2.0-5.0 for tagged)
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int = 4,
        min_duration: int = 2,
        threshold_init: float = 0.0,
        importance_range: Tuple[float, float] = (2.0, 5.0),
        untagged_weight: float = 1.0,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_dim: Dimension of input hidden states
            n_heads: Number of attention heads for global pathway
            min_duration: Kernel size for temporal smoothing (mimics MIN_RIPPLE_DURATION)
            threshold_init: Initial value for learnable threshold
            importance_range: (lo, hi) for tagged importance weights
            untagged_weight: Weight for non-salient tokens (default 1.0)
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.min_duration = min_duration
        self.untagged_weight = untagged_weight

        # Local pathway: MLPs operating on individual hidden states
        # Analogous to detecting ripple amplitude at a single electrode
        self.local_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        # Global pathway: Cross-token attention to capture population synchrony
        # Analogous to detecting coordinated firing across the population
        self.global_query = nn.Linear(hidden_dim, hidden_dim)
        self.global_key = nn.Linear(hidden_dim, hidden_dim)
        self.global_value = nn.Linear(hidden_dim, hidden_dim)
        self.global_proj = nn.Linear(hidden_dim, 1)
        self.attn_dropout = nn.Dropout(dropout)

        # Temporal smoothing: Causal convolution for duration constraints
        # Mimics MIN_RIPPLE_DURATION_MS / MAX_RIPPLE_DURATION_MS
        if min_duration > 1:
            self.temporal_smooth = nn.Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=min_duration,
                padding=min_duration - 1,  # Causal padding
                bias=True,
            )
            # Initialize to averaging filter
            nn.init.constant_(self.temporal_smooth.weight, 1.0 / min_duration)
            nn.init.constant_(self.temporal_smooth.bias, 0.0)
        else:
            self.temporal_smooth = None

        # Learnable threshold (analogous to SALIENCE_THRESHOLD_STD * std)
        self.threshold = nn.Parameter(torch.tensor(threshold_init))

        # Amplitude-to-importance mapping [2.0, 5.0] range from tagger.py
        self.importance_lo = nn.Parameter(torch.tensor(importance_range[0]), requires_grad=False)
        self.importance_hi = nn.Parameter(torch.tensor(importance_range[1]), requires_grad=False)

        # Combination weights for local and global pathways
        self.pathway_weight = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute salience scores and importance weights.

        Args:
            hidden_states: (batch, seq_len, hidden_dim) from base transformer
            attention_mask: (batch, seq_len) with 1 for valid, 0 for padding

        Returns:
            salience_scores: (batch, seq_len) in [0, 1]
            importance_weights: (batch, seq_len) in [untagged_weight, importance_hi]
        """
        B, T, D = hidden_states.shape
        device = hidden_states.device

        # === Local pathway: per-token intrinsic salience ===
        local_scores = self.local_gate(hidden_states).squeeze(-1)  # (B, T)

        # === Global pathway: contextual salience via multi-head attention ===
        # Query, Key, Value projections
        Q = self.global_query(hidden_states)  # (B, T, D)
        K = self.global_key(hidden_states)    # (B, T, D)
        V = self.global_value(hidden_states)  # (B, T, D)

        # Reshape for multi-head attention
        head_dim = D // self.n_heads
        Q = Q.view(B, T, self.n_heads, head_dim).transpose(1, 2)  # (B, H, T, head_dim)
        K = K.view(B, T, self.n_heads, head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_heads, head_dim).transpose(1, 2)

        # Attention scores: how much does each token relate to others
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(head_dim)  # (B, H, T, T)

        # Apply attention mask (causal or padding)
        if attention_mask is not None:
            # Expand mask for broadcasting: (B, 1, 1, T)
            mask = attention_mask[:, None, None, :].bool()
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # Weighted aggregation
        attn_output = torch.matmul(attn_probs, V)  # (B, H, T, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)

        # Project to scalar "synchrony" score
        global_scores = self.global_proj(attn_output).squeeze(-1)  # (B, T)

        # === Combine pathways ===
        # Learnable weighting between local and global
        w = torch.sigmoid(self.pathway_weight)
        combined = w * local_scores + (1 - w) * global_scores  # (B, T)

        # === Temporal smoothing ===
        if self.temporal_smooth is not None:
            # (B, T) -> (B, 1, T) -> conv -> (B, 1, T+padding) -> [:, :, :T]
            combined_padded = combined.unsqueeze(1)
            combined_smooth = self.temporal_smooth(combined_padded)[:, 0, :T]
        else:
            combined_smooth = combined

        # === Apply threshold and sigmoid ===
        # Analogous to: thresh = mean + SALIENCE_THRESHOLD_STD * std
        salience_logits = combined_smooth - self.threshold
        salience_scores = torch.sigmoid(salience_logits)  # (B, T)

        # === Map to importance weights ===
        # Untagged (low salience) → 1.0
        # Tagged (high salience) → [2.0, 5.0] scaled by salience magnitude
        # This is a soft/continuous version of the discrete tagging in tagger.py
        lo = self.importance_lo
        hi = self.importance_hi
        importance_weights = self.untagged_weight + salience_scores * (hi - self.untagged_weight)

        # Apply mask to outputs
        if attention_mask is not None:
            salience_scores = salience_scores * attention_mask
            importance_weights = importance_weights * attention_mask + \
                                 self.untagged_weight * (1 - attention_mask)

        return salience_scores, importance_weights

    def get_salience_stats(
        self,
        salience_scores: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute statistics about salience distribution (for monitoring/debugging).

        Args:
            salience_scores: (batch, seq_len) salience scores
            attention_mask: (batch, seq_len) mask

        Returns:
            Dictionary with salience statistics
        """
        if attention_mask is not None:
            valid_scores = salience_scores[attention_mask.bool()]
        else:
            valid_scores = salience_scores.flatten()

        # Threshold for "tagged" (using 0.5 as hard threshold for stats)
        tagged_ratio = (valid_scores > 0.5).float().mean().item()

        return {
            "mean_salience": valid_scores.mean().item(),
            "std_salience": valid_scores.std().item(),
            "min_salience": valid_scores.min().item(),
            "max_salience": valid_scores.max().item(),
            "tagged_ratio": tagged_ratio,
            "threshold": self.threshold.item(),
            "pathway_weight": torch.sigmoid(self.pathway_weight).item(),
        }
