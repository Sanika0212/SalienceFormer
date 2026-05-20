"""
Embedding Drift Calibrator Module

Maintains stable memory retrieval despite distribution shift in embeddings.
This is the learned equivalent of DriftCalibrator in brain/encoder.py.

Key innovations:
- No existing memory-augmented transformer explicitly handles representational drift
- Critical for continual learning scenarios
- Learns affine correction: h' = Ah + b to project back to base manifold
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class EmbeddingDriftCalibrator(nn.Module):
    """
    Maintains stable memory retrieval via learned affine correction.

    During continual learning or domain shift, the base transformer's hidden states
    may drift. This module:
    1. Establishes anchor embeddings from initial training data
    2. Measures drift as mean distance to nearest anchor
    3. Learns affine correction (h' = Ah + b) to project back to base manifold

    Analogous to hippocampal re-mapping where place cells maintain consistent
    coding despite neural drift.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        n_anchors: int = 64,
        drift_threshold: float = 0.3,
        update_momentum: float = 0.99,
        correction_strength: float = 1.0,
    ):
        """
        Args:
            hidden_dim: Dimension of hidden states
            n_anchors: Number of anchor embeddings to maintain
            drift_threshold: Apply correction when drift exceeds this
            update_momentum: Momentum for running statistics
            correction_strength: Scale factor for affine correction (0=none, 1=full)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_anchors = n_anchors
        self.drift_threshold = drift_threshold
        self.update_momentum = update_momentum
        self.correction_strength = correction_strength

        # Anchor embeddings (established during initial training)
        self.register_buffer('anchors', torch.zeros(n_anchors, hidden_dim))
        self.register_buffer('anchor_initialized', torch.tensor(False))

        # Learnable affine correction: h' = h @ A + b
        # Initialize to identity transformation
        self.affine_A = nn.Parameter(torch.eye(hidden_dim))
        self.affine_b = nn.Parameter(torch.zeros(hidden_dim))

        # Running statistics for drift detection
        self.register_buffer('running_mean', torch.zeros(hidden_dim))
        self.register_buffer('running_var', torch.ones(hidden_dim))
        self.register_buffer('n_updates', torch.tensor(0, dtype=torch.long))

        # Drift history for monitoring
        self.register_buffer('drift_history', torch.zeros(100))
        self.register_buffer('drift_ptr', torch.tensor(0, dtype=torch.long))

    def initialize_anchors(self, hidden_states: torch.Tensor) -> None:
        """
        Establish base manifold anchors from initial training data.

        Called automatically on first forward pass or explicitly during training setup.

        Args:
            hidden_states: (batch, seq_len, hidden_dim) initial training embeddings
        """
        # Flatten batch and sequence dimensions
        flat = hidden_states.view(-1, self.hidden_dim)
        n = flat.size(0)

        if n < self.n_anchors:
            # Not enough samples: use all with repetition
            indices = torch.randint(0, n, (self.n_anchors,), device=flat.device)
        else:
            # Select evenly-spaced samples as anchors
            indices = torch.linspace(0, n - 1, self.n_anchors, device=flat.device).long()

        self.anchors = flat[indices].detach().clone()
        self.anchor_initialized = torch.tensor(True, device=flat.device)

        # Initialize running stats
        self.running_mean = flat.mean(dim=0).detach()
        self.running_var = flat.var(dim=0).detach() + 1e-8

    def measure_drift(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute drift magnitude as mean distance to nearest anchor.

        This matches the DriftCalibrator.measure_drift() formula:
        drift = mean(min_dist(points, anchors))

        Args:
            hidden_states: (batch, seq_len, hidden_dim)

        Returns:
            drift_magnitude: scalar tensor
        """
        flat = hidden_states.view(-1, self.hidden_dim)

        # Distance to each anchor: (N, n_anchors)
        # Using cdist for efficient pairwise distances
        dists = torch.cdist(flat, self.anchors)  # (N, n_anchors)

        # Minimum distance to any anchor for each point
        min_dists = dists.min(dim=1)[0]  # (N,)

        # Mean of minimum distances
        drift_mag = min_dists.mean()

        # Update drift history
        idx = self.drift_ptr.item()
        self.drift_history[idx] = drift_mag.detach()
        self.drift_ptr = (self.drift_ptr + 1) % 100

        return drift_mag

    def apply_correction(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply learned affine correction to project back to base manifold.

        Formula: h' = h @ A + b

        The strength of correction is modulated by self.correction_strength.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)

        Returns:
            corrected_states: (batch, seq_len, hidden_dim)
        """
        # h' = h @ A + b
        corrected = torch.matmul(hidden_states, self.affine_A) + self.affine_b

        # Blend with original based on correction_strength
        if self.correction_strength < 1.0:
            corrected = (
                self.correction_strength * corrected +
                (1 - self.correction_strength) * hidden_states
            )

        return corrected

    def forward(
        self,
        hidden_states: torch.Tensor,
        update_stats: bool = True,
        force_correction: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply drift calibration to hidden states.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            update_stats: Whether to update running statistics (during training)
            force_correction: Always apply correction regardless of drift magnitude

        Returns:
            corrected_states: (batch, seq_len, hidden_dim)
            drift_magnitude: scalar tensor
        """
        # Initialize anchors on first call
        if not self.anchor_initialized:
            self.initialize_anchors(hidden_states)

        # Measure drift
        drift_mag = self.measure_drift(hidden_states)

        # Apply correction if drift exceeds threshold or forced
        if force_correction or drift_mag > self.drift_threshold:
            corrected = self.apply_correction(hidden_states)
        else:
            corrected = hidden_states

        # Update running stats during training
        if update_stats and self.training:
            flat = hidden_states.view(-1, self.hidden_dim)
            batch_mean = flat.mean(dim=0)
            batch_var = flat.var(dim=0) + 1e-8

            self.running_mean = (
                self.update_momentum * self.running_mean +
                (1 - self.update_momentum) * batch_mean.detach()
            )
            self.running_var = (
                self.update_momentum * self.running_var +
                (1 - self.update_momentum) * batch_var.detach()
            )
            self.n_updates += 1

        return corrected, drift_mag

    def get_drift_stats(self) -> dict:
        """Get statistics about drift and correction state."""
        # Get valid drift history entries
        if self.n_updates > 0:
            valid_len = min(self.n_updates.item(), 100)
            recent_drifts = self.drift_history[:valid_len]

            return {
                "current_drift": self.drift_history[(self.drift_ptr - 1) % 100].item(),
                "mean_drift": recent_drifts.mean().item(),
                "max_drift": recent_drifts.max().item(),
                "drift_threshold": self.drift_threshold,
                "n_updates": self.n_updates.item(),
                "anchor_initialized": self.anchor_initialized.item(),
                "correction_norm": torch.norm(self.affine_A - torch.eye(
                    self.hidden_dim, device=self.affine_A.device
                )).item(),
                "bias_norm": torch.norm(self.affine_b).item(),
            }
        else:
            return {
                "current_drift": 0.0,
                "mean_drift": 0.0,
                "max_drift": 0.0,
                "drift_threshold": self.drift_threshold,
                "n_updates": 0,
                "anchor_initialized": False,
                "correction_norm": 0.0,
                "bias_norm": 0.0,
            }

    def reset_anchors(self) -> None:
        """Reset anchor embeddings (call before new training phase)."""
        self.anchors.zero_()
        self.anchor_initialized = torch.tensor(False, device=self.anchors.device)
        self.running_mean.zero_()
        self.running_var.fill_(1.0)
        self.drift_history.zero_()
        self.drift_ptr.zero_()
        self.n_updates.zero_()

    def regularization_loss(self) -> torch.Tensor:
        """
        Compute regularization loss to prevent excessive correction.

        Penalizes deviation from identity transformation.

        Returns:
            Scalar regularization loss
        """
        # Penalize deviation from identity: ||A - I||^2 + ||b||^2
        identity = torch.eye(self.hidden_dim, device=self.affine_A.device)
        a_reg = torch.norm(self.affine_A - identity) ** 2
        b_reg = torch.norm(self.affine_b) ** 2

        return a_reg + b_reg
