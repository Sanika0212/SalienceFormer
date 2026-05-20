"""
Unit tests for SalienceFormer modules.

Tests cover:
- SalienceGate: shape, range, gradient flow
- DifferentiablePriorityBuffer: write, replay, decay
- EmbeddingDriftCalibrator: drift measurement, correction
- SalienceFormer: end-to-end forward pass
"""

import pytest
import torch
import torch.nn as nn

from salienceformer.config import SalienceFormerConfig
from salienceformer.salience.gate import SalienceGate
from salienceformer.memory.buffer import DifferentiablePriorityBuffer
from salienceformer.drift.calibrator import EmbeddingDriftCalibrator
from salienceformer.losses import SalienceFormerLoss, pearson_correlation


class TestSalienceGate:
    """Tests for SalienceGate module."""

    @pytest.fixture
    def gate(self):
        return SalienceGate(hidden_dim=64, n_heads=4, min_duration=2)

    @pytest.fixture
    def sample_input(self):
        batch, seq_len, hidden_dim = 2, 16, 64
        return torch.randn(batch, seq_len, hidden_dim)

    def test_output_shapes(self, gate, sample_input):
        """Salience gate outputs correct shapes."""
        salience, weights = gate(sample_input)

        assert salience.shape == (2, 16), f"Expected (2, 16), got {salience.shape}"
        assert weights.shape == (2, 16), f"Expected (2, 16), got {weights.shape}"

    def test_salience_range(self, gate, sample_input):
        """Salience scores are in [0, 1]."""
        salience, _ = gate(sample_input)

        assert salience.min() >= 0.0, f"Salience min {salience.min()} < 0"
        assert salience.max() <= 1.0, f"Salience max {salience.max()} > 1"

    def test_importance_weight_range(self, gate, sample_input):
        """Importance weights are in [1.0, 5.0]."""
        _, weights = gate(sample_input)

        assert weights.min() >= 1.0, f"Weight min {weights.min()} < 1.0"
        assert weights.max() <= 5.0, f"Weight max {weights.max()} > 5.0"

    def test_gradient_flow(self, gate, sample_input):
        """Gradients flow through salience gate."""
        sample_input.requires_grad = True
        salience, weights = gate(sample_input)

        loss = salience.mean() + weights.mean()
        loss.backward()

        assert sample_input.grad is not None, "No gradient on input"
        assert sample_input.grad.abs().sum() > 0, "Zero gradient"

    def test_attention_mask(self, gate, sample_input):
        """Attention mask properly zeros out padded positions."""
        mask = torch.ones(2, 16)
        mask[:, 8:] = 0  # Mask second half

        salience, weights = gate(sample_input, attention_mask=mask)

        # Masked positions should have zero salience
        assert (salience[:, 8:] == 0).all(), "Masked positions should have zero salience"

    def test_salience_stats(self, gate, sample_input):
        """get_salience_stats returns valid statistics."""
        salience, _ = gate(sample_input)
        stats = gate.get_salience_stats(salience)

        assert "mean_salience" in stats
        assert "tagged_ratio" in stats
        assert 0 <= stats["tagged_ratio"] <= 1


class TestDifferentiablePriorityBuffer:
    """Tests for DifferentiablePriorityBuffer module."""

    @pytest.fixture
    def buffer(self):
        return DifferentiablePriorityBuffer(
            buffer_size=32,
            hidden_dim=64,
            decay_rate=0.9,
            max_replay_rounds=5,
            priority_threshold=1.0,
        )

    @pytest.fixture
    def sample_input(self):
        batch, seq_len, hidden_dim = 2, 8, 64
        hidden = torch.randn(batch, seq_len, hidden_dim)
        # Some tokens above threshold, some below
        weights = torch.tensor([
            [0.5, 2.0, 1.5, 3.0, 0.8, 4.0, 1.0, 2.5],
            [1.2, 0.9, 3.5, 2.0, 0.5, 1.8, 4.5, 0.7],
        ])
        return hidden, weights

    def test_write_stores_high_priority(self, buffer, sample_input):
        """Buffer stores entries with importance > threshold."""
        hidden, weights = sample_input

        buffer.write(hidden, weights)

        # Should have stored some entries
        n_valid = buffer.valid_mask.sum().item()
        assert n_valid > 0, "No entries written to buffer"

        # Count entries above threshold: weights > 1.0
        expected_writes = (weights > 1.0).sum().item()
        assert n_valid == expected_writes, f"Expected {expected_writes} writes, got {n_valid}"

    def test_write_ignores_low_priority(self, buffer):
        """Buffer ignores entries with importance <= threshold."""
        hidden = torch.randn(2, 4, 64)
        weights = torch.ones(2, 4) * 0.5  # All below threshold

        buffer.write(hidden, weights)

        assert buffer.valid_mask.sum() == 0, "Should not write low-priority entries"

    def test_replay_returns_correct_shape(self, buffer, sample_input):
        """Replay consolidation returns correct shapes."""
        hidden, weights = sample_input
        buffer.write(hidden, weights)

        query = torch.randn(2, 4, 64)
        consolidated, replay_log = buffer.replay_consolidation(query)

        assert consolidated.shape == (2, 64), f"Expected (2, 64), got {consolidated.shape}"

    def test_replay_with_empty_buffer(self, buffer):
        """Replay with empty buffer returns zeros."""
        query = torch.randn(2, 4, 64)
        consolidated, replay_log = buffer.replay_consolidation(query)

        assert consolidated.shape == (2, 64)
        assert (consolidated == 0).all(), "Empty buffer should return zeros"

    def test_effective_priority_decay(self, buffer, sample_input):
        """Effective priority decays with age."""
        hidden, weights = sample_input
        buffer.write(hidden, weights)

        initial_eff = buffer.compute_effective_priority().clone()

        # Age the buffer
        buffer.ages += 1

        decayed_eff = buffer.compute_effective_priority()

        # Effective priority should be lower
        valid_initial = initial_eff[buffer.valid_mask]
        valid_decayed = decayed_eff[buffer.valid_mask]

        assert (valid_decayed < valid_initial).all(), "Priority should decay with age"

    def test_buffer_stats(self, buffer, sample_input):
        """get_buffer_stats returns valid statistics."""
        hidden, weights = sample_input
        buffer.write(hidden, weights)

        stats = buffer.get_buffer_stats()

        assert stats["n_entries"] > 0
        assert 0 <= stats["buffer_utilization"] <= 1
        assert stats["mean_priority"] > 0

    def test_reset_clears_buffer(self, buffer, sample_input):
        """reset() clears all buffer entries."""
        hidden, weights = sample_input
        buffer.write(hidden, weights)

        assert buffer.valid_mask.sum() > 0

        buffer.reset()

        assert buffer.valid_mask.sum() == 0, "Buffer should be empty after reset"


class TestEmbeddingDriftCalibrator:
    """Tests for EmbeddingDriftCalibrator module."""

    @pytest.fixture
    def calibrator(self):
        return EmbeddingDriftCalibrator(
            hidden_dim=64,
            n_anchors=16,
            drift_threshold=0.3,
        )

    @pytest.fixture
    def sample_input(self):
        return torch.randn(2, 8, 64)

    def test_anchor_initialization(self, calibrator, sample_input):
        """Anchors are initialized on first forward pass."""
        assert not calibrator.anchor_initialized

        calibrator(sample_input)

        assert calibrator.anchor_initialized
        assert calibrator.anchors.abs().sum() > 0

    def test_output_shape(self, calibrator, sample_input):
        """Calibrator outputs correct shape."""
        corrected, drift_mag = calibrator(sample_input)

        assert corrected.shape == sample_input.shape
        assert drift_mag.dim() == 0  # Scalar

    def test_drift_increases_with_shift(self, calibrator, sample_input):
        """Drift magnitude increases when input distribution shifts."""
        # Initialize with original data
        calibrator(sample_input)
        _, drift_original = calibrator(sample_input)

        # Shift the distribution
        shifted_input = sample_input + 10.0
        _, drift_shifted = calibrator(shifted_input)

        assert drift_shifted > drift_original, "Drift should increase with distribution shift"

    def test_correction_applied_when_drift_high(self, calibrator):
        """Affine correction is applied when drift exceeds threshold."""
        # Initialize with base data
        base_input = torch.randn(2, 8, 64)
        calibrator(base_input)

        # Modify affine parameters so correction is non-trivial
        with torch.no_grad():
            calibrator.affine_A.add_(torch.randn_like(calibrator.affine_A) * 0.1)
            calibrator.affine_b.add_(torch.randn(calibrator.hidden_dim) * 0.1)

        # Create high-drift input
        drifted_input = base_input + 5.0

        # With high drift, correction should change the output
        corrected, drift_mag = calibrator(drifted_input)

        if drift_mag > calibrator.drift_threshold:
            # Output should differ from input (since A != I and b != 0)
            diff = (corrected - drifted_input).abs().mean()
            assert diff > 0, "Correction should modify high-drift input"

    def test_gradient_flow(self, calibrator, sample_input):
        """Gradients flow through calibrator."""
        sample_input.requires_grad = True
        corrected, _ = calibrator(sample_input)

        loss = corrected.mean()
        loss.backward()

        assert sample_input.grad is not None

    def test_regularization_loss(self, calibrator, sample_input):
        """Regularization loss is computed correctly."""
        calibrator(sample_input)  # Initialize

        reg_loss = calibrator.regularization_loss()

        assert reg_loss >= 0, "Regularization loss should be non-negative"

    def test_drift_stats(self, calibrator, sample_input):
        """get_drift_stats returns valid statistics."""
        calibrator(sample_input)
        stats = calibrator.get_drift_stats()

        assert "current_drift" in stats
        assert "anchor_initialized" in stats
        assert stats["anchor_initialized"]


class TestSalienceFormerLoss:
    """Tests for SalienceFormerLoss module."""

    @pytest.fixture
    def loss_fn(self):
        return SalienceFormerLoss()

    def test_loss_computation(self, loss_fn):
        """Loss computes without error."""
        batch, seq, vocab = 2, 16, 100

        logits = torch.randn(batch, seq, vocab)
        labels = torch.randint(0, vocab, (batch, seq))
        salience = torch.rand(batch, seq)
        weights = 1.0 + salience * 4.0  # [1.0, 5.0]

        losses = loss_fn(logits, labels, salience, weights)

        assert "loss" in losses
        assert losses["loss"].dim() == 0  # Scalar

    def test_loss_components(self, loss_fn):
        """All loss components are computed."""
        batch, seq, vocab = 2, 16, 100

        logits = torch.randn(batch, seq, vocab)
        labels = torch.randint(0, vocab, (batch, seq))
        salience = torch.rand(batch, seq)
        weights = 1.0 + salience * 4.0

        losses = loss_fn(logits, labels, salience, weights)

        expected_keys = ["loss", "lm_loss", "weighted_lm_loss", "sparsity_loss", "memory_util_loss"]
        for key in expected_keys:
            assert key in losses, f"Missing loss component: {key}"

    def test_loss_with_ignore_index(self, loss_fn):
        """Loss correctly ignores padding tokens."""
        batch, seq, vocab = 2, 16, 100

        logits = torch.randn(batch, seq, vocab)
        labels = torch.randint(0, vocab, (batch, seq))
        labels[:, 8:] = -100  # Padding
        salience = torch.rand(batch, seq)
        weights = 1.0 + salience * 4.0

        losses = loss_fn(logits, labels, salience, weights)

        assert not torch.isnan(losses["loss"]), "Loss should not be NaN with padding"


class TestPearsonCorrelation:
    """Tests for pearson_correlation utility."""

    def test_perfect_correlation(self):
        """Perfect positive correlation returns 1.0."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])

        r = pearson_correlation(x, y)

        assert abs(r.item() - 1.0) < 1e-5, f"Expected 1.0, got {r.item()}"

    def test_perfect_negative_correlation(self):
        """Perfect negative correlation returns -1.0."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])

        r = pearson_correlation(x, y)

        assert abs(r.item() + 1.0) < 1e-5, f"Expected -1.0, got {r.item()}"

    def test_no_correlation(self):
        """Uncorrelated data returns ~0."""
        torch.manual_seed(42)
        x = torch.randn(1000)
        y = torch.randn(1000)

        r = pearson_correlation(x, y)

        assert abs(r.item()) < 0.1, f"Expected ~0, got {r.item()}"


class TestSalienceFormerConfig:
    """Tests for SalienceFormerConfig."""

    def test_default_config(self):
        """Default config creates valid instance."""
        config = SalienceFormerConfig()

        assert config.buffer_size > 0
        assert 0 < config.decay_rate < 1
        assert config.importance_weight_range[0] < config.importance_weight_range[1]

    def test_invalid_decay_rate(self):
        """Invalid decay_rate raises error."""
        with pytest.raises(AssertionError):
            SalienceFormerConfig(decay_rate=1.5)

    def test_invalid_weight_range(self):
        """Invalid importance_weight_range raises error."""
        with pytest.raises(AssertionError):
            SalienceFormerConfig(importance_weight_range=(5.0, 2.0))


class TestIntegration:
    """Integration tests for full forward pass without base model."""

    def test_modules_work_together(self):
        """All modules integrate correctly."""
        batch, seq, hidden = 2, 16, 64

        # Create modules
        gate = SalienceGate(hidden_dim=hidden)
        buffer = DifferentiablePriorityBuffer(buffer_size=32, hidden_dim=hidden)
        calibrator = EmbeddingDriftCalibrator(hidden_dim=hidden)

        # Simulate forward pass
        hidden_states = torch.randn(batch, seq, hidden)

        # 1. Salience scoring
        salience, weights = gate(hidden_states)

        # 2. Drift correction
        corrected, drift = calibrator(hidden_states)

        # 3. Memory write
        buffer.write(corrected, weights)

        # 4. Memory retrieval
        consolidated, _ = buffer.replay_consolidation(corrected)

        # Verify outputs
        assert salience.shape == (batch, seq)
        assert corrected.shape == (batch, seq, hidden)
        assert consolidated.shape == (batch, hidden)
        assert buffer.valid_mask.sum() > 0
