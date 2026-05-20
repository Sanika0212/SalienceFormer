"""
Tests for the evaluation module.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn

from evaluation.metrics import (
    compute_perplexity,
    compute_bleu,
    compute_rouge,
    compute_f1,
    compute_exact_match,
    EvaluationMetrics,
)
from evaluation.statistics import (
    compute_confidence_interval,
    paired_significance_test,
    unpaired_significance_test,
    aggregate_seeds,
    multiple_comparison_correction,
)
from evaluation.ablation import (
    AblationConfig,
    AblationType,
    create_ablation_variants,
    create_ablated_config,
)
from salienceformer.config import SalienceFormerConfig


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_bleu_perfect_match(self):
        """Test BLEU score for perfect match."""
        predictions = ["the cat sat on the mat"]
        references = [["the cat sat on the mat"]]

        result = compute_bleu(predictions, references)

        assert result["bleu"] == pytest.approx(1.0, abs=0.01)
        assert result["bleu_1"] == pytest.approx(1.0, abs=0.01)

    def test_bleu_no_match(self):
        """Test BLEU score for no match."""
        predictions = ["foo bar baz"]
        references = [["the cat sat on the mat"]]

        result = compute_bleu(predictions, references)

        assert result["bleu"] < 0.1

    def test_bleu_partial_match(self):
        """Test BLEU score for partial match."""
        predictions = ["the cat sat on a mat"]
        references = [["the cat sat on the mat"]]

        result = compute_bleu(predictions, references)

        assert 0.5 < result["bleu"] < 1.0

    def test_rouge_perfect_match(self):
        """Test ROUGE scores for perfect match."""
        predictions = ["the cat sat on the mat"]
        references = ["the cat sat on the mat"]

        result = compute_rouge(predictions, references)

        assert result["rouge_1"] == pytest.approx(1.0, abs=0.01)
        assert result["rouge_2"] == pytest.approx(1.0, abs=0.01)
        assert result["rouge_l"] == pytest.approx(1.0, abs=0.01)

    def test_rouge_partial_match(self):
        """Test ROUGE scores for partial match."""
        predictions = ["the cat sat"]
        references = ["the cat sat on the mat"]

        result = compute_rouge(predictions, references)

        assert result["rouge_1"] < 1.0
        assert result["rouge_1"] > 0.5  # Some overlap

    def test_f1_score(self):
        """Test F1 score computation."""
        predictions = ["the quick brown fox"]
        references = ["the slow brown fox"]

        result = compute_f1(predictions, references)

        # 3 out of 4 words match
        assert result["precision"] == pytest.approx(0.75, abs=0.01)
        assert result["recall"] == pytest.approx(0.75, abs=0.01)
        assert result["f1"] == pytest.approx(0.75, abs=0.01)

    def test_exact_match(self):
        """Test exact match computation."""
        predictions = ["hello world", "foo bar", "test"]
        references = ["hello world", "foo baz", "test"]

        result = compute_exact_match(predictions, references)

        assert result["exact_match"] == pytest.approx(2/3, abs=0.01)
        assert result["matches"] == 2
        assert result["total"] == 3

    def test_exact_match_normalized(self):
        """Test exact match with normalization."""
        predictions = ["Hello World", "  test  "]
        references = ["hello world", "test"]

        result = compute_exact_match(predictions, references, normalize=True)

        assert result["exact_match"] == 1.0

    def test_evaluation_metrics_to_dict(self):
        """Test EvaluationMetrics conversion to dict."""
        metrics = EvaluationMetrics(
            perplexity=15.5,
            loss=2.74,
            bleu=0.35,
        )

        d = metrics.to_dict()

        assert d["perplexity"] == 15.5
        assert d["loss"] == 2.74
        assert d["bleu"] == 0.35
        assert "rouge_1" not in d  # None values excluded


class TestStatistics:
    """Tests for statistical utilities."""

    def test_confidence_interval_basic(self):
        """Test confidence interval computation."""
        values = [10, 12, 14, 11, 13]

        mean, lower, upper = compute_confidence_interval(values, confidence=0.95)

        assert mean == pytest.approx(12.0, abs=0.01)
        assert lower < mean
        assert upper > mean
        assert upper - lower > 0

    def test_confidence_interval_single_value(self):
        """Test CI with single value."""
        values = [5.0]

        mean, lower, upper = compute_confidence_interval(values)

        assert mean == 5.0
        assert lower == 5.0
        assert upper == 5.0

    def test_paired_significance_test_significant(self):
        """Test paired test with significant difference."""
        values_a = [10, 12, 14, 11, 13, 15, 16, 14]
        values_b = [5, 6, 7, 5, 6, 8, 9, 7]

        result = paired_significance_test(values_a, values_b, alpha=0.05)

        assert result.significant
        assert result.p_value < 0.05
        assert result.effect_size > 0  # A > B

    def test_paired_significance_test_not_significant(self):
        """Test paired test with no significant difference."""
        np.random.seed(42)
        values_a = np.random.normal(10, 1, 10).tolist()
        values_b = np.random.normal(10, 1, 10).tolist()

        result = paired_significance_test(values_a, values_b, alpha=0.05)

        # May or may not be significant due to randomness
        assert result.p_value >= 0
        assert result.p_value <= 1

    def test_unpaired_significance_test(self):
        """Test unpaired significance test."""
        values_a = [10, 12, 14, 11, 13]
        values_b = [5, 6, 7, 5, 6]

        result = unpaired_significance_test(values_a, values_b, alpha=0.05)

        assert result.significant
        assert result.p_value < 0.05

    def test_aggregate_seeds(self):
        """Test result aggregation across seeds."""
        results = [
            {"perplexity": 15.0, "loss": 2.7},
            {"perplexity": 16.0, "loss": 2.8},
            {"perplexity": 14.5, "loss": 2.65},
        ]

        aggregated = aggregate_seeds(results)

        assert "perplexity" in aggregated
        assert aggregated["perplexity"]["mean"] == pytest.approx(15.17, abs=0.1)
        assert aggregated["perplexity"]["n_seeds"] == 3
        assert "std" in aggregated["perplexity"]
        assert "ci_lower" in aggregated["perplexity"]
        assert "ci_upper" in aggregated["perplexity"]

    def test_multiple_comparison_bonferroni(self):
        """Test Bonferroni correction."""
        p_values = [0.01, 0.03, 0.05, 0.10]

        adjusted, significant = multiple_comparison_correction(
            p_values, method="bonferroni", alpha=0.05
        )

        # Bonferroni multiplies by n
        assert adjusted[0] == pytest.approx(0.04, abs=0.01)  # 0.01 * 4
        assert adjusted[1] == pytest.approx(0.12, abs=0.01)  # 0.03 * 4
        assert significant[0]  # 0.04 < 0.05
        assert not significant[1]  # 0.12 > 0.05

    def test_multiple_comparison_holm(self):
        """Test Holm correction."""
        p_values = [0.01, 0.04, 0.03, 0.10]

        adjusted, significant = multiple_comparison_correction(
            p_values, method="holm", alpha=0.05
        )

        # Should be more powerful than Bonferroni
        assert all(0 <= p <= 1 for p in adjusted)


class TestAblation:
    """Tests for ablation framework."""

    def test_create_ablation_variants(self):
        """Test standard ablation variant creation."""
        variants = create_ablation_variants()

        assert len(variants) > 0

        # Check we have key ablations
        names = [v.name for v in variants]
        assert "full_salienceformer" in names
        assert "no_salience_gate" in names
        assert "no_memory_buffer" in names
        assert "no_drift_calibrator" in names

    def test_ablation_config_to_dict(self):
        """Test AblationConfig serialization."""
        config = AblationConfig(
            name="test",
            ablation_type=AblationType.REMOVE_SALIENCE,
            description="Test ablation",
            disable_salience_gate=True,
        )

        d = config.to_dict()

        assert d["name"] == "test"
        assert d["ablation_type"] == "no_salience"
        assert d["disable_salience_gate"] is True

    def test_create_ablated_config(self):
        """Test SalienceFormerConfig modification for ablation."""
        base_config = SalienceFormerConfig(
            buffer_size=2048,
            decay_rate=0.9,
        )

        ablation = AblationConfig(
            name="buffer_512",
            ablation_type=AblationType.VARY_BUFFER_SIZE,
            description="Small buffer",
            buffer_size=512,
        )

        new_config = create_ablated_config(base_config, ablation)

        assert new_config.buffer_size == 512
        assert new_config.decay_rate == 0.9  # Unchanged

    def test_create_ablated_config_decay_rate(self):
        """Test decay rate ablation."""
        base_config = SalienceFormerConfig(decay_rate=0.9)

        ablation = AblationConfig(
            name="decay_0.95",
            ablation_type=AblationType.VARY_DECAY_RATE,
            description="Slow decay",
            decay_rate=0.95,
        )

        new_config = create_ablated_config(base_config, ablation)

        assert new_config.decay_rate == 0.95

    def test_create_ablated_config_importance_range(self):
        """Test importance range ablation."""
        base_config = SalienceFormerConfig(importance_weight_range=(2.0, 5.0))

        ablation = AblationConfig(
            name="importance_1_10",
            ablation_type=AblationType.VARY_IMPORTANCE_RANGE,
            description="Wide range",
            importance_range=(1.0, 10.0),
        )

        new_config = create_ablated_config(base_config, ablation)

        assert new_config.importance_weight_range == (1.0, 10.0)


class MockModel(nn.Module):
    """Mock model for testing perplexity computation."""

    def __init__(self, vocab_size: int = 100, hidden_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.embedding(input_ids)
        logits = self.linear(x)
        return {"logits": logits}


class TestPerplexity:
    """Tests for perplexity computation."""

    def test_compute_perplexity_basic(self):
        """Test basic perplexity computation."""
        model = MockModel()
        model.eval()

        # Create simple dataloader with batched inputs
        class SimpleDataloader:
            def __init__(self):
                # Each item is a batch
                self.batches = [
                    {
                        "input_ids": torch.randint(0, 100, (2, 32)),
                        "attention_mask": torch.ones(2, 32, dtype=torch.long),
                    }
                    for _ in range(2)
                ]

            def __iter__(self):
                return iter(self.batches)

            def __len__(self):
                return len(self.batches)

        dataloader = SimpleDataloader()

        result = compute_perplexity(model, dataloader, torch.device("cpu"))

        assert "perplexity" in result
        assert "loss" in result
        assert result["perplexity"] > 0
        assert result["loss"] > 0

    def test_perplexity_lower_bound(self):
        """Test that perplexity is at least 1."""
        model = MockModel()
        model.eval()

        class SimpleDataloader:
            def __init__(self):
                self.batches = [
                    {
                        "input_ids": torch.randint(0, 100, (2, 16)),
                        "attention_mask": torch.ones(2, 16, dtype=torch.long),
                    }
                ]

            def __iter__(self):
                return iter(self.batches)

            def __len__(self):
                return len(self.batches)

        result = compute_perplexity(model, SimpleDataloader(), torch.device("cpu"))

        # Perplexity should be >= 1 (exp(0) = 1 is minimum)
        assert result["perplexity"] >= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
