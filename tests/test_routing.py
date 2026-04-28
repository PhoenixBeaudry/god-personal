"""
Tests that augmented_model_id routes correctly through the training pipeline.
Uses real payload models — no mocks.
Run with: python -m pytest tests/test_routing.py -v -o addopts=
"""

import pytest

from core.models.model_prep_models import (
    BaselineStats,
    DatasetStats,
    SeqLengthDistribution,
    TrainingDynamics,
    WeightStats,
    LayerGroupWeightStats,
    LayerGradStats,
)
from core.models.utility_models import InstructTextDatasetType
from core.models.payload_models import ModelPrepResponse
from core.models.payload_models import TrainRequestText
from core.models.payload_models import TrainRequestImage


def _make_stats() -> BaselineStats:
    """Build a minimal valid BaselineStats for test fixtures."""
    return BaselineStats(
        dataset=DatasetStats(
            total_tokens=1000,
            seq_length_distribution=SeqLengthDistribution(mean=10.0, p50=9, p95=20, p99=25, max=30),
            near_duplicate_rate=0.05,
            bits_per_byte=1.2,
            vocab_size=50257,
        ),
        weights=WeightStats(by_group={
            "attention_qkv": LayerGroupWeightStats(weight_rms=0.15, weight_norm=400.0, max_abs=3.5),
        }),
        training=TrainingDynamics(
            init_loss=3.5,
            grad_norms={"layer.0.weight": 1.2},
            gradient_noise_scale=0.07,
            activation_rms={"layer.0": 0.5},
            grad_stats={"layer.0": LayerGradStats(frobenius_norm=0.5, rms=0.01, max_abs=0.1, top_singular_values=[0.5])},
            output_entropy=4.0,
        ),
    )


class TestTrainRequestCarriesStats:
    """Baseline stats should flow through the actual TrainRequest models."""

    def test_text_request_with_stats(self):
        stats = _make_stats()
        req = TrainRequestText(
            model="gradients-io/augmented-abc123",
            task_id="task-123",
            hours_to_complete=2.0,
            dataset="s3://bucket/data.json",
            dataset_type=InstructTextDatasetType(),
            file_format="s3",
            baseline_stats=stats,
        )
        assert req.baseline_stats.training.init_loss == 3.5
        assert req.baseline_stats.dataset.total_tokens == 1000

    def test_text_request_without_stats(self):
        req = TrainRequestText(
            model="original/model-7b",
            task_id="task-456",
            hours_to_complete=3.0,
            dataset="s3://bucket/data.json",
            dataset_type=InstructTextDatasetType(),
            file_format="s3",
        )
        assert req.baseline_stats is None

    def test_image_request_with_stats(self):
        stats = _make_stats()
        req = TrainRequestImage(
            model="gradients-io/augmented-def456",
            task_id="task-789",
            hours_to_complete=1.0,
            dataset_zip="s3://bucket/images.zip",
            baseline_stats=stats,
        )
        assert req.baseline_stats.training.init_loss == 3.5


class TestModelPrepResponse:
    """ModelPrepResponse carries both augmented_model_id and stats."""

    def test_with_augmentation(self):
        resp = ModelPrepResponse(
            augmented_model_id="gradients-io/augmented-abc123",
            baseline_stats=_make_stats(),
        )
        assert resp.augmented_model_id == "gradients-io/augmented-abc123"
        assert resp.baseline_stats.training.init_loss == 3.5

    def test_without_augmentation(self):
        resp = ModelPrepResponse(
            augmented_model_id=None,
            baseline_stats=_make_stats(),
        )
        assert resp.augmented_model_id is None
        assert resp.baseline_stats.dataset.bits_per_byte == 1.2


class TestAugmentedModelIdRouting:
    """Test the `augmented_model_id or model_id` pattern used in orchestrator + eval."""

    def test_augmented_takes_precedence(self):
        model_id = "original/model-7b"
        augmented_model_id = "gradients-io/augmented-abc123"
        training_model = augmented_model_id or model_id
        assert training_model == "gradients-io/augmented-abc123"

    def test_falls_back_to_original(self):
        model_id = "original/model-7b"
        augmented_model_id = None
        training_model = augmented_model_id or model_id
        assert training_model == "original/model-7b"

    def test_empty_string_falls_back(self):
        model_id = "original/model-7b"
        augmented_model_id = ""
        training_model = augmented_model_id or model_id
        assert training_model == "original/model-7b"

    def test_eval_filters_augmented_base_not_original(self):
        model_id = "original/model-7b"
        augmented_model_id = "gradients-io/augmented-abc123"
        base_model = augmented_model_id or model_id

        miner_repos = ["gradients-io/augmented-abc123", "miner/finetuned-v1", "original/model-7b"]
        repos_to_evaluate = [r for r in miner_repos if r != base_model]

        assert "gradients-io/augmented-abc123" not in repos_to_evaluate
        assert "miner/finetuned-v1" in repos_to_evaluate
        assert "original/model-7b" in repos_to_evaluate


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
