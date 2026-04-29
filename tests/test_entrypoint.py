"""
Tests for the model prep container entrypoint logic with a real model.
Run with: python -m pytest tests/test_entrypoint.py -v -o addopts= -s
"""

import json
import tempfile

import torch
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from core.models.model_prep_models import AugmentationConfig
from core.models.model_prep_models import AugmentationScope
from core.models.model_prep_models import AugmentationType
from core.models.model_prep_models import BaselineStats
from core.models.payload_models import ModelPrepResponse
from trainer.model_prep.augmentation import augment_model
from trainer.model_prep.stats import compute_text_stats
from trainer.model_prep.entrypoint import build_augmentation_config, generate_anonymous_repo_name


SAMPLE_DATA = [
    {"instruction": "What is 2+2?", "output": "4"},
    {"instruction": "Say hello", "output": "Hello!"},
    {"instruction": "Name a color", "output": "Blue"},
]


def _create_tiny_model_and_data(tmp_dir: str):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained("gpt2", n_layer=2, n_head=2, n_embd=64, vocab_size=tokenizer.vocab_size)
    model = AutoModelForCausalLM.from_config(config)

    model_path = f"{tmp_dir}/model"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    data_path = f"{tmp_dir}/train_data.json"
    with open(data_path, "w") as f:
        json.dump(SAMPLE_DATA, f)

    return model_path, data_path, model, tokenizer


class TestBuildAugmentationConfig:
    def test_returns_none_when_no_aug_type(self):
        class Args:
            aug_type = None
            scope = None
            seed = None
            intensity = None

        assert build_augmentation_config(Args()) is None

    def test_returns_config_when_set(self):
        class Args:
            aug_type = "gaussian_noise"
            scope = "all_layers"
            seed = 42
            intensity = 0.01

        config = build_augmentation_config(Args())
        assert config is not None
        assert config.aug_type == AugmentationType.GAUSSIAN_NOISE
        assert config.scope == AugmentationScope.ALL_LAYERS
        assert config.seed == 42
        assert config.intensity == 0.01


class TestAnonymousRepoName:
    def test_deterministic(self):
        a = generate_anonymous_repo_name("meta-llama/Llama-2-7b", 42)
        b = generate_anonymous_repo_name("meta-llama/Llama-2-7b", 42)
        assert a == b

    def test_different_models_different_names(self):
        a = generate_anonymous_repo_name("meta-llama/Llama-2-7b", 42)
        b = generate_anonymous_repo_name("mistral/Mistral-7B", 42)
        assert a != b

    def test_different_seeds_different_names(self):
        a = generate_anonymous_repo_name("meta-llama/Llama-2-7b", 42)
        b = generate_anonymous_repo_name("meta-llama/Llama-2-7b", 43)
        assert a != b

    def test_does_not_contain_model_name(self):
        name = generate_anonymous_repo_name("meta-llama/Llama-2-7b", 42)
        assert "llama" not in name.lower()
        assert "meta" not in name.lower()

    def test_format(self):
        name = generate_anonymous_repo_name("meta-llama/Llama-2-7b", 42)
        parts = name.split("/")
        assert len(parts) == 2
        assert parts[1].startswith("augmented-")


class TestFullPipeline:
    """Test the complete augment → stats → response flow with a real model."""

    def test_augment_then_stats_produces_valid_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path, data_path, model, tokenizer = _create_tiny_model_and_data(tmp)

            config = AugmentationConfig(
                aug_type=AugmentationType.GAUSSIAN_NOISE,
                scope=AugmentationScope.ALL_LAYERS,
                seed=42,
                intensity=0.01,
            )
            augment_model(model, config)
            stats = compute_text_stats(model, tokenizer, SAMPLE_DATA)

            result = ModelPrepResponse(
                augmented_model_id="test/augmented-abc123",
                baseline_stats=stats,
            )

            assert result.augmented_model_id == "test/augmented-abc123"
            assert result.baseline_stats.training.init_loss > 0
            assert len(result.baseline_stats.training.grad_norms) > 0

    def test_stats_only_no_augmentation(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path, data_path, model, tokenizer = _create_tiny_model_and_data(tmp)

            stats = compute_text_stats(model, tokenizer, SAMPLE_DATA)
            result = ModelPrepResponse(
                augmented_model_id=None,
                baseline_stats=stats,
            )

            assert result.augmented_model_id is None
            assert result.baseline_stats.training.init_loss > 0

    def test_response_json_roundtrip(self):
        """ModelPrepResponse serialises and deserialises correctly — this is how
        run_model_prep_container parses the container's stdout."""
        with tempfile.TemporaryDirectory() as tmp:
            model_path, data_path, model, tokenizer = _create_tiny_model_and_data(tmp)

            stats = compute_text_stats(model, tokenizer, SAMPLE_DATA)
            result = ModelPrepResponse(
                augmented_model_id="test/aug-123",
                baseline_stats=stats,
            )

            json_str = result.model_dump_json()
            parsed = ModelPrepResponse.model_validate_json(json_str)

            assert parsed.augmented_model_id == result.augmented_model_id
            assert abs(parsed.baseline_stats.training.init_loss - result.baseline_stats.training.init_loss) < 1e-6
            assert parsed.baseline_stats.dataset.vocab_size == result.baseline_stats.dataset.vocab_size

    def test_each_augmentation_type_produces_different_stats(self):
        """Each augmentation type should change the model differently."""
        with tempfile.TemporaryDirectory() as tmp:
            model_path, data_path, _, tokenizer = _create_tiny_model_and_data(tmp)

            stats_by_type = {}
            for aug_type in AugmentationType:
                model = AutoModelForCausalLM.from_pretrained(model_path)
                config = AugmentationConfig(
                    aug_type=aug_type,
                    scope=AugmentationScope.ALL_LAYERS,
                    seed=42,
                    intensity=0.05,
                )
                augment_model(model, config)
                stats = compute_text_stats(model, tokenizer, SAMPLE_DATA)
                stats_by_type[aug_type] = stats

            # At least some types should produce different losses
            losses = [s.training.init_loss for s in stats_by_type.values()]
            assert len(set(f"{l:.4f}" for l in losses)) > 1, "Expected different losses for different augmentation types"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
