"""
End-to-end test of the model prep pipeline with a tiny model.
Tests augmentation + stats collection without Docker or HF upload.
Run with: python -m pytest tests/test_model_prep_e2e.py -v -o addopts= -s
"""

import json
import tempfile

import torch
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from core.models.model_prep_models import AugmentationConfig
from core.models.model_prep_models import AugmentationScope
from core.models.model_prep_models import AugmentationType
from trainer.model_prep.augmentation import augment_model
from trainer.model_prep.stats import compute_baseline_stats


def create_tiny_model(tmp_dir: str) -> tuple[str, AutoModelForCausalLM, AutoTokenizer]:
    """Create a minimal GPT2-style model for testing."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(
        "gpt2",
        n_layer=2,
        n_head=2,
        n_embd=64,
        vocab_size=tokenizer.vocab_size,
    )
    model = AutoModelForCausalLM.from_config(config)

    model_path = f"{tmp_dir}/tiny_model"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    return model_path, model, tokenizer


SAMPLE_DATA = [
    {"instruction": "What is 2+2?", "output": "4"},
    {"instruction": "Say hello", "output": "Hello!"},
    {"instruction": "Name a color", "output": "Blue"},
    {"instruction": "What is the capital of France?", "output": "Paris"},
    {"instruction": "Count to three", "output": "1, 2, 3"},
]


class TestAugmentModel:
    """Test augment_model with a real (tiny) transformer."""

    def test_gaussian_noise_modifies_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, model, _ = create_tiny_model(tmp)
            original_params = {n: p.clone() for n, p in model.named_parameters()}

            config = AugmentationConfig(
                aug_type=AugmentationType.GAUSSIAN_NOISE,
                scope=AugmentationScope.ALL_LAYERS,
                seed=42,
                intensity=0.01,
            )
            augment_model(model, config)

            changed = 0
            for name, param in model.named_parameters():
                if not torch.equal(param.data, original_params[name]):
                    changed += 1

            assert changed > 0, "No parameters were modified"

    def test_weight_scaling(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, model, _ = create_tiny_model(tmp)

            config = AugmentationConfig(
                aug_type=AugmentationType.WEIGHT_SCALING,
                scope=AugmentationScope.SINGLE_LAYER,
                seed=42,
                intensity=1.5,
            )
            augment_model(model, config)
            # If it didn't crash, the model is still valid

    def test_magnitude_pruning(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, model, _ = create_tiny_model(tmp)

            config = AugmentationConfig(
                aug_type=AugmentationType.MAGNITUDE_PRUNING,
                scope=AugmentationScope.LAYER_TYPE_GROUP,
                seed=42,
                intensity=0.1,
            )
            augment_model(model, config)

            # Check some zeros were introduced
            total_zeros = 0
            total_params = 0
            for _, p in model.named_parameters():
                total_zeros += (p == 0).sum().item()
                total_params += p.numel()
            assert total_zeros > 0, "Pruning should zero some weights"

    def test_layer_reinit(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, model, _ = create_tiny_model(tmp)
            original_params = {n: p.clone() for n, p in model.named_parameters()}

            config = AugmentationConfig(
                aug_type=AugmentationType.LAYER_REINIT,
                scope=AugmentationScope.MULTI_LAYER,
                seed=42,
                intensity=0.05,
            )
            augment_model(model, config)

            changed = 0
            for name, param in model.named_parameters():
                if not torch.equal(param.data, original_params[name]):
                    changed += 1
            assert changed > 0

    def test_deterministic_with_same_seed(self):
        """Same config + same model = identical result."""
        with tempfile.TemporaryDirectory() as tmp:
            path, _, _ = create_tiny_model(tmp)

            config = AugmentationConfig(
                aug_type=AugmentationType.GAUSSIAN_NOISE,
                scope=AugmentationScope.ALL_LAYERS,
                seed=12345,
                intensity=0.01,
            )

            model1 = AutoModelForCausalLM.from_pretrained(path)
            augment_model(model1, config)

            model2 = AutoModelForCausalLM.from_pretrained(path)
            augment_model(model2, config)

            for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
                assert torch.equal(p1, p2), f"Mismatch in {n1}"

    def test_model_still_runs_after_augmentation(self):
        """Augmented model can still do a forward pass."""
        with tempfile.TemporaryDirectory() as tmp:
            _, model, tokenizer = create_tiny_model(tmp)

            config = AugmentationConfig(
                aug_type=AugmentationType.GAUSSIAN_NOISE,
                scope=AugmentationScope.ALL_LAYERS,
                seed=42,
                intensity=0.01,
            )
            augment_model(model, config)

            inputs = tokenizer("Hello world", return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
            assert outputs.loss is not None
            assert outputs.loss.item() > 0


class TestBaselineStats:
    """Test baseline stats computation with a real model."""

    def test_computes_loss_and_grad_norm(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, model, tokenizer = create_tiny_model(tmp)
            stats = compute_baseline_stats(model, tokenizer, SAMPLE_DATA)

            assert stats.loss > 0, "Loss should be positive"
            assert stats.grad_norm > 0, "Grad norm should be positive"

    def test_stats_after_augmentation(self):
        """Stats should differ after augmentation."""
        with tempfile.TemporaryDirectory() as tmp:
            path, _, tokenizer = create_tiny_model(tmp)

            model_orig = AutoModelForCausalLM.from_pretrained(path)
            stats_orig = compute_baseline_stats(model_orig, tokenizer, SAMPLE_DATA)

            model_aug = AutoModelForCausalLM.from_pretrained(path)
            config = AugmentationConfig(
                aug_type=AugmentationType.GAUSSIAN_NOISE,
                scope=AugmentationScope.ALL_LAYERS,
                seed=42,
                intensity=0.02,
            )
            augment_model(model_aug, config)
            stats_aug = compute_baseline_stats(model_aug, tokenizer, SAMPLE_DATA)

            # Stats should differ (augmentation changes model behavior)
            assert stats_orig.loss != stats_aug.loss or stats_orig.grad_norm != stats_aug.grad_norm


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
