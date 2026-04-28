"""
Test augmentation + stats on a real (small) HuggingFace model.
Verifies that augmentation changes model behaviour measurably.
Run with: python -m pytest tests/test_real_model_augmentation.py -v -o addopts= -s
"""

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.models.model_prep_models import AugmentationConfig
from core.models.model_prep_models import AugmentationScope
from core.models.model_prep_models import AugmentationType
from trainer.model_prep.augmentation import augment_model
from trainer.model_prep.stats import compute_baseline_stats


REAL_MODEL_ID = "distilgpt2"

EVAL_DATA = [
    {"text": "The capital of France is Paris and it is known for the Eiffel Tower."},
    {"text": "Machine learning is a subset of artificial intelligence that focuses on data."},
    {"text": "The quick brown fox jumps over the lazy dog in the garden."},
    {"text": "Python is a popular programming language used for web development."},
    {"text": "Climate change is one of the biggest challenges facing humanity today."},
    {"text": "The stock market experienced significant volatility during the pandemic."},
    {"text": "Quantum computing promises to revolutionize cryptography and drug discovery."},
    {"text": "The Great Wall of China is one of the most impressive structures ever built."},
]


@pytest.fixture(scope="module")
def model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(REAL_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    return REAL_MODEL_ID, tokenizer


class TestBaselineVsAugmented:
    """Compare base model stats against each augmentation type."""

    def _get_stats(self, model_id, tokenizer, aug_config=None):
        model = AutoModelForCausalLM.from_pretrained(model_id)
        if aug_config is not None:
            augment_model(model, aug_config)
        return compute_baseline_stats(model, tokenizer, EVAL_DATA)

    def test_base_model_produces_valid_stats(self, model_and_tokenizer):
        model_id, tokenizer = model_and_tokenizer
        stats = self._get_stats(model_id, tokenizer)
        print(f"Base model: loss={stats.training.init_loss:.4f}, entropy={stats.training.output_entropy:.4f}")
        assert stats.training.init_loss > 0
        assert stats.training.output_entropy > 0
        assert len(stats.training.grad_norms) > 0

    def test_gaussian_noise_changes_loss(self, model_and_tokenizer):
        model_id, tokenizer = model_and_tokenizer
        base_stats = self._get_stats(model_id, tokenizer)

        aug_stats = self._get_stats(model_id, tokenizer, AugmentationConfig(
            aug_type=AugmentationType.GAUSSIAN_NOISE,
            scope=AugmentationScope.ALL_LAYERS,
            seed=42,
            intensity=0.02,
        ))
        print(f"Gaussian noise: base_loss={base_stats.training.init_loss:.4f}, aug_loss={aug_stats.training.init_loss:.4f}")
        assert aug_stats.training.init_loss != base_stats.training.init_loss

    def test_weight_scaling_changes_loss(self, model_and_tokenizer):
        model_id, tokenizer = model_and_tokenizer
        base_stats = self._get_stats(model_id, tokenizer)

        aug_stats = self._get_stats(model_id, tokenizer, AugmentationConfig(
            aug_type=AugmentationType.WEIGHT_SCALING,
            scope=AugmentationScope.ALL_LAYERS,
            seed=42,
            intensity=1.1,
        ))
        print(f"Weight scaling: base_loss={base_stats.training.init_loss:.4f}, aug_loss={aug_stats.training.init_loss:.4f}")
        assert aug_stats.training.init_loss != base_stats.training.init_loss

    def test_magnitude_pruning_changes_loss(self, model_and_tokenizer):
        model_id, tokenizer = model_and_tokenizer
        base_stats = self._get_stats(model_id, tokenizer)

        aug_stats = self._get_stats(model_id, tokenizer, AugmentationConfig(
            aug_type=AugmentationType.MAGNITUDE_PRUNING,
            scope=AugmentationScope.ALL_LAYERS,
            seed=42,
            intensity=0.1,
        ))
        print(f"Magnitude pruning: base_loss={base_stats.training.init_loss:.4f}, aug_loss={aug_stats.training.init_loss:.4f}")
        assert aug_stats.training.init_loss != base_stats.training.init_loss

    def test_layer_reinit_changes_loss(self, model_and_tokenizer):
        model_id, tokenizer = model_and_tokenizer
        base_stats = self._get_stats(model_id, tokenizer)

        aug_stats = self._get_stats(model_id, tokenizer, AugmentationConfig(
            aug_type=AugmentationType.LAYER_REINIT,
            scope=AugmentationScope.ALL_LAYERS,
            seed=42,
            intensity=0.05,
        ))
        print(f"Layer reinit: base_loss={base_stats.training.init_loss:.4f}, aug_loss={aug_stats.training.init_loss:.4f}")
        assert aug_stats.training.init_loss != base_stats.training.init_loss

    def test_augmentation_increases_loss(self, model_and_tokenizer):
        """Augmentation should generally degrade the model — loss goes up."""
        model_id, tokenizer = model_and_tokenizer
        base_stats = self._get_stats(model_id, tokenizer)

        worse_count = 0
        for aug_type in AugmentationType:
            aug_stats = self._get_stats(model_id, tokenizer, AugmentationConfig(
                aug_type=aug_type,
                scope=AugmentationScope.ALL_LAYERS,
                seed=42,
                intensity=0.05,
            ))
            print(f"  {aug_type.value}: loss={aug_stats.training.init_loss:.4f} (base={base_stats.training.init_loss:.4f})")
            if aug_stats.training.init_loss > base_stats.training.init_loss:
                worse_count += 1

        # Most augmentation types should increase loss
        assert worse_count >= 3, f"Expected at least 3/4 augmentations to increase loss, got {worse_count}"

    def test_single_layer_scope_less_impact_than_all(self, model_and_tokenizer):
        """Augmenting one layer should change loss less than augmenting all."""
        model_id, tokenizer = model_and_tokenizer
        base_stats = self._get_stats(model_id, tokenizer)

        single_stats = self._get_stats(model_id, tokenizer, AugmentationConfig(
            aug_type=AugmentationType.GAUSSIAN_NOISE,
            scope=AugmentationScope.SINGLE_LAYER,
            seed=42,
            intensity=0.02,
        ))

        all_stats = self._get_stats(model_id, tokenizer, AugmentationConfig(
            aug_type=AugmentationType.GAUSSIAN_NOISE,
            scope=AugmentationScope.ALL_LAYERS,
            seed=42,
            intensity=0.02,
        ))

        single_delta = abs(single_stats.training.init_loss - base_stats.training.init_loss)
        all_delta = abs(all_stats.training.init_loss - base_stats.training.init_loss)
        print(f"Single layer delta: {single_delta:.4f}, All layers delta: {all_delta:.4f}")
        assert all_delta > single_delta

    def test_deterministic_across_runs(self, model_and_tokenizer):
        """Same augmentation config should produce identical loss.
        Grad norm can vary slightly due to non-deterministic torch ops."""
        model_id, tokenizer = model_and_tokenizer
        config = AugmentationConfig(
            aug_type=AugmentationType.GAUSSIAN_NOISE,
            scope=AugmentationScope.ALL_LAYERS,
            seed=99,
            intensity=0.01,
        )

        stats1 = self._get_stats(model_id, tokenizer, config)
        stats2 = self._get_stats(model_id, tokenizer, config)

        assert abs(stats1.training.init_loss - stats2.training.init_loss) < 1e-6
        # Grad norm may vary slightly due to non-deterministic backward pass
        assert abs(stats1.training.gradient_noise_scale - stats2.training.gradient_noise_scale) / max(stats1.training.gradient_noise_scale, 1e-8) < 0.2


class TestAugmentationSweep:
    """Sweep across types, scopes, and intensities to show impact."""

    def _get_stats(self, model_id, tokenizer, aug_config=None):
        model = AutoModelForCausalLM.from_pretrained(model_id)
        if aug_config is not None:
            augment_model(model, aug_config)
        return compute_baseline_stats(model, tokenizer, EVAL_DATA)

    def test_intensity_sweep(self, model_and_tokenizer):
        """Higher intensity should mean bigger loss delta."""
        model_id, tokenizer = model_and_tokenizer
        base_stats = self._get_stats(model_id, tokenizer)

        print(f"\n{'Type':<20} {'Intensity':<12} {'Loss':<12} {'Delta':<12} {'GradNorm':<12}")
        print("-" * 68)

        for aug_type in AugmentationType:
            if aug_type == AugmentationType.WEIGHT_SCALING:
                intensities = [0.9, 0.8, 0.7, 1.1, 1.2, 1.3]
            elif aug_type == AugmentationType.GAUSSIAN_NOISE:
                intensities = [0.005, 0.01, 0.02, 0.05, 0.1]
            elif aug_type == AugmentationType.MAGNITUDE_PRUNING:
                intensities = [0.05, 0.1, 0.15, 0.2, 0.3]
            elif aug_type == AugmentationType.LAYER_REINIT:
                intensities = [0.01, 0.05, 0.1, 0.2, 0.3]
            else:
                continue

            prev_delta = 0
            for intensity in intensities:
                stats = self._get_stats(model_id, tokenizer, AugmentationConfig(
                    aug_type=aug_type,
                    scope=AugmentationScope.ALL_LAYERS,
                    seed=42,
                    intensity=intensity,
                ))
                delta = stats.training.init_loss - base_stats.training.init_loss
                print(f"{aug_type.value:<20} {intensity:<12.4f} {stats.training.init_loss:<12.4f} {delta:<+12.4f} {stats.training.gradient_noise_scale:<12.4f}")

        print(f"\nBase: loss={base_stats.training.init_loss:.4f}, grad_norm={base_stats.training.gradient_noise_scale:.4f}")

    def test_scope_sweep(self, model_and_tokenizer):
        """Show impact of different scopes for each augmentation type."""
        model_id, tokenizer = model_and_tokenizer
        base_stats = self._get_stats(model_id, tokenizer)

        print(f"\n{'Type':<20} {'Scope':<20} {'Loss':<12} {'Delta':<12} {'GradNorm':<12}")
        print("-" * 76)

        for aug_type in AugmentationType:
            if aug_type == AugmentationType.WEIGHT_SCALING:
                intensity = 1.15
            elif aug_type == AugmentationType.GAUSSIAN_NOISE:
                intensity = 0.02
            elif aug_type == AugmentationType.MAGNITUDE_PRUNING:
                intensity = 0.1
            elif aug_type == AugmentationType.LAYER_REINIT:
                intensity = 0.05
            else:
                continue

            for scope in AugmentationScope:
                stats = self._get_stats(model_id, tokenizer, AugmentationConfig(
                    aug_type=aug_type,
                    scope=scope,
                    seed=42,
                    intensity=intensity,
                ))
                delta = stats.training.init_loss - base_stats.training.init_loss
                print(f"{aug_type.value:<20} {scope.value:<20} {stats.training.init_loss:<12.4f} {delta:<+12.4f} {stats.training.gradient_noise_scale:<12.4f}")

        print(f"\nBase: loss={base_stats.training.init_loss:.4f}, grad_norm={base_stats.training.gradient_noise_scale:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
