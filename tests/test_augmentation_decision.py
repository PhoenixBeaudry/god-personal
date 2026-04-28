"""
Tests for the augmentation decision logic.
Run with: python -m pytest tests/test_augmentation_decision.py -v -o addopts=
"""

import random

import pytest

from core.models.model_prep_models import AugmentationConfig
from core.models.model_prep_models import AugmentationScope
from core.models.model_prep_models import AugmentationType
from core.models.utility_models import TaskType
from validator.utils.augmentation_decision import (
    maybe_get_augmentation_config,
    seeded_intensity,
    weighted_choice,
)


def test_returns_none_when_text_disabled(monkeypatch):
    monkeypatch.setattr("validator.utils.augmentation_decision.vcst.AUGMENTATION_ENABLED_TEXT", False)
    monkeypatch.setattr("validator.utils.augmentation_decision.vcst.AUGMENTATION_PROBABILITY", 1.0)
    result = maybe_get_augmentation_config(TaskType.INSTRUCTTEXTTASK)
    assert result is None


def test_returns_none_when_image_disabled(monkeypatch):
    monkeypatch.setattr("validator.utils.augmentation_decision.vcst.AUGMENTATION_ENABLED_IMAGE", False)
    monkeypatch.setattr("validator.utils.augmentation_decision.vcst.AUGMENTATION_PROBABILITY", 1.0)
    result = maybe_get_augmentation_config(TaskType.IMAGETASK)
    assert result is None


def test_returns_none_when_env_disabled(monkeypatch):
    monkeypatch.setattr("validator.utils.augmentation_decision.vcst.AUGMENTATION_ENABLED_ENV", False)
    monkeypatch.setattr("validator.utils.augmentation_decision.vcst.AUGMENTATION_PROBABILITY", 1.0)
    result = maybe_get_augmentation_config(TaskType.ENVIRONMENTTASK)
    assert result is None


def test_returns_config_when_enabled(monkeypatch):
    monkeypatch.setattr("validator.utils.augmentation_decision.vcst.AUGMENTATION_ENABLED_TEXT", True)
    monkeypatch.setattr("validator.utils.augmentation_decision.vcst.AUGMENTATION_PROBABILITY", 1.0)
    result = maybe_get_augmentation_config(TaskType.INSTRUCTTEXTTASK)
    assert isinstance(result, AugmentationConfig)
    assert isinstance(result.aug_type, AugmentationType)
    assert isinstance(result.scope, AugmentationScope)
    assert isinstance(result.seed, int)
    assert isinstance(result.intensity, float)


def test_zero_probability_returns_none(monkeypatch):
    monkeypatch.setattr("validator.utils.augmentation_decision.vcst.AUGMENTATION_ENABLED_TEXT", True)
    monkeypatch.setattr("validator.utils.augmentation_decision.vcst.AUGMENTATION_PROBABILITY", 0.0)
    for _ in range(50):
        result = maybe_get_augmentation_config(TaskType.INSTRUCTTEXTTASK)
        assert result is None


def test_probability_distribution(monkeypatch):
    monkeypatch.setattr("validator.utils.augmentation_decision.vcst.AUGMENTATION_ENABLED_TEXT", True)
    monkeypatch.setattr("validator.utils.augmentation_decision.vcst.AUGMENTATION_PROBABILITY", 0.5)
    results = [maybe_get_augmentation_config(TaskType.INSTRUCTTEXTTASK) for _ in range(1000)]
    augmented = sum(1 for r in results if r is not None)
    assert 350 < augmented < 650


def test_seed_reproducibility(monkeypatch):
    """Same seed should produce identical config."""
    monkeypatch.setattr("validator.utils.augmentation_decision.vcst.AUGMENTATION_ENABLED_TEXT", True)
    monkeypatch.setattr("validator.utils.augmentation_decision.vcst.AUGMENTATION_PROBABILITY", 1.0)

    import validator.core.constants as vcst

    configs = [maybe_get_augmentation_config(TaskType.INSTRUCTTEXTTASK) for _ in range(20)]

    for config in configs:
        assert config is not None
        rng = random.Random(config.seed)
        aug_type = weighted_choice(vcst.AUGMENTATION_TYPE_WEIGHTS, rng)
        scope = weighted_choice(vcst.AUGMENTATION_SCOPE_WEIGHTS, rng)
        intensity = seeded_intensity(aug_type, rng)
        assert config.aug_type == aug_type
        assert config.scope == scope
        assert abs(config.intensity - intensity) < 1e-10


def test_all_text_subtypes_use_text_flag(monkeypatch):
    monkeypatch.setattr("validator.utils.augmentation_decision.vcst.AUGMENTATION_ENABLED_TEXT", True)
    monkeypatch.setattr("validator.utils.augmentation_decision.vcst.AUGMENTATION_PROBABILITY", 1.0)
    for task_type in [TaskType.INSTRUCTTEXTTASK, TaskType.DPOTASK, TaskType.GRPOTASK, TaskType.CHATTASK]:
        result = maybe_get_augmentation_config(task_type)
        assert result is not None, f"Expected config for {task_type}"


def test_weighted_choice_normalises():
    """Weights don't need to sum to 1."""
    weights = {AugmentationType.GAUSSIAN_NOISE: 10, AugmentationType.WEIGHT_SCALING: 90}
    rng = random.Random(42)
    results = [weighted_choice(weights, rng) for _ in range(1000)]
    noise_count = sum(1 for r in results if r == AugmentationType.GAUSSIAN_NOISE)
    assert 50 < noise_count < 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
