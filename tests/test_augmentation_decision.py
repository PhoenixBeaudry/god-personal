"""
Tests for the augmentation decision function (maybe_get_augmentation_config).
Run with: python -m pytest tests/test_augmentation_decision.py -v
"""

import pytest

from core.models.utility_models import AugmentationConfig
from core.models.utility_models import AugmentationScope
from core.models.utility_models import AugmentationType
from core.models.utility_models import TaskType
from validator.tasks.synthetic_scheduler import maybe_get_augmentation_config


def test_returns_none_when_text_disabled(monkeypatch):
    monkeypatch.setattr("validator.core.constants.AUGMENTATION_ENABLED_TEXT", False)
    monkeypatch.setattr("validator.core.constants.AUGMENTATION_PROBABILITY", 1.0)
    result = maybe_get_augmentation_config(TaskType.INSTRUCTTEXTTASK)
    assert result is None


def test_returns_none_when_image_disabled(monkeypatch):
    monkeypatch.setattr("validator.core.constants.AUGMENTATION_ENABLED_IMAGE", False)
    monkeypatch.setattr("validator.core.constants.AUGMENTATION_PROBABILITY", 1.0)
    result = maybe_get_augmentation_config(TaskType.IMAGETASK)
    assert result is None


def test_returns_none_when_env_disabled(monkeypatch):
    monkeypatch.setattr("validator.core.constants.AUGMENTATION_ENABLED_ENV", False)
    monkeypatch.setattr("validator.core.constants.AUGMENTATION_PROBABILITY", 1.0)
    result = maybe_get_augmentation_config(TaskType.ENVIRONMENTTASK)
    assert result is None


def test_returns_config_when_enabled(monkeypatch):
    monkeypatch.setattr("validator.core.constants.AUGMENTATION_ENABLED_TEXT", True)
    monkeypatch.setattr("validator.core.constants.AUGMENTATION_PROBABILITY", 1.0)
    result = maybe_get_augmentation_config(TaskType.INSTRUCTTEXTTASK)
    assert isinstance(result, AugmentationConfig)
    assert isinstance(result.aug_type, AugmentationType)
    assert isinstance(result.scope, AugmentationScope)
    assert isinstance(result.seed, int)
    assert isinstance(result.intensity, float)


def test_zero_probability_returns_none(monkeypatch):
    monkeypatch.setattr("validator.core.constants.AUGMENTATION_ENABLED_TEXT", True)
    monkeypatch.setattr("validator.core.constants.AUGMENTATION_PROBABILITY", 0.0)
    for _ in range(50):
        result = maybe_get_augmentation_config(TaskType.INSTRUCTTEXTTASK)
        assert result is None


def test_probability_distribution(monkeypatch):
    monkeypatch.setattr("validator.core.constants.AUGMENTATION_ENABLED_TEXT", True)
    monkeypatch.setattr("validator.core.constants.AUGMENTATION_PROBABILITY", 0.5)
    results = [maybe_get_augmentation_config(TaskType.INSTRUCTTEXTTASK) for _ in range(1000)]
    augmented = sum(1 for r in results if r is not None)
    # Should be roughly 50% ± some variance
    assert 350 < augmented < 650


def test_seed_reproducibility(monkeypatch):
    """Same seed should produce identical config (aug_type, scope, intensity)."""
    monkeypatch.setattr("validator.core.constants.AUGMENTATION_ENABLED_TEXT", True)
    monkeypatch.setattr("validator.core.constants.AUGMENTATION_PROBABILITY", 1.0)
    configs = []
    for _ in range(100):
        c = maybe_get_augmentation_config(TaskType.INSTRUCTTEXTTASK)
        if c is not None:
            configs.append(c)

    # Each config's seed should deterministically produce its own aug_type/scope/intensity
    for config in configs[:10]:
        import random
        rng = random.Random(config.seed)
        # Re-derive from seed — should match
        from validator.tasks.synthetic_scheduler import _weighted_choice, _seeded_intensity
        import validator.core.constants as vcst
        aug_type = _weighted_choice(vcst.AUGMENTATION_TYPE_WEIGHTS, rng)
        scope = _weighted_choice(vcst.AUGMENTATION_SCOPE_WEIGHTS, rng)
        intensity = _seeded_intensity(aug_type, rng)
        assert config.aug_type == aug_type
        assert config.scope == scope
        assert abs(config.intensity - intensity) < 1e-10


def test_all_text_subtypes_use_text_flag(monkeypatch):
    monkeypatch.setattr("validator.core.constants.AUGMENTATION_ENABLED_TEXT", True)
    monkeypatch.setattr("validator.core.constants.AUGMENTATION_PROBABILITY", 1.0)
    for task_type in [TaskType.INSTRUCTTEXTTASK, TaskType.DPOTASK, TaskType.GRPOTASK, TaskType.CHATTASK]:
        result = maybe_get_augmentation_config(task_type)
        assert result is not None, f"Expected config for {task_type}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
