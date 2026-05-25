import random

import validator.shared.constants as vcst
from core.models.model_prep_models import AugmentationScope
from core.models.model_prep_models import AugmentationType
from core.models.utility_models import TaskType
from validator.tasks.augmentation import augmentation_enabled_for_task
from validator.tasks.augmentation import maybe_get_augmentation_config


def test_augmentation_enabled_for_task_uses_task_families(monkeypatch):
    monkeypatch.setattr(vcst, "AUGMENTATION_ENABLED_TEXT", True)
    monkeypatch.setattr(vcst, "AUGMENTATION_ENABLED_IMAGE", False)
    monkeypatch.setattr(vcst, "AUGMENTATION_ENABLED_ENV", False)

    assert augmentation_enabled_for_task(TaskType.INSTRUCTTEXTTASK) is True
    assert augmentation_enabled_for_task(TaskType.DPOTASK) is True
    assert augmentation_enabled_for_task(TaskType.GRPOTASK) is True
    assert augmentation_enabled_for_task(TaskType.CHATTASK) is True
    assert augmentation_enabled_for_task(TaskType.IMAGETASK) is False
    assert augmentation_enabled_for_task(TaskType.ENVIRONMENTTASK) is False


def test_maybe_get_augmentation_config_returns_none_when_family_disabled(monkeypatch):
    monkeypatch.setattr(vcst, "AUGMENTATION_ENABLED_IMAGE", False)
    monkeypatch.setattr(vcst, "AUGMENTATION_PROBABILITY", 1.0)

    assert maybe_get_augmentation_config(TaskType.IMAGETASK) is None


def test_maybe_get_augmentation_config_is_seed_reproducible(monkeypatch):
    monkeypatch.setattr(vcst, "AUGMENTATION_ENABLED_TEXT", True)
    monkeypatch.setattr(vcst, "AUGMENTATION_PROBABILITY", 1.0)
    monkeypatch.setattr(vcst, "AUGMENTATION_TYPE_WEIGHTS", {AugmentationType.GAUSSIAN_NOISE: 1.0})
    monkeypatch.setattr(vcst, "AUGMENTATION_SCOPE_WEIGHTS", {AugmentationScope.SINGLE_LAYER: 1.0})
    monkeypatch.setattr(vcst, "AUGMENTATION_INTENSITY_RANGES", {AugmentationType.GAUSSIAN_NOISE: (0.1, 0.2)})
    monkeypatch.setattr(random, "random", lambda: 0.0)
    monkeypatch.setattr(random, "randint", lambda _low, _high: 1234)

    config = maybe_get_augmentation_config(TaskType.GRPOTASK)
    repeated_config = maybe_get_augmentation_config(TaskType.GRPOTASK)

    assert config is not None
    assert repeated_config is not None
    assert config == repeated_config
    assert config.aug_type == AugmentationType.GAUSSIAN_NOISE
    assert config.scope == AugmentationScope.SINGLE_LAYER
    assert config.seed == 1234
    assert 0.1 <= config.intensity <= 0.2
