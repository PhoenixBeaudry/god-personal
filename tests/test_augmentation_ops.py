"""
Tests for augmentation operations: layer selection and weight modification.
Run with: python -m pytest tests/test_augmentation_ops.py -v
"""

import numpy as np
import torch
import pytest

from core.models.utility_models import AugmentationConfig
from core.models.utility_models import AugmentationScope
from core.models.utility_models import AugmentationType
from trainer.model_prep.augmentation import apply_augmentation
from trainer.model_prep.augmentation import augment_model
from trainer.model_prep.augmentation import select_target_layers


# --- Layer selection tests ---

SAMPLE_LAYER_NAMES = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.1.self_attn.q_proj.weight",
    "model.layers.1.self_attn.k_proj.weight",
    "model.layers.1.self_attn.v_proj.weight",
    "model.layers.1.self_attn.o_proj.weight",
    "model.layers.1.mlp.gate_proj.weight",
    "model.layers.1.mlp.up_proj.weight",
    "model.layers.1.mlp.down_proj.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.1.input_layernorm.weight",
    "model.embed_tokens.weight",
    "lm_head.weight",
]


def test_single_layer_returns_one():
    result = select_target_layers(SAMPLE_LAYER_NAMES, AugmentationScope.SINGLE_LAYER, seed=42)
    assert len(result) == 1
    assert "norm" not in result[0].lower()
    assert "embed" not in result[0].lower()


def test_single_layer_deterministic():
    a = select_target_layers(SAMPLE_LAYER_NAMES, AugmentationScope.SINGLE_LAYER, seed=42)
    b = select_target_layers(SAMPLE_LAYER_NAMES, AugmentationScope.SINGLE_LAYER, seed=42)
    assert a == b


def test_single_layer_different_seeds():
    results = set()
    for seed in range(100):
        r = select_target_layers(SAMPLE_LAYER_NAMES, AugmentationScope.SINGLE_LAYER, seed=seed)
        results.add(r[0])
    assert len(results) > 1


def test_layer_type_group_all_same_type():
    result = select_target_layers(SAMPLE_LAYER_NAMES, AugmentationScope.LAYER_TYPE_GROUP, seed=42)
    assert len(result) >= 1
    # All results should share the same type suffix
    types = set()
    for name in result:
        suffix = name.split(".")[-2]  # e.g., "q_proj"
        types.add(suffix)
    assert len(types) == 1


def test_multi_layer_returns_subset():
    result = select_target_layers(SAMPLE_LAYER_NAMES, AugmentationScope.MULTI_LAYER, seed=42)
    weight_layers = [n for n in SAMPLE_LAYER_NAMES if "weight" in n and "norm" not in n.lower() and "embed" not in n.lower()]
    assert 1 <= len(result) <= len(weight_layers)
    assert len(result) < len(weight_layers)  # should be a proper subset most of the time


def test_all_layers_returns_all_weights():
    result = select_target_layers(SAMPLE_LAYER_NAMES, AugmentationScope.ALL_LAYERS, seed=42)
    expected = [n for n in SAMPLE_LAYER_NAMES if "weight" in n and "norm" not in n.lower() and "embed" not in n.lower()]
    assert result == expected


def test_excludes_norms_and_embeddings():
    for scope in AugmentationScope:
        result = select_target_layers(SAMPLE_LAYER_NAMES, scope, seed=42)
        for name in result:
            assert "norm" not in name.lower()
            assert "embed" not in name.lower()


# --- Augmentation operation tests ---

def make_tensor(shape=(64, 64), seed=0):
    rng = np.random.default_rng(seed)
    data = rng.normal(0, 1, size=shape).astype(np.float32)
    return torch.from_numpy(data)


def test_gaussian_noise_changes_tensor():
    t = make_tensor()
    rng = np.random.default_rng(42)
    result = apply_augmentation(t, AugmentationType.GAUSSIAN_NOISE, 0.01, rng)
    assert not torch.equal(t, result)


def test_gaussian_noise_deterministic():
    t = make_tensor()
    r1 = apply_augmentation(t.clone(), AugmentationType.GAUSSIAN_NOISE, 0.01, np.random.default_rng(42))
    r2 = apply_augmentation(t.clone(), AugmentationType.GAUSSIAN_NOISE, 0.01, np.random.default_rng(42))
    assert torch.equal(r1, r2)


def test_gaussian_noise_scales_with_intensity():
    t = make_tensor()
    low = apply_augmentation(t.clone(), AugmentationType.GAUSSIAN_NOISE, 0.001, np.random.default_rng(42))
    high = apply_augmentation(t.clone(), AugmentationType.GAUSSIAN_NOISE, 0.1, np.random.default_rng(42))
    diff_low = (t - low).abs().mean().item()
    diff_high = (t - high).abs().mean().item()
    assert diff_high > diff_low


def test_weight_scaling():
    t = make_tensor()
    result = apply_augmentation(t, AugmentationType.WEIGHT_SCALING, 1.5, np.random.default_rng(42))
    expected = t * 1.5
    assert torch.allclose(result, expected, atol=1e-5)


def test_magnitude_pruning_zeros_correct_fraction():
    t = make_tensor(shape=(100, 100))
    intensity = 0.1  # prune bottom 10%
    result = apply_augmentation(t, AugmentationType.MAGNITUDE_PRUNING, intensity, np.random.default_rng(42))
    zero_fraction = (result == 0).float().mean().item()
    assert zero_fraction >= 0.08  # roughly 10%, allow some tolerance
    assert zero_fraction <= 0.15


def test_magnitude_pruning_preserves_large_weights():
    t = make_tensor(shape=(100, 100))
    result = apply_augmentation(t, AugmentationType.MAGNITUDE_PRUNING, 0.1, np.random.default_rng(42))
    # The largest weight should survive pruning
    max_idx = t.abs().argmax()
    assert result.flatten()[max_idx] != 0


def test_layer_reinit_modifies_fraction():
    t = make_tensor(shape=(100, 100))
    intensity = 0.1
    result = apply_augmentation(t, AugmentationType.LAYER_REINIT, intensity, np.random.default_rng(42))
    changed = (t != result).float().mean().item()
    assert changed >= 0.05  # at least some changed
    assert changed <= 0.20  # not too many


def test_layer_reinit_deterministic():
    t = make_tensor()
    r1 = apply_augmentation(t.clone(), AugmentationType.LAYER_REINIT, 0.1, np.random.default_rng(42))
    r2 = apply_augmentation(t.clone(), AugmentationType.LAYER_REINIT, 0.1, np.random.default_rng(42))
    assert torch.equal(r1, r2)


def test_preserves_dtype():
    for dtype in [torch.float16, torch.float32, torch.bfloat16]:
        t = make_tensor().to(dtype)
        for aug_type in AugmentationType:
            result = apply_augmentation(t.clone(), aug_type, 0.01, np.random.default_rng(42))
            assert result.dtype == dtype, f"dtype mismatch for {aug_type}: {result.dtype} != {dtype}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
