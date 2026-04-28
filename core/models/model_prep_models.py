"""
Pydantic models for model prep: augmentation config and baseline stats.
"""

from enum import Enum

from pydantic import BaseModel


# --- Augmentation models ---

class AugmentationType(str, Enum):
    GAUSSIAN_NOISE = "gaussian_noise"
    WEIGHT_SCALING = "weight_scaling"
    MAGNITUDE_PRUNING = "magnitude_pruning"
    LAYER_REINIT = "layer_reinit"


class AugmentationScope(str, Enum):
    SINGLE_LAYER = "single_layer"
    LAYER_TYPE_GROUP = "layer_type_group"
    MULTI_LAYER = "multi_layer"
    ALL_LAYERS = "all_layers"


class AugmentationConfig(BaseModel):
    aug_type: AugmentationType
    scope: AugmentationScope
    seed: int
    intensity: float


# --- Stats models ---

class SeqLengthDistribution(BaseModel):
    mean: float
    p50: int
    p95: int
    p99: int
    max: int


class DatasetStats(BaseModel):
    total_tokens: int
    seq_length_distribution: SeqLengthDistribution
    near_duplicate_rate: float
    bits_per_byte: float
    vocab_size: int


class LayerGroupWeightStats(BaseModel):
    weight_rms: float
    weight_norm: float
    max_abs: float


class WeightStats(BaseModel):
    by_group: dict[str, LayerGroupWeightStats]


class LayerGradStats(BaseModel):
    frobenius_norm: float
    rms: float
    max_abs: float
    top_singular_values: list[float]


class TrainingDynamics(BaseModel):
    init_loss: float
    grad_norms: dict[str, float]
    gradient_noise_scale: float
    activation_rms: dict[str, float]
    grad_stats: dict[str, LayerGradStats]
    output_entropy: float


class BaselineStats(BaseModel):
    dataset: DatasetStats
    weights: WeightStats
    training: TrainingDynamics
