from datetime import datetime

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class QualityMetrics(BaseModel):
    total_score: float
    total_count: int
    total_success: int
    total_quality: int
    avg_quality_score: float
    success_rate: float
    quality_rate: float


class WorkloadMetrics(BaseModel):
    competition_hours: int = Field(ge=0)
    total_params_billions: float = Field(ge=0.0)


class ModelMetrics(BaseModel):
    modal_model: str
    unique_models: int = Field(ge=0)
    unique_datasets: int = Field(ge=0)


class NodeStats(BaseModel):
    quality_metrics: QualityMetrics
    workload_metrics: WorkloadMetrics
    model_metrics: ModelMetrics

    model_config = ConfigDict(protected_namespaces=())


class AllNodeStats(BaseModel):
    daily: NodeStats
    three_day: NodeStats
    weekly: NodeStats
    monthly: NodeStats
    all_time: NodeStats

    @classmethod
    def get_periods_sql_mapping(cls) -> dict[str, str]:
        return {"daily": "24 hours", "three_day": "3 days", "weekly": "7 days", "monthly": "30 days", "all_time": "all"}


class NetworkStats(BaseModel):
    number_of_jobs_training: int
    number_of_jobs_preevaluation: int
    number_of_jobs_evaluating: int
    number_of_jobs_success: int
    next_training_end: datetime | None
    job_can_be_made: bool = True


class DetailedNetworkStats(NetworkStats):
    instruct_training: int = 0
    instruct_preevaluation: int = 0
    instruct_evaluating: int = 0
    instruct_success: int = 0

    dpo_training: int = 0
    dpo_preevaluation: int = 0
    dpo_evaluating: int = 0
    dpo_success: int = 0

    grpo_training: int = 0
    grpo_preevaluation: int = 0
    grpo_evaluating: int = 0
    grpo_success: int = 0

    image_training: int = 0
    image_preevaluation: int = 0
    image_evaluating: int = 0
    image_success: int = 0

