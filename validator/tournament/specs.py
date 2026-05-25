from dataclasses import dataclass

import validator.shared.constants as cst
from core.models.tournament_models import TournamentType
from core.models.utility_models import TEXT_TOURNAMENT_TASK_TYPES
from core.models.utility_models import TaskType
from validator.tournament import constants as t_cst


RAO_PER_TAO = 1_000_000_000


@dataclass(frozen=True)
class TournamentSpec:
    """Runtime configuration for one tournament family.

    The spec stores constant names instead of copied values so existing tests,
    config overrides, and monkeypatches that change the legacy constants keep
    working exactly as before.
    """

    tournament_type: TournamentType
    base_weight_constant: str
    max_weight_constant: str
    participation_fee_constant: str
    minimum_participants_constant: str
    schedule_day_constant: str
    schedule_hour_constant: str
    task_types: tuple[TaskType, ...]
    benchmark_task_types: tuple[TaskType, ...]
    uses_environment_rounds: bool = False

    @property
    def base_weight(self) -> float:
        return getattr(cst, self.base_weight_constant)

    @property
    def max_weight(self) -> float:
        return getattr(cst, self.max_weight_constant)

    @property
    def participation_fee_rao(self) -> int:
        return getattr(t_cst, self.participation_fee_constant)

    @property
    def participation_fee_label(self) -> str:
        return f"{self.participation_fee_rao / RAO_PER_TAO:g} TAO"

    @property
    def minimum_participants(self) -> int:
        return getattr(cst, self.minimum_participants_constant)

    @property
    def schedule(self) -> tuple[int, int]:
        return (getattr(cst, self.schedule_day_constant), getattr(cst, self.schedule_hour_constant))

    @property
    def audit_data_field(self) -> str:
        return f"{self.tournament_type.value}_tournament_data"

    @property
    def audit_weight_field(self) -> str:
        return f"{self.tournament_type.value}_tournament_weight"

    @property
    def performance_diff_field(self) -> str:
        return f"{self.tournament_type.value}_performance_diff"

    @property
    def burn_proportion_field(self) -> str:
        return f"{self.tournament_type.value}_burn_proportion"


TOURNAMENT_SPECS: dict[TournamentType, TournamentSpec] = {
    TournamentType.TEXT: TournamentSpec(
        tournament_type=TournamentType.TEXT,
        base_weight_constant="TOURNAMENT_TEXT_WEIGHT",
        max_weight_constant="MAX_TEXT_TOURNAMENT_WEIGHT",
        participation_fee_constant="TOURNAMENT_TEXT_PARTICIPATION_FEE_RAO",
        minimum_participants_constant="MIN_MINERS_FOR_TOURN",
        schedule_day_constant="TOURNAMENT_SCHEDULE_TEXT_DAY_OF_WEEK",
        schedule_hour_constant="TOURNAMENT_SCHEDULE_TEXT_HOUR",
        task_types=TEXT_TOURNAMENT_TASK_TYPES,
        benchmark_task_types=TEXT_TOURNAMENT_TASK_TYPES,
    ),
    TournamentType.IMAGE: TournamentSpec(
        tournament_type=TournamentType.IMAGE,
        base_weight_constant="TOURNAMENT_IMAGE_WEIGHT",
        max_weight_constant="MAX_IMAGE_TOURNAMENT_WEIGHT",
        participation_fee_constant="TOURNAMENT_IMAGE_PARTICIPATION_FEE_RAO",
        minimum_participants_constant="MIN_MINERS_FOR_TOURN",
        schedule_day_constant="TOURNAMENT_SCHEDULE_IMAGE_DAY_OF_WEEK",
        schedule_hour_constant="TOURNAMENT_SCHEDULE_IMAGE_HOUR",
        task_types=(TaskType.IMAGETASK,),
        benchmark_task_types=(TaskType.IMAGETASK,),
    ),
    TournamentType.ENVIRONMENT: TournamentSpec(
        tournament_type=TournamentType.ENVIRONMENT,
        base_weight_constant="TOURNAMENT_ENVIRONMENT_WEIGHT",
        max_weight_constant="MAX_ENVIRONMENT_TOURNAMENT_WEIGHT",
        participation_fee_constant="TOURNAMENT_ENVIRONMENT_PARTICIPATION_FEE_RAO",
        minimum_participants_constant="MIN_MINERS_FOR_ENV_TOURN",
        schedule_day_constant="TOURNAMENT_SCHEDULE_ENVIRONMENT_DAY_OF_WEEK",
        schedule_hour_constant="TOURNAMENT_SCHEDULE_ENVIRONMENT_HOUR",
        task_types=(TaskType.ENVIRONMENTTASK,),
        benchmark_task_types=(TaskType.ENVIRONMENTTASK,),
        uses_environment_rounds=True,
    ),
}


def get_tournament_spec(tournament_type: TournamentType | str) -> TournamentSpec:
    try:
        normalized_type = TournamentType(tournament_type)
    except ValueError as exc:
        raise ValueError(f"Unknown tournament type: {tournament_type}") from exc

    return TOURNAMENT_SPECS[normalized_type]


def get_tournament_type_for_task_type(task_type: TaskType | str) -> TournamentType:
    normalized_task_type = task_type if isinstance(task_type, TaskType) else TaskType(task_type)
    for tournament_type, spec in TOURNAMENT_SPECS.items():
        if normalized_task_type in spec.task_types:
            return tournament_type

    raise ValueError(f"No tournament type configured for task type: {task_type}")


def tournament_types() -> tuple[TournamentType, ...]:
    return tuple(TOURNAMENT_SPECS)


def get_tournament_base_weight(tournament_type: TournamentType | str) -> float:
    return get_tournament_spec(tournament_type).base_weight


def get_tournament_max_weight(tournament_type: TournamentType | str) -> float:
    return get_tournament_spec(tournament_type).max_weight


def get_tournament_schedule(tournament_type: TournamentType | str) -> tuple[int, int]:
    return get_tournament_spec(tournament_type).schedule
