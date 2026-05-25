import pytest

import validator.shared.constants as cst
from core.models.tournament_models import TournamentType
from core.models.utility_models import TaskType
from validator.tournament import constants as t_cst
from validator.tournament.specs import get_tournament_base_weight
from validator.tournament.specs import get_tournament_max_weight
from validator.tournament.specs import get_tournament_schedule
from validator.tournament.specs import get_tournament_spec
from validator.tournament.specs import get_tournament_type_for_task_type
from validator.tournament.specs import tournament_types


def test_tournament_specs_match_legacy_constants():
    text = get_tournament_spec(TournamentType.TEXT)
    image = get_tournament_spec(TournamentType.IMAGE)
    environment = get_tournament_spec(TournamentType.ENVIRONMENT)

    assert text.base_weight == cst.TOURNAMENT_TEXT_WEIGHT
    assert image.base_weight == cst.TOURNAMENT_IMAGE_WEIGHT
    assert environment.base_weight == cst.TOURNAMENT_ENVIRONMENT_WEIGHT

    assert text.max_weight == cst.MAX_TEXT_TOURNAMENT_WEIGHT
    assert image.max_weight == cst.MAX_IMAGE_TOURNAMENT_WEIGHT
    assert environment.max_weight == cst.MAX_ENVIRONMENT_TOURNAMENT_WEIGHT

    assert text.participation_fee_rao == t_cst.TOURNAMENT_TEXT_PARTICIPATION_FEE_RAO
    assert image.participation_fee_rao == t_cst.TOURNAMENT_IMAGE_PARTICIPATION_FEE_RAO
    assert environment.participation_fee_rao == t_cst.TOURNAMENT_ENVIRONMENT_PARTICIPATION_FEE_RAO

    assert text.minimum_participants == cst.MIN_MINERS_FOR_TOURN
    assert image.minimum_participants == cst.MIN_MINERS_FOR_TOURN
    assert environment.minimum_participants == cst.MIN_MINERS_FOR_ENV_TOURN


def test_tournament_specs_capture_task_contracts():
    assert tournament_types() == (TournamentType.TEXT, TournamentType.IMAGE, TournamentType.ENVIRONMENT)
    assert get_tournament_spec("text").benchmark_task_types == (
        TaskType.INSTRUCTTEXTTASK,
        TaskType.CHATTASK,
        TaskType.DPOTASK,
        TaskType.GRPOTASK,
    )
    assert get_tournament_spec("image").benchmark_task_types == (TaskType.IMAGETASK,)
    assert get_tournament_spec("environment").benchmark_task_types == (TaskType.ENVIRONMENTTASK,)
    assert get_tournament_spec("environment").uses_environment_rounds is True


def test_tournament_specs_expose_legacy_response_field_names():
    text = get_tournament_spec(TournamentType.TEXT)
    image = get_tournament_spec(TournamentType.IMAGE)
    environment = get_tournament_spec(TournamentType.ENVIRONMENT)

    assert text.audit_data_field == "text_tournament_data"
    assert image.audit_weight_field == "image_tournament_weight"
    assert environment.performance_diff_field == "environment_performance_diff"
    assert environment.burn_proportion_field == "environment_burn_proportion"


def test_tournament_specs_map_task_types_to_tournament_types():
    assert get_tournament_type_for_task_type(TaskType.INSTRUCTTEXTTASK) == TournamentType.TEXT
    assert get_tournament_type_for_task_type(TaskType.CHATTASK) == TournamentType.TEXT
    assert get_tournament_type_for_task_type(TaskType.DPOTASK) == TournamentType.TEXT
    assert get_tournament_type_for_task_type(TaskType.GRPOTASK) == TournamentType.TEXT
    assert get_tournament_type_for_task_type(TaskType.IMAGETASK) == TournamentType.IMAGE
    assert get_tournament_type_for_task_type(TaskType.ENVIRONMENTTASK) == TournamentType.ENVIRONMENT


def test_tournament_spec_helpers_accept_string_values():
    assert get_tournament_base_weight("text") == cst.TOURNAMENT_TEXT_WEIGHT
    assert get_tournament_max_weight("image") == cst.MAX_IMAGE_TOURNAMENT_WEIGHT
    assert get_tournament_schedule("environment") == (
        cst.TOURNAMENT_SCHEDULE_ENVIRONMENT_DAY_OF_WEEK,
        cst.TOURNAMENT_SCHEDULE_ENVIRONMENT_HOUR,
    )


def test_tournament_spec_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unknown tournament type"):
        get_tournament_spec("audio")
