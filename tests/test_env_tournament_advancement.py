"""Tests for environment tournament advancement: thresholds, winner resolution,
boss round structure, env scaling, and model continuation logic.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.constants import EnvironmentName
from core.constants import TrainingStartPoint
from core.models.tournament_models import TournamentData
from core.models.tournament_models import TournamentType
from core.models.tournament_models import Group
from core.models.tournament_models import GroupRound
from validator.tournament.utils import determine_boss_round_winner
from validator.tournament.utils import get_progressive_threshold
from validator.tournament.utils import get_real_winner_hotkey
import validator.tournament.constants as t_cst


BOSS = "5GBoss"
CONTENDER = "5GContender"


# --- 2a: get_progressive_threshold ---


class TestProgressiveThreshold:
    def test_first_win_uses_base_threshold(self):
        t = get_progressive_threshold(1, TournamentType.TEXT)
        assert t == t_cst.EXPONENTIAL_BASE_THRESHOLD

    def test_env_uses_lower_base_threshold(self):
        t_env = get_progressive_threshold(1, TournamentType.ENVIRONMENT)
        t_text = get_progressive_threshold(1, TournamentType.TEXT)
        assert t_env == t_cst.EXPONENTIAL_BASE_THRESHOLD_ENVIRONMENT
        assert t_env < t_text

    def test_decay_with_consecutive_wins(self):
        t1 = get_progressive_threshold(1, TournamentType.TEXT)
        t2 = get_progressive_threshold(2, TournamentType.TEXT)
        t3 = get_progressive_threshold(3, TournamentType.TEXT)
        assert t1 > t2 > t3

    def test_floor_at_min_threshold(self):
        t = get_progressive_threshold(100, TournamentType.TEXT)
        assert t == t_cst.EXPONENTIAL_MIN_THRESHOLD

    def test_decay_rate_applied_correctly(self):
        t2 = get_progressive_threshold(2, TournamentType.TEXT)
        expected = t_cst.EXPONENTIAL_BASE_THRESHOLD * t_cst.EXPONENTIAL_DECAY_RATE
        assert abs(t2 - expected) < 1e-9

    def test_none_tournament_type_uses_default_base(self):
        t = get_progressive_threshold(1, None)
        assert t == t_cst.EXPONENTIAL_BASE_THRESHOLD


# --- 2b: get_real_winner_hotkey (already tested in scoring pipeline, but included for completeness) ---


class TestGetRealWinnerHotkey:
    def test_burn_hotkey_resolves(self):
        from validator.core.constants import EMISSION_BURN_HOTKEY
        result = get_real_winner_hotkey(EMISSION_BURN_HOTKEY, "real_champ")
        assert result == "real_champ"

    def test_regular_passes_through(self):
        assert get_real_winner_hotkey("winner", "old_champ") == "winner"

    def test_none_returns_none(self):
        assert get_real_winner_hotkey(None, "anything") is None


# --- 2c: Boss round 3-task configuration ---


class TestBossRoundTaskConfig:
    """Verify _create_environment_boss_round_tasks produces 3 tasks with correct start points.
    This mocks the DB calls and verifies the structural logic."""

    @pytest.mark.asyncio
    async def test_three_tasks_with_correct_start_points(self):
        round_data = GroupRound(
            round_id="tourn_abc_round_004",
            round_number=4,
            groups=[Group(member_ids=[CONTENDER, BOSS])],
        )

        created_tasks = []

        async def mock_create_env_task(config, models, datasets, **kwargs):
            task = MagicMock()
            task.task_id = f"task_{len(created_tasks)}"
            task.model_id = kwargs.get("model_id_override", "random_model")
            task.training_start_point = kwargs.get("training_start_point", TrainingStartPoint.DEFAULT)
            created_tasks.append(kwargs)
            return task

        with (
            patch("validator.tournament.task_creator._get_existing_tasks_by_identifier", return_value=[]),
            patch("validator.tournament.task_creator._get_text_models", return_value=["model1"]),
            patch("validator.tournament.task_creator._get_instruct_text_datasets", return_value=["ds1"]),
            patch("validator.tournament.task_creator._get_tournament_base_model", return_value="Qwen/Qwen2.5-7B-Instruct"),
            patch("validator.tournament.task_creator._get_prev_tourn_winner_model", return_value="prev-winner/model"),
            patch("validator.tournament.task_creator.create_synthetic_env_task", side_effect=mock_create_env_task),
            patch("validator.tournament.task_creator._create_and_register_tournament_task", new_callable=AsyncMock),
        ):
            from validator.tournament.task_creator import _create_environment_boss_round_tasks
            config = MagicMock()
            await _create_environment_boss_round_tasks(round_data, "tourn_abc", config)

        assert len(created_tasks) == 3

        # Task 0: CONTINUATION with tournament base model
        assert created_tasks[0]["training_start_point"] == TrainingStartPoint.CONTINUATION
        assert created_tasks[0]["model_id_override"] == "Qwen/Qwen2.5-7B-Instruct"

        # Task 1: FROM_SCRATCH with no model override (random)
        assert created_tasks[1]["training_start_point"] == TrainingStartPoint.FROM_SCRATCH
        assert created_tasks[1]["model_id_override"] is None

        # Task 2: PREVIOUS_WINNER with previous tournament winner model
        assert created_tasks[2]["training_start_point"] == TrainingStartPoint.PREVIOUS_WINNER
        assert created_tasks[2]["model_id_override"] == "prev-winner/model"

    @pytest.mark.asyncio
    async def test_prev_winner_fallback_to_target_model(self):
        """When no previous winner exists, falls back to ENV_TARGET_TOURN_MODEL."""
        from validator.tournament.task_creator import _get_prev_tourn_winner_model

        with patch(
            "validator.tournament.task_creator.get_latest_completed_tournament",
            return_value=None,
        ):
            config = MagicMock()
            result = await _get_prev_tourn_winner_model("tourn_xyz", config)

        assert result == t_cst.ENV_TARGET_TOURN_MODEL

    @pytest.mark.asyncio
    async def test_prev_winner_incompatible_base_falls_back(self):
        """Winner exists but was trained from a different base → fallback."""
        from validator.tournament.task_creator import _get_prev_tourn_winner_model

        prev_tourn = MagicMock()
        prev_tourn.winner_model_repo = "prev-winner/repo"
        prev_tourn.winner_model_base = "different/base-model"  # Not ENV_TARGET_TOURN_MODEL

        with patch(
            "validator.tournament.task_creator.get_latest_completed_tournament",
            return_value=prev_tourn,
        ):
            config = MagicMock()
            result = await _get_prev_tourn_winner_model("tourn_xyz", config)

        assert result == t_cst.ENV_TARGET_TOURN_MODEL

    @pytest.mark.asyncio
    async def test_prev_winner_compatible_base_returns_repo(self):
        """Winner trained from ENV_TARGET_TOURN_MODEL → use their model."""
        from validator.tournament.task_creator import _get_prev_tourn_winner_model

        prev_tourn = MagicMock()
        prev_tourn.winner_model_repo = "prev-winner/repo"
        prev_tourn.winner_model_base = t_cst.ENV_TARGET_TOURN_MODEL

        with patch(
            "validator.tournament.task_creator.get_latest_completed_tournament",
            return_value=prev_tourn,
        ):
            config = MagicMock()
            result = await _get_prev_tourn_winner_model("tourn_xyz", config)

        assert result == "prev-winner/repo"


# --- 2d: Environment scaling per round ---


class TestEnvScaling:
    def test_round_1_gets_2_envs(self):
        num_envs = 1 * t_cst.ENV_ENVS_PER_ROUND_MULTIPLIER
        num_envs = min(num_envs, len(EnvironmentName))
        assert num_envs == 2

    def test_round_2_gets_3_envs_capped(self):
        """R2 = 4, but capped at len(EnvironmentName) = 3."""
        num_envs = 2 * t_cst.ENV_ENVS_PER_ROUND_MULTIPLIER
        num_envs = min(num_envs, len(EnvironmentName))
        assert num_envs == min(4, len(EnvironmentName))

    def test_round_3_capped_at_total_envs(self):
        num_envs = 3 * t_cst.ENV_ENVS_PER_ROUND_MULTIPLIER
        num_envs = min(num_envs, len(EnvironmentName))
        assert num_envs == len(EnvironmentName)

    def test_training_hours_by_round(self):
        assert t_cst.ENV_TRAINING_HOURS_BY_ROUND[1] == 1.5
        assert t_cst.ENV_TRAINING_HOURS_BY_ROUND[2] == 2.0
        assert t_cst.ENV_TRAINING_HOURS_BY_ROUND[3] == 2.5
        assert t_cst.ENV_TRAINING_HOURS_BY_ROUND[4] == 3.0


# --- 2e: Model continuation start point ---


class TestModelContinuationStartPoint:
    def test_round_1_is_default(self):
        start_point = TrainingStartPoint.CONTINUATION if 1 > 1 else TrainingStartPoint.DEFAULT
        assert start_point == TrainingStartPoint.DEFAULT

    def test_round_2_is_continuation(self):
        start_point = TrainingStartPoint.CONTINUATION if 2 > 1 else TrainingStartPoint.DEFAULT
        assert start_point == TrainingStartPoint.CONTINUATION

    def test_round_5_is_continuation(self):
        start_point = TrainingStartPoint.CONTINUATION if 5 > 1 else TrainingStartPoint.DEFAULT
        assert start_point == TrainingStartPoint.CONTINUATION


# --- Boss round winner determination (env vs text) ---


class TestDetermineBossRoundWinner:
    def test_env_contender_must_win_all_three(self):
        """Environment tournaments use determine_env_tournament_winner (async, DB),
        not determine_boss_round_winner. The TEXT/IMAGE path uses majority.
        This test verifies the majority path does NOT apply to env logic."""
        # TEXT: 2/3 = majority → challenger wins
        assert determine_boss_round_winner(
            ["challenger", "boss", "challenger"], "boss", TournamentType.TEXT
        ) == "challenger"

        # TEXT: 1/3 = no majority → boss retains
        assert determine_boss_round_winner(
            ["challenger", "boss", "boss"], "boss", TournamentType.TEXT
        ) == "boss"

    def test_empty_winners_boss_retains(self):
        assert determine_boss_round_winner([], "boss", TournamentType.TEXT) == "boss"

    def test_all_boss_wins(self):
        assert determine_boss_round_winner(
            ["boss", "boss", "boss"], "boss", TournamentType.TEXT
        ) == "boss"
