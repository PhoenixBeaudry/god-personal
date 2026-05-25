"""Environment tournament group and boss-round winner helpers."""

import numpy as np

from core.logging import get_logger
from core.models.tournament_models import TournamentData
from core.models.tournament_models import TournamentRoundData
from core.models.tournament_models import TournamentTask
from validator.db.database import PSQLDB
from validator.db.sql.tournaments import get_tournament_group_members
from validator.db.sql.tournaments import get_tournament_rounds
from validator.db.sql.tournaments import get_tournament_tasks
from validator.evaluation.ranking import calculate_miner_ranking_and_scores
from validator.shared.config import Config
from validator.shared.constants import EMISSION_BURN_HOTKEY
from validator.tournament import constants as t_cst
from validator.tournament.task_results import get_scores_for_task
from validator.tournament.task_results import get_task_results_for_ranking


logger = get_logger(__name__)


async def determine_env_tournament_winner(
    tournament: TournamentData, _finalists: list[str], _config: Config, psql_db: PSQLDB,
) -> list[str]:
    """Determine environment winner from boss round only.

    Single contender must beat boss on ALL 3 boss round tasks (no threshold,
    strictly higher score). If not, boss retains.
    """
    boss_hotkey = EMISSION_BURN_HOTKEY

    all_rounds = await get_tournament_rounds(tournament.tournament_id, psql_db)
    if not all_rounds:
        return [boss_hotkey]

    final_round = next((r for r in all_rounds if r.is_final_round), None)
    if not final_round:
        logger.warning("No final round found for environment tournament; boss wins by default")
        return [boss_hotkey]

    final_tasks = await get_tournament_tasks(final_round.round_id, psql_db)
    if not final_tasks:
        logger.warning("No boss round tasks found; boss wins by default")
        return [boss_hotkey]

    # Identify the single contender from boss round scores
    contender: str | None = None
    for task in final_tasks:
        scores = await get_scores_for_task(task.task_id, psql_db)
        for hotkey in scores:
            if hotkey != boss_hotkey:
                contender = hotkey
                break
        if contender:
            break

    if not contender:
        logger.info("No contender found in boss round; boss wins by default")
        return [boss_hotkey]

    # Contender must beat boss on ALL boss round tasks
    for task in final_tasks:
        scores = await get_scores_for_task(task.task_id, psql_db)
        contender_score = scores.get(contender)
        boss_score = scores.get(boss_hotkey)

        if contender_score is None:
            logger.info(f"Contender {contender} has no score on task {task.task_id}; boss retains")
            return [boss_hotkey, contender]

        if boss_score is not None and contender_score <= boss_score:
            logger.info(
                f"Boss retains: contender {contender} scored {contender_score:.2f} vs boss {boss_score:.2f} "
                f"on task {task.task_id}"
            )
            return [boss_hotkey, contender]

    logger.info(f"Contender {contender} wins environment tournament: beat boss on all {len(final_tasks)} boss round tasks")
    return [contender, boss_hotkey]


async def get_environment_group_winners(
    completed_round: TournamentRoundData, round_tasks: list[TournamentTask], psql_db: PSQLDB, config: Config
) -> list[str]:
    """Get winners from environment tournament group rounds.

    For the final round, return all finalists (boss + contender) and defer
    champion decision to determine_env_tournament_winner().
    """
    boss_hotkey = EMISSION_BURN_HOTKEY

    if completed_round.is_final_round:
        if not round_tasks:
            return [boss_hotkey]
        group_id = round_tasks[0].group_id
        if not group_id:
            return [boss_hotkey]
        participants = await get_tournament_group_members(group_id, psql_db)
        participant_hotkeys = [p.hotkey for p in participants]
        if boss_hotkey not in participant_hotkeys:
            participant_hotkeys.append(boss_hotkey)
        return participant_hotkeys

    if not round_tasks:
        logger.warning(f"No tasks found for environment round {completed_round.round_id}")
        return []

    single_group = len(round_tasks) == 1
    all_winners: list[str] = []

    for task in round_tasks:
        group_id = task.group_id
        if not group_id:
            logger.warning(f"No group_id on task {task.task_id}, skipping")
            continue

        participants = await get_tournament_group_members(group_id, psql_db)
        participant_hotkeys = [p.hotkey for p in participants]
        if not participant_hotkeys:
            logger.warning(f"Environment group {group_id} has no participants")
            continue

        miner_results = await get_task_results_for_ranking(task.task_id, psql_db)
        if not miner_results:
            logger.warning(f"No valid results for task {task.task_id}")
            continue

        ranked_results = calculate_miner_ranking_and_scores(miner_results)
        participant_scores: dict[str, float] = {}
        for result in ranked_results:
            if result.adjusted_loss is None or np.isnan(result.adjusted_loss):
                continue
            participant_scores[result.hotkey] = result.adjusted_loss

        if not participant_scores:
            logger.warning(f"Group {group_id} has no valid scores")
            continue

        sorted_participants = sorted(participant_scores.items(), key=lambda x: x[1], reverse=True)
        boss_score = participant_scores.get(boss_hotkey)
        non_boss_sorted = [(hotkey, score) for hotkey, score in sorted_participants if hotkey != boss_hotkey]

        # Boss retains only when down to a single group and boss wins/ties
        if single_group and boss_score is not None and non_boss_sorted:
            top_challenger_score = non_boss_sorted[0][1]
            if boss_score >= top_challenger_score:
                logger.info(
                    f"Environment group {group_id}: boss score {boss_score} >= top challenger {top_challenger_score} "
                    f"— single group, boss retains"
                )
                continue

        # Advance up to ENV_ADVANCE_PER_GROUP but always eliminate at least 1 to guarantee convergence
        top_to_advance = max(1, min(t_cst.ENV_ADVANCE_PER_GROUP, len(non_boss_sorted) - 1))
        if top_to_advance > 0 and len(non_boss_sorted) > top_to_advance:
            cutoff_score = non_boss_sorted[top_to_advance - 1][1]
            group_winners = [h for h, s in non_boss_sorted if s >= cutoff_score]
        else:
            group_winners = [h for h, _ in non_boss_sorted[:top_to_advance]]

        logger.info(f"Environment group {group_id}: advancing {len(group_winners)} winners: {group_winners}")
        all_winners.extend(group_winners)

    logger.info(f"Environment round {completed_round.round_number}: advancing {len(all_winners)} total non-boss winners")
    return all_winners
