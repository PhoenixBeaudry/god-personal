"""Round-level tournament winner selection helpers."""

from collections import Counter

import numpy as np

from core.logging import get_logger
from core.models.tournament_models import RoundType
from core.models.tournament_models import TournamentRoundData
from core.models.tournament_models import TournamentTask
from core.models.tournament_models import TournamentType
from core.models.utility_models import TrainingStatus
from core.models.utility_models import is_environment_task
from core.models.utility_models import scores_higher_is_better
from validator.db.database import PSQLDB
from validator.db.sql.submissions_and_scoring import get_task_winner
from validator.db.sql.tasks import get_task
from validator.db.sql.tournaments import count_champion_consecutive_wins
from validator.db.sql.tournaments import get_tournament
from validator.db.sql.tournaments import get_tournament_group_members
from validator.db.sql.tournaments import get_tournament_tasks
from validator.db.sql.tournaments import get_training_status_for_task_and_hotkeys
from validator.evaluation.ranking import calculate_miner_ranking_and_scores
from validator.shared.config import Config
from validator.shared.constants import EMISSION_BURN_HOTKEY
from validator.tournament.environment_results import get_environment_group_winners
from validator.tournament.task_results import get_task_results_for_ranking
from validator.tournament.thresholds import get_progressive_threshold
from validator.tournament.thresholds import update_threshold_adjusted_quality_scores_for_task


logger = get_logger(__name__)


def determine_boss_round_winner(task_winners: list[str], boss_hotkey: str, tournament_type: TournamentType) -> str:
    """
    Determine the winner of a boss round based on task results and tournament type.

    Args:
        task_winners: List of hotkeys that won each task in the boss round
        boss_hotkey: The defending champion's hotkey
        tournament_type: Type of tournament (TEXT or IMAGE)

    Returns:
        Hotkey of the boss round winner
    """
    if not task_winners:
        logger.error("No valid task winners found in boss round - all tasks failed to determine winners")
        logger.info(f"Defaulting to boss as winner due to evaluation failures: {boss_hotkey}")
        return boss_hotkey

    # Count wins for each contestant
    win_counts = Counter(task_winners)
    total_tasks = len(task_winners)

    # Find the opponent (non-boss hotkey)
    opponent_hotkey = None
    for hotkey in win_counts.keys():
        if hotkey != boss_hotkey:
            opponent_hotkey = hotkey
            break

    opponent_wins = win_counts.get(opponent_hotkey, 0) if opponent_hotkey else 0

    # Apply different winning requirements based on tournament type
    # Both IMAGE and TEXT tournaments: Challenger must win more than half (majority) of tasks to become new boss
    required_wins = (total_tasks // 2) + 1
    if opponent_hotkey and opponent_wins > total_tasks // 2:
        logger.info(
            f"{tournament_type.value} tournament: Challenger wins boss round with majority: "
            f"{opponent_wins}/{total_tasks} tasks won (required {required_wins})"
        )
        return opponent_hotkey
    else:
        boss_wins = win_counts.get(boss_hotkey, 0)
        if opponent_hotkey:
            logger.info(
                f"{tournament_type.value} tournament: Boss retains title - challenger won "
                f"{opponent_wins}/{total_tasks} tasks (requires {required_wins}/{total_tasks} to dethrone), "
                f"boss won {boss_wins}/{total_tasks}"
            )
        else:
            logger.info(f"{tournament_type.value} tournament: Boss retains title by default")
        return boss_hotkey


async def get_knockout_winners(
    completed_round: TournamentRoundData, round_tasks: list[TournamentTask], psql_db: PSQLDB, config: Config
) -> list[str]:
    """Get winners from knockout round."""
    winners = []

    if not completed_round.is_final_round:
        # Use simple quality score comparison for regular knockout rounds
        for task in round_tasks:
            winner = await get_task_winner(task.task_id, psql_db)
            if winner:
                winners.append(winner)
    else:
        # Boss round. Progressive threshold system based on consecutive wins.
        boss_hotkey = EMISSION_BURN_HOTKEY
        opponent_hotkey = None
        task_winners = []

        # Get tournament info to determine the current champion and their consecutive wins
        tournament = await get_tournament(completed_round.tournament_id, psql_db)
        if not tournament:
            logger.error(f"Could not find tournament {completed_round.tournament_id}")
            return []

        # Get the current champion (base_winner_hotkey) and count their consecutive wins
        current_champion = tournament.base_winner_hotkey or boss_hotkey
        consecutive_wins = await count_champion_consecutive_wins(psql_db, tournament.tournament_type, current_champion)

        # Calculate the progressive threshold
        threshold_percentage = get_progressive_threshold(consecutive_wins, tournament.tournament_type)
        logger.info(
            f"Champion {current_champion} has {consecutive_wins} consecutive wins, "
            f"using {threshold_percentage * 100:.1f}% threshold"
        )

        for task in round_tasks:
            logger.info(f"Processing boss round task {task.task_id}")

            task_object = await get_task(task.task_id, psql_db)

            miner_results = await get_task_results_for_ranking(task.task_id, psql_db)
            if not miner_results:
                logger.warning(f"No valid results for boss round task {task.task_id}. Winner is base contestant.")
                task_winners.append(boss_hotkey)
                continue

            ranked_results = calculate_miner_ranking_and_scores(miner_results)

            boss_loss = None
            opponent_loss = None
            opponent_hotkey = None

            for result in ranked_results:
                if result.hotkey == boss_hotkey:
                    boss_loss = result.adjusted_loss
                else:
                    if opponent_hotkey is None:
                        opponent_hotkey = result.hotkey
                        opponent_loss = result.adjusted_loss

            if boss_loss is None or opponent_loss is None:
                logger.warning(f"Boss round task {task.task_id} missing boss or opponent loss")
                # Check training status to determine winner when evaluation results are missing
                training_statuses = await get_training_status_for_task_and_hotkeys(
                    task.task_id, [boss_hotkey, opponent_hotkey], psql_db
                )

                boss_training_success = training_statuses.get(boss_hotkey) == TrainingStatus.SUCCESS
                opponent_training_success = training_statuses.get(opponent_hotkey) == TrainingStatus.SUCCESS

                if opponent_training_success and not boss_training_success:
                    logger.info(f"Boss training failed, opponent succeeded - opponent wins task {task.task_id}")
                    task_winners.append(opponent_hotkey)
                elif boss_training_success and not opponent_training_success:
                    logger.info(f"Opponent training failed, boss succeeded - boss wins task {task.task_id}")
                    task_winners.append(boss_hotkey)
                elif not boss_training_success and not opponent_training_success:
                    logger.info(f"Both training failed - boss wins by default for task {task.task_id}")
                    task_winners.append(boss_hotkey)
                else:
                    # Both training succeeded but at least one has missing/invalid evaluation results
                    # Check who has valid evaluation results and award to them
                    boss_has_valid_eval = boss_loss is not None
                    opponent_has_valid_eval = opponent_loss is not None

                    if opponent_has_valid_eval and not boss_has_valid_eval:
                        logger.info(f"Boss evaluation failed, opponent succeeded - opponent wins task {task.task_id}")
                        task_winners.append(opponent_hotkey)
                    elif boss_has_valid_eval and not opponent_has_valid_eval:
                        logger.info(f"Opponent evaluation failed, boss succeeded - boss wins task {task.task_id}")
                        task_winners.append(boss_hotkey)
                    else:
                        logger.warning(
                            f"Both evaluation failed or both succeeded but missing results - skipping task {task.task_id}"
                        )
                continue

            logger.info(f"Boss round task {task.task_id}: Boss loss: {boss_loss:.6f}, Opponent loss: {opponent_loss:.6f}")

            # Apply progressive threshold system
            boss_multiplier = 1 + threshold_percentage  # For higher-is-better tasks
            boss_divisor = 1 - threshold_percentage  # For lower-is-better tasks

            if scores_higher_is_better(task_object.task_type):
                # For GRPO and environment tasks, higher scores are better
                task_label = "Environment" if is_environment_task(task_object.task_type) else "GRPO"
                if boss_loss * boss_multiplier > opponent_loss:
                    task_winner = boss_hotkey
                    task_winners.append(task_winner)
                    logger.info(
                        f"{task_label} task: Boss wins (higher is better): {boss_loss:.6f} * "
                        f"{boss_multiplier:.3f} = {boss_loss * boss_multiplier:.6f} > {opponent_loss:.6f}"
                    )
                else:
                    task_winner = opponent_hotkey
                    task_winners.append(task_winner)
                    logger.info(
                        f"{task_label} task: Opponent wins (higher is better): "
                        f"{opponent_loss:.6f} >= {boss_loss * boss_multiplier:.6f}"
                    )
            else:
                # For other tasks, lower scores are better
                if boss_loss * boss_divisor < opponent_loss:
                    task_winner = boss_hotkey
                    task_winners.append(task_winner)
                    logger.info(
                        f"{task_object.task_type} task: Boss wins (lower is better): "
                        f"{boss_loss:.6f} * {boss_divisor:.3f} = {boss_loss * boss_divisor:.6f} < {opponent_loss:.6f}"
                    )
                else:
                    task_winner = opponent_hotkey
                    task_winners.append(task_winner)
                    logger.info(
                        f"{task_object.task_type} task: Opponent wins (lower is better): "
                        f"{opponent_loss:.6f} <= {boss_loss * boss_divisor:.6f}"
                    )

            await update_threshold_adjusted_quality_scores_for_task(
                task_id=task.task_id,
                winner_hotkey=task_winner,
                threshold_percentage=threshold_percentage,
                compared_hotkeys=[boss_hotkey, opponent_hotkey],
                psql_db=psql_db,
            )

        boss_round_winner = determine_boss_round_winner(task_winners, boss_hotkey, tournament.tournament_type)

        winners = [boss_round_winner]

    return winners


async def get_group_winners(
    completed_round: TournamentRoundData, round_tasks: list[TournamentTask], psql_db: PSQLDB, config: Config = None
) -> list[str]:
    """Get winners from group round based on adjusted loss scores."""

    # Check if this is an environment task
    is_environment = False
    if round_tasks:
        first_task_object = await get_task(round_tasks[0].task_id, psql_db)
        is_environment = bool(first_task_object and is_environment_task(first_task_object.task_type))

    if is_environment:
        return await get_environment_group_winners(completed_round, round_tasks, psql_db, config)

    # Determine how many winners to advance
    if completed_round.is_final_round:
        TOP_WINNERS_TO_ADVANCE = 1
    else:
        TOP_WINNERS_TO_ADVANCE = 8

    all_winners = []

    for task in round_tasks:
        group_id = task.group_id
        task_id = task.task_id

        logger.info(f"Processing group {group_id} in round {completed_round.round_id}")

        participants = await get_tournament_group_members(group_id, psql_db)
        participant_hotkeys = [p.hotkey for p in participants]
        logger.info(f"Group {group_id} and task {task_id} have {len(participant_hotkeys)} participants")

        if not participant_hotkeys:
            logger.warning(f"Group {group_id} has no participants")
            continue

        miner_results = await get_task_results_for_ranking(task_id, psql_db)
        if not miner_results:
            logger.warning(f"No valid results for task {task_id}")
            continue

        ranked_results = calculate_miner_ranking_and_scores(miner_results)

        participant_scores = {}
        for result in ranked_results:
            hotkey = result.hotkey
            adjusted_loss = result.adjusted_loss

            if adjusted_loss is None or np.isnan(adjusted_loss):
                continue

            participant_scores[hotkey] = adjusted_loss

        if not participant_scores:
            logger.warning(f"Group {group_id} has no valid scores - proceeding with no winners")
            continue

        task_object = await get_task(task_id, psql_db)
        higher_is_better = bool(task_object and scores_higher_is_better(task_object.task_type))

        sorted_participants = sorted(participant_scores.items(), key=lambda x: x[1], reverse=higher_is_better)
        ranking_direction = "descending (higher is better)" if higher_is_better else "ascending (lower is better)"

        logger.info(
            f"Group {group_id} participants sorted by adjusted loss ({ranking_direction}): "
            f"{[(hotkey, f'{loss:.6f}') for hotkey, loss in sorted_participants]}"
        )

        num_to_advance = min(TOP_WINNERS_TO_ADVANCE, len(sorted_participants))
        group_winners = [hotkey for hotkey, _ in sorted_participants[:num_to_advance]]

        logger.info(f"Group {group_id}: Advancing top {num_to_advance} by adjusted loss: {group_winners}")
        all_winners.extend(group_winners)

    return all_winners


async def get_round_winners(completed_round: TournamentRoundData, psql_db: PSQLDB, config: Config) -> list[str]:
    """Get winners from the completed round."""
    round_tasks = await get_tournament_tasks(completed_round.round_id, psql_db)

    if completed_round.round_type == RoundType.KNOCKOUT:
        winners = await get_knockout_winners(completed_round, round_tasks, psql_db, config)
    else:
        winners = await get_group_winners(completed_round, round_tasks, psql_db, config)

    unique_winners = list(dict.fromkeys(winners))
    if len(winners) != len(unique_winners):
        logger.info(f"Removed {len(winners) - len(unique_winners)} duplicate winners from round {completed_round.round_id}")
        logger.info(f"Original winners: {winners}")
        logger.info(f"Unique winners: {unique_winners}")

    return unique_winners
