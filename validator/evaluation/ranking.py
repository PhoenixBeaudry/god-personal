import math

import numpy as np

import validator.shared.constants as cts
from core.logging import LogContext
from core.logging import get_logger
from core.models.utility_models import TaskType
from validator.shared.models import MinerResultsImage
from validator.shared.models import MinerResultsText


logger = get_logger(__name__)


def calculate_miner_ranking_and_scores(
    miner_results: list[MinerResultsText | MinerResultsImage],
) -> list[MinerResultsText | MinerResultsImage]:
    logger.info("Beginning score calculation...")

    valid_results = []
    # Initialize all scores to 0.0 and set appropriate reasons
    for result in miner_results:
        with LogContext(miner_hotkey=result.hotkey):
            result.score = 0.0
            # atp, we only set score_reason in these cases (all are invalid and is_finetune == False):
            # "Invalid/No repo submitted", "Evaluation failed", "Duplicated submission"
            if result.score_reason:
                continue
            elif not result.is_finetune:
                result.score_reason = "Non-finetuned submission"
                logger.info(f"Miner {result.hotkey}: Non-finetuned, score initialized to 0.0")
            elif np.isnan(result.test_loss):
                result.score_reason = "Invalid test loss"
                logger.info(f"Miner {result.hotkey}: Invalid test loss, score initialized to 0.0")
            else:
                valid_results.append(result)

    if not valid_results:
        logger.warning("No valid finetuned submissions found. All scores set to 0.0")
        return miner_results

    is_grpo_task = False
    if valid_results and isinstance(valid_results[0], MinerResultsText):
        is_grpo_task = valid_results[0].task_type == TaskType.GRPOTASK
        if is_grpo_task:
            logger.info("Processing GRPO task - higher loss is better")
        else:
            logger.info(f"Processing {valid_results[0].task_type} - using test_loss for ranking")

    is_env_task = False
    if valid_results and isinstance(valid_results[0], MinerResultsText):
        is_env_task = valid_results[0].task_type == TaskType.ENVIRONMENTTASK
        if is_env_task:
            logger.info("Processing Env task - higher score is better")
        else:
            logger.info(f"Processing {valid_results[0].task_type} - using test_loss for ranking")

    logger.info("Using test loss for ranking")
    ranked_results = []
    for result in valid_results:
        result.adjusted_loss = result.test_loss
        ranked_results.append((result, result.test_loss))
        logger.info(f"Miner {result.hotkey}: test_loss {result.test_loss:.6f}")

    if is_grpo_task:
        # For GRPO, sort in reverse order (higher value is better)
        ranked_results.sort(key=lambda x: float("-inf") if math.isnan(x[1]) else -x[1])
        ranking_type = "GRPO score (bigger is better)"
    elif is_env_task:
        # For Env taks, sort in reverse order (higher value is better)
        ranked_results.sort(key=lambda x: float("-inf") if math.isnan(x[1]) else -x[1])
        ranking_type = "Environment score (bigger is better)"
    else:
        # For other tasks, sort normally (lower loss is better)
        ranked_results.sort(key=lambda x: float("inf") if math.isnan(x[1]) else x[1])
        ranking_type = "test_loss"

    if ranked_results:
        top_result, top_metric = ranked_results[0]
        with LogContext(miner_hotkey=top_result.hotkey):
            top_result.score = cts.FIRST_PLACE_SCORE
            top_result.score_reason = f"Ranked 1st by {ranking_type}"
            logger.info(
                f"Miner {top_result.hotkey} (finetuned):"
                f" test_loss={top_result.test_loss:.4f}"
                f" {ranking_type}={top_metric:.4f}"
                f" score={top_result.score:.4f}"
                f" score_reason={top_result.score_reason}"
            )

    total_valid_miners = len(valid_results)
    if total_valid_miners > cts.MIN_IDEAL_NUM_MINERS_IN_POOL:
        penalty_count = max(1, int(total_valid_miners * 0.25))
        penalty_start_idx = total_valid_miners - penalty_count

        for result, metric in ranked_results[1:penalty_start_idx]:
            with LogContext(miner_hotkey=result.hotkey):
                result.score_reason = f"Ranked below top 1 by {ranking_type}"
                logger.info(
                    f"Miner {result.hotkey} (finetuned):"
                    f" test_loss={result.test_loss:.4f}"
                    f" {ranking_type}={metric:.4f}"
                    f" score=0.0"
                    f" score_reason={result.score_reason}"
                )

        for result, metric in ranked_results[penalty_start_idx:]:
            with LogContext(miner_hotkey=result.hotkey):
                result.score = cts.SCORE_PENALTY
                result.score_reason = f"Bottom 25% ranked by {ranking_type}"
                logger.info(
                    f"Miner {result.hotkey} (finetuned):"
                    f" test_loss={result.test_loss:.4f}"
                    f" {ranking_type}={metric:.4f}"
                    f" score={result.score:.4f}"
                    f" score_reason={result.score_reason}"
                )
    else:
        for result, metric in ranked_results[1:]:
            with LogContext(miner_hotkey=result.hotkey):
                result.score_reason = f"Ranked below top 1 by {ranking_type}"
                logger.info(
                    f"Miner {result.hotkey} (finetuned):"
                    f" test_loss={result.test_loss:.4f}"
                    f" {ranking_type}={metric:.4f}"
                    f" score=0.0"
                    f" score_reason={result.score_reason}"
                )

    # Apply penalty scores to failed submissions when valid submissions exist
    if valid_results:
        for result in miner_results:
            # Find failed submissions that haven't been scored yet
            if (not result.is_finetune or np.isnan(result.test_loss)) and result.score == 0.0:
                result.score = cts.SCORE_PENALTY
                logger.info(
                    f"Miner {result.hotkey}: Failed submission ({result.score_reason}), "
                    f"applying penalty score {cts.SCORE_PENALTY}"
                )

    return miner_results
