"""PvP game runner: plays head-to-head games and tallies results.

Drives OpenSpiel's evaluate_bots with two LLMBots, one per model.
Each seed is played twice with swapped positions for fairness.
"""

import functools
import logging
import random

import numpy as np
import pyspiel
from open_spiel.python.algorithms import evaluate_bots

from core.constants import EnvironmentName, ENVIRONMENT_CONFIGS
from validator.core import constants as vcst
from core.models.pvp_models import (
    ChatCompletionConfig,
    GameInstance,
    GameOutcome,
    GameScoringContext,
    PvPEnvironmentResult,
    PvPMatchupConfig,
)
from validator.evaluation.pvp.agents import (
    BaseGameAgent,
    GinRummyAgent,
    LeducPokerAgent,
    LiarsDiceAgent,
)
from validator.evaluation.pvp.bot import LLMBot
from validator.evaluation.pvp.scoring import determine_outcome

logger = logging.getLogger(__name__)

_AGENT_REGISTRY: dict[EnvironmentName, type[BaseGameAgent]] = {
    EnvironmentName.LIARS_DICE: LiarsDiceAgent,
    EnvironmentName.LEDUC_POKER: LeducPokerAgent,
    EnvironmentName.GIN_RUMMY: GinRummyAgent,
}


def run_matchup(
    env_name: EnvironmentName,
    matchup_config: PvPMatchupConfig,
    config_a: ChatCompletionConfig,
    config_b: ChatCompletionConfig,
    base_seed: int,
) -> PvPEnvironmentResult:
    """Run a full PvP matchup for one environment.

    Plays matchup_config.num_games seeds, each twice (swapped positions).
    """
    agent = _AGENT_REGISTRY[env_name]()
    instances = _build_instances(env_name, agent, matchup_config.num_games, base_seed)
    return _execute_matchup(env_name, instances, config_a, config_b, agent)


def _build_instances(
    env_name: EnvironmentName,
    agent: BaseGameAgent,
    num_games: int,
    base_seed: int,
) -> list[GameInstance]:
    """Generate paired GameInstances (original + position-swapped) for each seed."""
    env_config = ENVIRONMENT_CONFIGS[env_name]
    seed_rng = random.Random(base_seed)
    instances: list[GameInstance] = []

    for _ in range(num_games):
        seed = seed_rng.randint(1, vcst.PVP_SEED_RANGE_MAX)
        task_rng = random.Random(seed)
        task_id = task_rng.randint(env_config.task_id_min + 1, env_config.task_id_max)
        config_id = task_id % vcst.PVP_CONFIG_ID_DIVISOR
        game_params = agent.generate_params(config_id)

        game = pyspiel.load_game(agent.game_name, game_params)
        game_type = game.get_type()

        base = GameInstance(
            game_name=agent.game_name,
            game_params=game_params,
            model_a_player_id=0,
            seed=seed,
            is_zero_sum=game_type.utility == pyspiel.GameType.Utility.ZERO_SUM,
            min_utility=game.min_utility(),
            max_utility=game.max_utility(),
        )
        swapped = base.model_copy(update={"model_a_player_id": 1, "seed": seed + vcst.PVP_SEED_OFFSET_SWAP})

        instances.append(base)
        instances.append(swapped)

    return instances


def _execute_matchup(
    env_name: EnvironmentName,
    instances: list[GameInstance],
    config_a: ChatCompletionConfig,
    config_b: ChatCompletionConfig,
    agent: BaseGameAgent,
) -> PvPEnvironmentResult:
    """Play all game instances and tally results."""
    play = functools.partial(_play_game, config_a=config_a, config_b=config_b, agent=agent)

    result = PvPEnvironmentResult()
    for i, instance in enumerate(instances):
        outcome = play(instance)
        _tally(result, outcome)

        if (i + 1) % vcst.PVP_LOG_INTERVAL_GAMES == 0:
            logger.info(
                "%s: %d/%d games, a=%d b=%d draws=%d",
                env_name.value, i + 1, len(instances),
                result.model_a_wins, result.model_b_wins, result.draws,
            )

    logger.info(
        "%s complete: %d games, a=%d b=%d draws=%d",
        env_name.value, result.total_games,
        result.model_a_wins, result.model_b_wins, result.draws,
    )
    return result


def _play_game(
    instance: GameInstance,
    config_a: ChatCompletionConfig,
    config_b: ChatCompletionConfig,
    agent: BaseGameAgent,
) -> GameOutcome:
    """Play a single game and return outcome from model_a's perspective."""
    game = pyspiel.load_game(instance.game_name, instance.game_params)
    model_b_player_id = 1 - instance.model_a_player_id

    bot_a = LLMBot(
        game=game,
        player_id=instance.model_a_player_id,
        config=config_a,
        agent=agent,
        rng_seed=instance.seed,
    )
    bot_b = LLMBot(
        game=game,
        player_id=model_b_player_id,
        config=config_b,
        agent=agent,
        rng_seed=instance.seed + 1,
    )

    bots = [None, None]
    bots[instance.model_a_player_id] = bot_a
    bots[model_b_player_id] = bot_b

    state = game.new_initial_state()
    returns = evaluate_bots.evaluate_bots(state, bots, np.random.RandomState(instance.seed))

    scoring = GameScoringContext(
        returns=list(returns),
        player_id=instance.model_a_player_id,
        is_zero_sum=instance.is_zero_sum,
        min_utility=instance.min_utility,
        max_utility=instance.max_utility,
    )
    return determine_outcome(scoring)


def _tally(result: PvPEnvironmentResult, outcome: GameOutcome) -> None:
    result.total_games += 1
    if outcome == GameOutcome.WIN:
        result.model_a_wins += 1
    elif outcome == GameOutcome.LOSS:
        result.model_b_wins += 1
    else:
        result.draws += 1
