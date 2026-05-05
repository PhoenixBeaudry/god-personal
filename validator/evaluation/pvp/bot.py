"""LLM Bot implementation for OpenSpiel PvP evaluation.

Wraps an LLM inference endpoint as a pyspiel.Bot, maintaining
conversation history and parsing actions from model responses.
"""

import asyncio
import concurrent.futures
import logging
import re

import numpy as np
import pyspiel

from core.models.pvp_models import ChatCompletionConfig, ChatMessage, ChatResult
from validator.core import constants as vcst
from validator.evaluation.pvp.agents import BaseGameAgent
from validator.evaluation.pvp.chat import chat_completion

logger = logging.getLogger(__name__)


class LLMBot(pyspiel.Bot):
    """OpenSpiel Bot backed by an LLM via OpenAI-compatible API.

    Maintains full conversation history per game. On each step(),
    generates a user prompt from the game state, calls the LLM,
    and parses an action ID from the response.
    """

    def __init__(
        self,
        game: pyspiel.Game,
        player_id: int,
        config: ChatCompletionConfig,
        agent: BaseGameAgent,
        rng_seed: int,
        executor: concurrent.futures.ThreadPoolExecutor,
    ):
        pyspiel.Bot.__init__(self)
        self._game = game
        self._player_id = player_id
        self._config = config
        self._agent = agent
        self._rng = np.random.RandomState(rng_seed)
        self._executor = executor
        self._conversation: list[ChatMessage] = []
        self._system_prompt_set = False

    def restart_at(self, state: pyspiel.State) -> None:
        self._conversation.clear()
        self._system_prompt_set = False

    def inform_action(self, state: pyspiel.State, player_id: int, action: int) -> None:
        pass

    def step(self, state: pyspiel.State) -> int:
        """Choose an action by querying the LLM.

        Called by evaluate_bots during game play.
        """
        if not self._system_prompt_set:
            system_prompt = self._agent.generate_system_prompt()
            self._conversation.append(ChatMessage(role="system", content=system_prompt))
            self._system_prompt_set = True

        legal_actions = state.legal_actions(self._player_id)
        user_prompt = self._agent.generate_user_prompt(state, self._player_id, legal_actions)
        self._conversation.append(ChatMessage(role="user", content=user_prompt))

        for attempt in range(vcst.PVP_BOT_MAX_PARSING_RETRIES + 1):
            result = self._call_llm()

            if result.content is None:
                self._conversation.append(
                    ChatMessage(role="assistant", content="")
                )
                continue

            self._conversation.append(
                ChatMessage(role="assistant", content=result.content)
            )

            parsed_action = _parse_action(result.content, legal_actions)
            if parsed_action is not None:
                return parsed_action

            retry_msg = (
                f"Invalid response. Respond with ONLY the action ID number. "
                f"Attempt {attempt + 1}/{vcst.PVP_BOT_MAX_PARSING_RETRIES + 1}."
            )
            self._conversation.append(ChatMessage(role="user", content=retry_msg))

        # All retries exhausted — fall back to random legal action
        logger.warning(
            "LLM failed to produce valid action after %d attempts, using random fallback",
            vcst.PVP_BOT_MAX_PARSING_RETRIES + 1,
        )
        return int(self._rng.choice(legal_actions))

    def _call_llm(self) -> ChatResult:
        """Call the LLM synchronously via thread pool (evaluate_bots runs in a thread)."""

        async def _run() -> ChatResult:
            return await chat_completion(self._config, self._conversation)

        def _run_in_loop() -> ChatResult:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(_run())
            finally:
                loop.close()

        future = self._executor.submit(_run_in_loop)
        return future.result()


def _parse_action(response: str, legal_actions: list[int]) -> int | None:
    """Parse an action ID from LLM response text.

    Strategies (in priority order):
    1. Response is purely a number
    2. Find a legal action ID mentioned in the text
    """
    cleaned = response.strip()

    # Strategy 1: pure number
    match = re.match(r"^\s*(\d+)\s*$", cleaned)
    if match:
        action = int(match.group(1))
        if action in legal_actions:
            return action

    # Strategy 2: first legal action ID found in text
    for action in legal_actions:
        if re.search(rf"\b{action}\b", cleaned):
            return action

    return None
