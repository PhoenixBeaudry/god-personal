"""Tests for LLMBot: action parsing, retry logic, conversation management.

Uses a scripted ChatFn to test the full step() path without any HTTP calls.
"""

import pytest

from core.models.pvp_models import (
    ChatCompletionConfig,
    ChatMessage,
    ChatResult,
    ChatRole,
)
from validator.evaluation.pvp.bot import LLMBot, _parse_action
from validator.evaluation.pvp.chat import strip_think_tags


# --- Scripted ChatFn for deterministic testing ---


def _make_scripted_chat_fn(responses: list[str]):
    """Return a ChatFn that yields responses in order, cycling if exhausted."""
    call_count = 0

    def chat_fn(config: ChatCompletionConfig, messages: list[ChatMessage]) -> ChatResult:
        nonlocal call_count
        idx = min(call_count, len(responses) - 1)
        call_count += 1
        return ChatResult(content=responses[idx])

    return chat_fn


def _make_config() -> ChatCompletionConfig:
    return ChatCompletionConfig(
        inference_model="test-model",
        base_url="http://localhost:30000/v1",
    )


# --- _parse_action tests ---


class TestParseAction:

    def test_pure_number(self) -> None:
        assert _parse_action("5", [3, 5, 7]) == 5

    def test_pure_number_not_legal(self) -> None:
        assert _parse_action("99", [3, 5, 7]) is None

    def test_last_legal_wins(self) -> None:
        """When multiple legal numbers appear, prefer the last one."""
        assert _parse_action("considering 3, I pick 7", [3, 7, 13]) == 7

    def test_no_legal_match(self) -> None:
        assert _parse_action("I fold", [3, 5, 7]) is None

    def test_empty_string(self) -> None:
        assert _parse_action("", [3, 5, 7]) is None

    def test_whitespace_around_number(self) -> None:
        assert _parse_action("  5  ", [3, 5, 7]) == 5

    def test_word_boundary_prevents_substring(self) -> None:
        """'13' should not match '3' via substring."""
        assert _parse_action("13", [3, 13]) == 13

    def test_single_legal_action(self) -> None:
        assert _parse_action("42", [42]) == 42


# --- strip_think_tags tests ---


class TestStripThinkTags:

    def test_complete_block(self) -> None:
        assert strip_think_tags("<think>reasoning</think>5") == "5"

    def test_thinking_variant(self) -> None:
        assert strip_think_tags("<thinking>stuff</thinking>7") == "7"

    def test_unclosed_tag(self) -> None:
        assert strip_think_tags("<think>still thinking... 5") == ""

    def test_no_tags(self) -> None:
        assert strip_think_tags("5") == "5"

    def test_only_closing_tag(self) -> None:
        assert strip_think_tags("garbage</think>5") == "5"

    def test_empty_after_strip(self) -> None:
        assert strip_think_tags("<think>only thinking</think>") == ""


# --- LLMBot.step() tests (require pyspiel) ---

pytest.importorskip("pyspiel")

import numpy as np
import pyspiel


def _make_bot(chat_fn, player_id: int = 0) -> LLMBot:
    game = pyspiel.load_game("leduc_poker", {"players": 2})
    from validator.evaluation.pvp.agents import LeducPokerAgent
    return LLMBot(
        game=game,
        player_id=player_id,
        chat_fn=chat_fn,
        config=_make_config(),
        agent=LeducPokerAgent(),
        rng_seed=42,
    )


def _get_state_with_legal_actions(player_id: int = 0):
    """Advance a leduc poker game to a point where player_id has legal actions."""
    game = pyspiel.load_game("leduc_poker", {"players": 2})
    state = game.new_initial_state()
    # Deal chance nodes until a player can act
    while state.is_chance_node():
        outcomes = state.chance_outcomes()
        action_list, prob_list = zip(*outcomes)
        state.apply_action(action_list[0])
    return state


class TestBotStep:

    def test_valid_action_first_try(self) -> None:
        state = _get_state_with_legal_actions()
        legal = state.legal_actions(0)
        chat_fn = _make_scripted_chat_fn([str(legal[0])])
        bot = _make_bot(chat_fn, player_id=0)
        action = bot.step(state)
        assert action == legal[0]

    def test_invalid_then_valid(self) -> None:
        state = _get_state_with_legal_actions()
        legal = state.legal_actions(0)
        chat_fn = _make_scripted_chat_fn(["nonsense", str(legal[0])])
        bot = _make_bot(chat_fn, player_id=0)
        action = bot.step(state)
        assert action == legal[0]

    def test_all_retries_exhausted_uses_random_fallback(self) -> None:
        state = _get_state_with_legal_actions()
        legal = state.legal_actions(0)
        chat_fn = _make_scripted_chat_fn(["bad"] * 10)
        bot = _make_bot(chat_fn, player_id=0)
        action = bot.step(state)
        assert action in legal

    def test_none_response_retries(self) -> None:
        state = _get_state_with_legal_actions()
        legal = state.legal_actions(0)
        # First call returns None, second returns valid
        call_count = 0

        def chat_fn(config, messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ChatResult(content=None)
            return ChatResult(content=str(legal[0]))

        bot = _make_bot(chat_fn, player_id=0)
        action = bot.step(state)
        assert action == legal[0]

    def test_restart_at_clears_conversation(self) -> None:
        state = _get_state_with_legal_actions()
        legal = state.legal_actions(0)
        chat_fn = _make_scripted_chat_fn([str(legal[0])])
        bot = _make_bot(chat_fn, player_id=0)
        bot.step(state)
        assert len(bot._conversation) > 0
        bot.restart_at(state)
        assert len(bot._conversation) == 0
        assert bot._system_prompt_set is False

    def test_conversation_alternates_roles(self) -> None:
        """After a successful step, conversation should alternate user/assistant correctly."""
        state = _get_state_with_legal_actions()
        legal = state.legal_actions(0)
        chat_fn = _make_scripted_chat_fn([str(legal[0])])
        bot = _make_bot(chat_fn, player_id=0)
        bot.step(state)

        roles = [msg.role for msg in bot._conversation]
        # Should be: SYSTEM, USER, ASSISTANT
        assert roles[0] == ChatRole.SYSTEM
        for i in range(1, len(roles) - 1, 2):
            assert roles[i] == ChatRole.USER
            assert roles[i + 1] == ChatRole.ASSISTANT
