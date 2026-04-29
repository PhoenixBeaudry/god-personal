"""
Per-type stats tests using real HF datasets with standardized column names.
Each test loads real data, maps to the column names the prep pipeline produces,
and verifies the correct stats type is returned with meaningful values.
Run with: python -m pytest tests/test_per_type_stats.py -v -o addopts= -s
"""

import pytest
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.models.model_prep_models import (
    DpoBaselineStats,
    GrpoBaselineStats,
    InstructBaselineStats,
)
from trainer.model_prep.stats import compute_text_stats


MODEL_ID = "distilgpt2"
MAX_SAMPLES = 20


@pytest.fixture(scope="module")
def model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    return model, tok


@pytest.fixture(scope="module")
def instruct_data():
    """yahma/alpaca-cleaned → standardized instruct columns."""
    ds = load_dataset("yahma/alpaca-cleaned", split=f"train[:{MAX_SAMPLES}]")
    return [
        {
            "instruct": r["instruction"],
            "output": r["output"],
            "input": r["input"] if r["input"] else None,
        }
        for r in ds
    ]


@pytest.fixture(scope="module")
def dpo_data():
    """Intel/orca_dpo_pairs → standardized DPO columns."""
    ds = load_dataset("Intel/orca_dpo_pairs", split=f"train[:{MAX_SAMPLES}]")
    return [
        {
            "prompt": r["question"],
            "chosen": r["chosen"],
            "rejected": r["rejected"],
        }
        for r in ds
    ]


@pytest.fixture(scope="module")
def grpo_data():
    """trl-lib/tldr → standardized GRPO columns (prompt only)."""
    ds = load_dataset("trl-lib/tldr", split=f"train[:{MAX_SAMPLES}]")
    return [{"prompt": r["prompt"]} for r in ds]


@pytest.fixture(scope="module")
def chat_data():
    """Construct multi-turn chat conversations from alpaca data."""
    ds = load_dataset("yahma/alpaca-cleaned", split=f"train[:{MAX_SAMPLES * 2}]")
    records = []
    for i in range(0, len(ds) - 1, 2):
        r1, r2 = ds[i], ds[i + 1]
        if not r1["output"].strip() or not r2["output"].strip():
            continue
        records.append({
            "conversations": [
                {"from": "user", "value": r1["instruction"]},
                {"from": "assistant", "value": r1["output"]},
                {"from": "user", "value": r2["instruction"]},
                {"from": "assistant", "value": r2["output"]},
            ]
        })
        if len(records) >= MAX_SAMPLES:
            break
    return records


class TestInstructStats:
    def test_returns_correct_type(self, model_and_tokenizer, instruct_data):
        model, tok = model_and_tokenizer
        stats = compute_text_stats(model, tok, instruct_data, task_type="instruct", max_samples=MAX_SAMPLES)
        assert isinstance(stats, InstructBaselineStats)

    def test_prompt_and_completion_tokens(self, model_and_tokenizer, instruct_data):
        model, tok = model_and_tokenizer
        stats = compute_text_stats(model, tok, instruct_data, task_type="instruct", max_samples=MAX_SAMPLES)
        assert stats.dataset.prompt_tokens > 0
        assert stats.dataset.completion_tokens > 0
        assert stats.dataset.prompt_tokens != stats.dataset.completion_tokens
        print(f"Prompt tokens: {stats.dataset.prompt_tokens}, Completion tokens: {stats.dataset.completion_tokens}")

    def test_completion_length_distribution(self, model_and_tokenizer, instruct_data):
        model, tok = model_and_tokenizer
        stats = compute_text_stats(model, tok, instruct_data, task_type="instruct", max_samples=MAX_SAMPLES)
        d = stats.dataset.completion_length_distribution
        assert d.mean > 0
        assert d.max >= d.p99 >= d.p95 >= d.p50
        print(f"Completion lengths: mean={d.mean:.1f}, p50={d.p50}, p95={d.p95}, max={d.max}")

    def test_masked_loss_differs_from_init(self, model_and_tokenizer, instruct_data):
        model, tok = model_and_tokenizer
        stats = compute_text_stats(model, tok, instruct_data, task_type="instruct", max_samples=MAX_SAMPLES)
        assert stats.training.masked_completion_loss > 0
        assert stats.training.masked_completion_loss != stats.training.init_loss
        print(f"Init loss: {stats.training.init_loss:.3f}, Masked loss: {stats.training.masked_completion_loss:.3f}")

    def test_all_fields_populated(self, model_and_tokenizer, instruct_data):
        model, tok = model_and_tokenizer
        stats = compute_text_stats(model, tok, instruct_data, task_type="instruct", max_samples=MAX_SAMPLES)
        assert stats.dataset.bits_per_byte > 0
        assert stats.dataset.vocab_size > 0
        assert len(stats.weights.by_group) > 0
        assert len(stats.training.grad_norms) > 0
        assert stats.training.output_entropy > 0


class TestDpoStats:
    def test_returns_correct_type(self, model_and_tokenizer, dpo_data):
        model, tok = model_and_tokenizer
        stats = compute_text_stats(model, tok, dpo_data, task_type="dpo", max_samples=MAX_SAMPLES)
        assert isinstance(stats, DpoBaselineStats)

    def test_separate_token_counts(self, model_and_tokenizer, dpo_data):
        model, tok = model_and_tokenizer
        stats = compute_text_stats(model, tok, dpo_data, task_type="dpo", max_samples=MAX_SAMPLES)
        assert stats.dataset.prompt_tokens > 0
        assert stats.dataset.chosen_tokens > 0
        assert stats.dataset.rejected_tokens > 0
        print(f"Prompt: {stats.dataset.prompt_tokens}, Chosen: {stats.dataset.chosen_tokens}, Rejected: {stats.dataset.rejected_tokens}")

    def test_chosen_rejected_distributions(self, model_and_tokenizer, dpo_data):
        model, tok = model_and_tokenizer
        stats = compute_text_stats(model, tok, dpo_data, task_type="dpo", max_samples=MAX_SAMPLES)
        assert stats.dataset.chosen_length_distribution.mean > 0
        assert stats.dataset.rejected_length_distribution.mean > 0
        print(f"Chosen mean: {stats.dataset.chosen_length_distribution.mean:.1f}, Rejected mean: {stats.dataset.rejected_length_distribution.mean:.1f}")

    def test_length_ratio(self, model_and_tokenizer, dpo_data):
        model, tok = model_and_tokenizer
        stats = compute_text_stats(model, tok, dpo_data, task_type="dpo", max_samples=MAX_SAMPLES)
        assert stats.dataset.chosen_rejected_length_ratio > 0
        print(f"Chosen/Rejected length ratio: {stats.dataset.chosen_rejected_length_ratio:.3f}")

    def test_log_probs_and_reward_gap(self, model_and_tokenizer, dpo_data):
        model, tok = model_and_tokenizer
        stats = compute_text_stats(model, tok, dpo_data, task_type="dpo", max_samples=MAX_SAMPLES)
        assert isinstance(stats.training.ref_log_prob_chosen, float)
        assert isinstance(stats.training.ref_log_prob_rejected, float)
        # Log probs should be negative
        assert stats.training.ref_log_prob_chosen < 0
        assert stats.training.ref_log_prob_rejected < 0
        # Reward gap is the difference
        expected_gap = stats.training.ref_log_prob_chosen - stats.training.ref_log_prob_rejected
        assert abs(stats.training.implicit_reward_gap - expected_gap) < 1e-6
        print(f"Chosen log-prob: {stats.training.ref_log_prob_chosen:.4f}")
        print(f"Rejected log-prob: {stats.training.ref_log_prob_rejected:.4f}")
        print(f"Implicit reward gap: {stats.training.implicit_reward_gap:.4f}")

    def test_all_fields_populated(self, model_and_tokenizer, dpo_data):
        model, tok = model_and_tokenizer
        stats = compute_text_stats(model, tok, dpo_data, task_type="dpo", max_samples=MAX_SAMPLES)
        assert len(stats.weights.by_group) > 0
        assert len(stats.training.grad_norms) > 0
        assert stats.training.output_entropy > 0


class TestGrpoStats:
    def test_returns_correct_type(self, model_and_tokenizer, grpo_data):
        model, tok = model_and_tokenizer
        stats = compute_text_stats(model, tok, grpo_data, task_type="grpo", max_samples=MAX_SAMPLES)
        assert isinstance(stats, GrpoBaselineStats)

    def test_prompt_tokens(self, model_and_tokenizer, grpo_data):
        model, tok = model_and_tokenizer
        stats = compute_text_stats(model, tok, grpo_data, task_type="grpo", max_samples=MAX_SAMPLES)
        assert stats.dataset.prompt_tokens > 0
        assert stats.dataset.prompt_length_distribution.mean > 0
        print(f"Prompt tokens: {stats.dataset.prompt_tokens}")
        print(f"Prompt length mean: {stats.dataset.prompt_length_distribution.mean:.1f}")

    def test_reward_baseline_with_simple_function(self, model_and_tokenizer, grpo_data):
        model, tok = model_and_tokenizer
        reward_fn_code = '''def reward_conciseness(completions, **kwargs):
    """Reward shorter completions."""
    return [100.0 / (len(c.split()) + 10) for c in completions]
'''
        from core.models.utility_models import RewardFunction
        reward_fns = [RewardFunction(reward_func=reward_fn_code, reward_weight=1.0)]

        stats = compute_text_stats(
            model, tok, grpo_data, task_type="grpo",
            max_samples=MAX_SAMPLES, reward_functions=reward_fns,
        )
        assert len(stats.training.baseline_reward_scores) > 0
        for name, score in stats.training.baseline_reward_scores.items():
            assert isinstance(score, float)
            print(f"Reward '{name}': {score:.4f}")

    def test_multiple_reward_functions(self, model_and_tokenizer, grpo_data):
        model, tok = model_and_tokenizer
        fn1 = '''def reward_length(completions, **kwargs):
    return [len(c.split()) for c in completions]
'''
        fn2 = '''def reward_has_period(completions, **kwargs):
    return [1.0 if "." in c else 0.0 for c in completions]
'''
        from core.models.utility_models import RewardFunction
        reward_fns = [
            RewardFunction(reward_func=fn1, reward_weight=0.5),
            RewardFunction(reward_func=fn2, reward_weight=0.5),
        ]

        stats = compute_text_stats(
            model, tok, grpo_data, task_type="grpo",
            max_samples=MAX_SAMPLES, reward_functions=reward_fns,
        )
        assert len(stats.training.baseline_reward_scores) == 2
        scores = list(stats.training.baseline_reward_scores.values())
        assert scores[0] != scores[1], "Different reward functions should give different scores"
        for name, score in stats.training.baseline_reward_scores.items():
            print(f"Reward '{name}': {score:.4f}")

    def test_all_fields_populated(self, model_and_tokenizer, grpo_data):
        model, tok = model_and_tokenizer
        stats = compute_text_stats(model, tok, grpo_data, task_type="grpo", max_samples=MAX_SAMPLES)
        assert len(stats.weights.by_group) > 0
        assert len(stats.training.grad_norms) > 0
        assert stats.training.output_entropy > 0


class TestChatStats:
    def test_returns_instruct_type(self, model_and_tokenizer, chat_data):
        """Chat uses InstructBaselineStats (same as instruct)."""
        model, tok = model_and_tokenizer
        stats = compute_text_stats(model, tok, chat_data, task_type="chat", max_samples=MAX_SAMPLES)
        assert isinstance(stats, InstructBaselineStats)

    def test_prompt_completion_from_turns(self, model_and_tokenizer, chat_data):
        model, tok = model_and_tokenizer
        stats = compute_text_stats(model, tok, chat_data, task_type="chat", max_samples=MAX_SAMPLES)
        assert stats.dataset.prompt_tokens > 0
        assert stats.dataset.completion_tokens > 0
        print(f"User (prompt) tokens: {stats.dataset.prompt_tokens}")
        print(f"Assistant (completion) tokens: {stats.dataset.completion_tokens}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
