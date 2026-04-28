"""
Test stats on real HF datasets — easy vs hard.
Verifies that the stats meaningfully distinguish between datasets.
Run with: python -m pytest tests/test_dataset_comparison.py -v -o addopts= -s
"""

import pytest
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trainer.model_prep.stats import compute_baseline_stats


MODEL_ID = "distilgpt2"


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="module")
def model():
    return AutoModelForCausalLM.from_pretrained(MODEL_ID)


@pytest.fixture(scope="module")
def easy_dataset():
    """Clean English Wikipedia — GPT2 should predict this well."""
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return [{"text": r["text"]} for r in wiki if len(r["text"].strip()) > 50][:50]


@pytest.fixture(scope="module")
def hard_dataset(tokenizer):
    """Random token sequences — impossible for the model to predict."""
    import random
    rng = random.Random(42)
    vocab = list(tokenizer.get_vocab().keys())[:5000]
    return [{"text": " ".join(rng.choices(vocab, k=30))} for _ in range(50)]


@pytest.fixture(scope="module")
def easy_stats(model, tokenizer, easy_dataset):
    return compute_baseline_stats(model, tokenizer, easy_dataset, max_samples=50)


@pytest.fixture(scope="module")
def hard_stats(model, tokenizer, hard_dataset):
    return compute_baseline_stats(model, tokenizer, hard_dataset, max_samples=50)


class TestDatasetDifferences:
    """Stats should clearly distinguish easy from hard data."""

    def test_hard_has_higher_loss(self, easy_stats, hard_stats):
        print(f"Easy loss: {easy_stats.training.init_loss:.3f}")
        print(f"Hard loss: {hard_stats.training.init_loss:.3f}")
        assert hard_stats.training.init_loss > easy_stats.training.init_loss

    def test_hard_has_higher_bpb(self, easy_stats, hard_stats):
        print(f"Easy BPB: {easy_stats.dataset.bits_per_byte:.3f}")
        print(f"Hard BPB: {hard_stats.dataset.bits_per_byte:.3f}")
        assert hard_stats.dataset.bits_per_byte > easy_stats.dataset.bits_per_byte

    def test_different_seq_length_distributions(self, easy_stats, hard_stats):
        easy_mean = easy_stats.dataset.seq_length_distribution.mean
        hard_mean = hard_stats.dataset.seq_length_distribution.mean
        print(f"Easy mean seq len: {easy_mean:.1f}")
        print(f"Hard mean seq len: {hard_mean:.1f}")
        # Different datasets should have different length distributions
        assert easy_mean != hard_mean

    def test_different_duplicate_rates(self, easy_stats, hard_stats):
        print(f"Easy dup rate: {easy_stats.dataset.near_duplicate_rate:.3f}")
        print(f"Hard dup rate: {hard_stats.dataset.near_duplicate_rate:.3f}")
        # Random tokens should have ~0 duplicates, wiki might have some
        assert hard_stats.dataset.near_duplicate_rate < 0.1

    def test_hard_has_higher_output_entropy(self, easy_stats, hard_stats):
        print(f"Easy entropy: {easy_stats.training.output_entropy:.3f}")
        print(f"Hard entropy: {hard_stats.training.output_entropy:.3f}")
        # Model should be more uncertain on random tokens
        assert hard_stats.training.output_entropy > easy_stats.training.output_entropy

    def test_weight_stats_identical(self, easy_stats, hard_stats):
        """Weight stats should not change — same model, different data."""
        for group in easy_stats.weights.by_group:
            easy_rms = easy_stats.weights.by_group[group].weight_rms
            hard_rms = hard_stats.weights.by_group[group].weight_rms
            assert abs(easy_rms - hard_rms) < 1e-6, f"Weight RMS differs for {group}"

    def test_vocab_size_same(self, easy_stats, hard_stats):
        assert easy_stats.dataset.vocab_size == hard_stats.dataset.vocab_size


class TestStatsValues:
    """Sanity checks on absolute values."""

    def test_easy_loss_reasonable(self, easy_stats):
        # distilgpt2 on English wikipedia should be ~3-5
        assert 2.0 < easy_stats.training.init_loss < 8.0

    def test_hard_loss_high(self, hard_stats):
        # Random tokens should give higher loss than natural text
        assert hard_stats.training.init_loss > 5.0

    def test_bpb_positive(self, easy_stats, hard_stats):
        assert easy_stats.dataset.bits_per_byte > 0
        assert hard_stats.dataset.bits_per_byte > 0

    def test_easy_bpb_under_2(self, easy_stats):
        # English text should be under 2 BPB for GPT2
        assert easy_stats.dataset.bits_per_byte < 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
