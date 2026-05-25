from math import ceil

import torch
import torch.nn.functional as F
from accelerate.utils import find_executable_batch_size
from datasets import Dataset
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from core.logging import get_logger
from validator.shared import constants as cst


logger = get_logger(__name__)

def calculate_kl_divergence(
    original_model: AutoModelForCausalLM,
    finetuned_model: AutoModelForCausalLM,
    dataset: Dataset,
    tokenizer: AutoTokenizer,
) -> float:
    """
    Calculate KL divergence between original and finetuned model outputs on a dataset.

    Args:
        original_model: The original/base model
        finetuned_model: The finetuned model
        dataset: Dataset to evaluate on
        tokenizer: Tokenizer for text processing

    Returns:
        Average KL divergence across the dataset
    """
    logger.info("Starting KL divergence calculation...")

    # Calculate max_length using same logic as GRPO evaluation
    max_length = cst.GRPO_KL_SEQUENCE_LENGTH
    max_embeddings = getattr(finetuned_model.config, "max_position_embeddings", None)
    if max_embeddings and max_embeddings < 2 * max_length:
        max_length = ceil(max_embeddings / 2)

    original_model.eval()
    finetuned_model.eval()

    @find_executable_batch_size(starting_batch_size=cst.GRPO_KL_BATCH_SIZE)
    def calculate_kl_with_batch_size(batch_size):
        logger.info(f"Attempting KL divergence calculation with batch size: {batch_size}")

        total_kl_div = 0.0
        total_samples = 0

        # Process dataset in batches
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            prompts = batch[cst.TRL_GRPO_FIELD_PROMPT]

            try:
                inputs = tokenizer(prompts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
                inputs = {k: v.cuda() for k, v in inputs.items()}
            except Exception as e:
                logger.warning(f"Failed to tokenize batch starting at index {i}: {e}")
                continue

            with torch.no_grad():
                try:
                    # Get logits from both models
                    original_outputs = original_model(**inputs)
                    finetuned_outputs = finetuned_model(**inputs)

                    original_logits = original_outputs.logits
                    finetuned_logits = finetuned_outputs.logits

                    # Convert logits to probabilities
                    original_probs = F.softmax(original_logits, dim=-1)
                    finetuned_log_probs = F.log_softmax(finetuned_logits, dim=-1)

                    # Calculate KL divergence: KL(original || finetuned)
                    kl_div = F.kl_div(finetuned_log_probs, original_probs, reduction="none")

                    # Average over sequence length and vocabulary, sum over batch
                    batch_kl = kl_div.sum(dim=-1).mean(dim=-1).sum().item()

                    total_kl_div += batch_kl
                    total_samples += len(prompts)

                    if (i // batch_size) % 10 == 0:
                        logger.info(f"Processed {i + len(prompts)} samples, current batch KL: {batch_kl / len(prompts):.6f}")

                except Exception as e:
                    logger.warning(f"Failed to compute KL divergence for batch starting at index {i}: {e}")
                    continue
                finally:
                    torch.cuda.empty_cache()

        if total_samples == 0:
            logger.error("No samples were successfully processed for KL divergence calculation")
            raise ValueError("No samples were successfully processed for KL divergence calculation")

        avg_kl_div = total_kl_div / total_samples
        logger.info(f"KL divergence calculation completed. Average KL divergence: {avg_kl_div:.6f} over {total_samples} samples")

        return avg_kl_div

    return calculate_kl_with_batch_size()
