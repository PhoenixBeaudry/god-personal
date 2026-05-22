from __future__ import annotations

import glob
import logging
import os
import time

from validator.evaluation.eval_environment import _build_sglang_command
from validator.evaluation.eval_environment import _download_lora_with_retry
from validator.evaluation.eval_environment import _download_model_with_retry
from validator.evaluation.eval_environment import _merge_base_and_lora
from validator.evaluation.utils import check_for_lora
from validator.evaluation.utils import check_lora_has_added_tokens


logger = logging.getLogger(__name__)


async def prepare_sglang_for_model(model_repo: str, original_model: str, base_seed: int) -> tuple[str, str, str]:
    """Resolve model artifacts and build the SGLang launch command."""

    sglang_command = os.getenv("SGLANG_START_CMD")
    if sglang_command:
        logger.info("SGLang: using SGLANG_START_CMD from environment")
        return model_repo, model_repo, sglang_command

    import asyncio

    t_det = time.time()
    is_lora = await asyncio.to_thread(check_for_lora, model_repo, False)
    should_merge_lora = False
    if is_lora:
        should_merge_lora = await asyncio.to_thread(check_lora_has_added_tokens, model_repo, False)
    logger.info(
        "LoRA detection in %.2fs: is_lora=%s merge_lora_to_base=%s",
        time.time() - t_det,
        is_lora,
        should_merge_lora,
    )

    if is_lora and not should_merge_lora:
        logger.info("Model path: LoRA + SGLang native (base=%s lora_repo=%s)", original_model, model_repo)
        model_path_for_sglang = await asyncio.to_thread(_download_model_with_retry, original_model)
        lora_dir = "/lora/trained_lora"
        await asyncio.to_thread(_download_lora_with_retry, model_repo, lora_dir)
        for model_file in glob.glob(os.path.join(lora_dir, "model-*.safetensors")):
            try:
                os.remove(model_file)
                logger.info("Removed incompatible LoRA file: %s", os.path.basename(model_file))
            except Exception as exc:
                logger.warning("Failed to remove %s: %s", model_file, exc)
        index_file = os.path.join(lora_dir, "model.safetensors.index.json")
        if os.path.exists(index_file):
            try:
                os.remove(index_file)
            except Exception as exc:
                logger.warning("Failed to remove index file: %s", exc)
        inference_model_name = f"{original_model}:trained_lora"
        command = (
            _build_sglang_command(model_path_for_sglang, base_seed)
            + " --enable-lora --lora-paths trained_lora=/lora/trained_lora --lora-backend triton"
        )
        return inference_model_name, model_path_for_sglang, command

    if is_lora and should_merge_lora:
        logger.info("Model path: merge LoRA into base then SGLang (base=%s lora=%s)", original_model, model_repo)
        base_path = await asyncio.to_thread(_download_model_with_retry, original_model)
        lora_temp_dir = "/tmp/lora/trained_lora"
        await asyncio.to_thread(_download_lora_with_retry, model_repo, lora_temp_dir)
        model_path_for_sglang = await asyncio.to_thread(_merge_base_and_lora, base_path, lora_temp_dir)
        return model_repo, model_path_for_sglang, _build_sglang_command(model_path_for_sglang, base_seed)

    logger.info("Model path: single HF repo (full weights) repo=%s", model_repo)
    model_path_for_sglang = await asyncio.to_thread(_download_model_with_retry, model_repo)
    return model_repo, model_path_for_sglang, _build_sglang_command(model_path_for_sglang, base_seed)

