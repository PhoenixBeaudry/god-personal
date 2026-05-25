"""Model and tokenizer loading helpers for text evaluation."""

import inspect
import json
import os
import re

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from core.logging import get_logger
from validator.evaluation.container_results import load_results_dict
from validator.evaluation.container_results import save_results_dict
from validator.infrastructure.retries import retry_on_5xx
from validator.shared import constants as cst


logger = get_logger(__name__)


def _iter_wrapped_models(model, seen: set[int] | None = None):
    if seen is None:
        seen = set()

    model_id = id(model)
    if model_id in seen:
        return

    seen.add(model_id)
    yield model
    for attr_name in ("base_model", "model"):
        wrapped_model = getattr(model, attr_name, None)
        if wrapped_model is not None and wrapped_model is not model:
            yield from _iter_wrapped_models(wrapped_model, seen)


def _model_explicitly_accepts_kwarg(model, kwarg_name: str) -> bool:
    for candidate_model in _iter_wrapped_models(model):
        for attr_name in ("forward", "prepare_inputs_for_generation"):
            method = getattr(candidate_model, attr_name, None)
            if method is None:
                continue
            try:
                parameters = inspect.signature(method).parameters
            except (TypeError, ValueError):
                continue
            if kwarg_name in parameters:
                return True
    return False


def sanitize_tokenizer_for_models(tokenizer: AutoTokenizer, *models: AutoModelForCausalLM) -> AutoTokenizer:
    """
    Remove token_type_ids only when one of the loaded models does not advertise support.
    This keeps segment ids available for the rare models that actually use them.
    """
    model_input_names = getattr(tokenizer, "model_input_names", None)
    if not model_input_names or "token_type_ids" not in model_input_names:
        return tokenizer

    if not models:
        return tokenizer

    unsupported_models = [
        type(model).__name__ for model in models if not _model_explicitly_accepts_kwarg(model, "token_type_ids")
    ]
    if not unsupported_models:
        return tokenizer

    tokenizer.model_input_names = [name for name in model_input_names if name != "token_type_ids"]
    logger.info(
        "Removed token_type_ids from tokenizer inputs because these models do not support them: "
        + ", ".join(unsupported_models)
    )
    return tokenizer


def create_finetuned_cache_dir() -> str:
    """Create and return a dedicated cache directory for finetuned models."""
    finetuned_cache_dir = os.path.join(cst.DOCKER_EVAL_HF_CACHE_DIR, "finetuned_repos")
    os.makedirs(finetuned_cache_dir, exist_ok=True)
    return finetuned_cache_dir


def patch_base_model_config_if_needed(base_model_name: str, cache_dir: str, context: str = "") -> bool:
    """
    Patch base model config.json if head_dim or partial_rotary_factor is None.

    This fixes Yarn models where head_dim is None in the config, which causes
    TypeError during model loading.
    """
    try:
        base_cache_path = os.path.join(cache_dir, "hub", f"models--{base_model_name.replace('/', '--')}")

        if not os.path.exists(base_cache_path):
            return False

        base_snapshots_dir = os.path.join(base_cache_path, "snapshots")
        if not os.path.exists(base_snapshots_dir):
            return False

        base_snapshots = sorted(os.listdir(base_snapshots_dir))
        if not base_snapshots:
            return False

        base_snapshot_path = os.path.join(base_snapshots_dir, base_snapshots[-1])
        base_config_file = os.path.join(base_snapshot_path, "config.json")

        if not os.path.exists(base_config_file):
            return False

        with open(base_config_file) as cfg_f:
            base_config_dict = json.load(cfg_f)

        needs_patch = False

        if base_config_dict.get("head_dim") is None:
            if base_config_dict.get("hidden_size") and base_config_dict.get("num_attention_heads"):
                calculated_head_dim = base_config_dict["hidden_size"] // base_config_dict["num_attention_heads"]
                base_config_dict["head_dim"] = calculated_head_dim
                context_str = f" ({context})" if context else ""
                logger.info(f"Patching head_dim={calculated_head_dim} in base model config{context_str}")
                needs_patch = True

        is_yarn_rope = base_config_dict.get("rope_scaling", {}).get("type") == "yarn"
        if base_config_dict.get("partial_rotary_factor") is None and is_yarn_rope:
            base_config_dict["partial_rotary_factor"] = 1.0
            context_str = f" ({context})" if context else ""
            logger.info(f"Patching partial_rotary_factor=1.0 in base model config{context_str}")
            needs_patch = True

        if needs_patch:
            with open(base_config_file, "w") as cfg_f:
                json.dump(base_config_dict, cfg_f, indent=2)
            context_str = f" ({context})" if context else ""
            logger.info(f"Patched base model config.json at {base_config_file}{context_str}")
            return True

        return False
    except Exception as e:
        logger.warning(f"Failed to patch base model config for {base_model_name}: {e}", exc_info=True)
        return False


@retry_on_5xx()
def load_model(model_name_or_path: str, is_base_model: bool = False, local_files_only: bool = False) -> AutoModelForCausalLM:
    try:
        if local_files_only:
            cache_dir = os.path.expanduser("~/.cache/huggingface")
            cache_path = os.path.join(cache_dir, "hub", f"models--{model_name_or_path.replace('/', '--')}")

            if os.path.exists(cache_path):
                snapshots_dir = os.path.join(cache_path, "snapshots")
                if os.path.exists(snapshots_dir):
                    snapshots = sorted(os.listdir(snapshots_dir))

                    for snapshot in snapshots:
                        snapshot_path = os.path.join(snapshots_dir, snapshot)
                        files = os.listdir(snapshot_path)

                        has_model_files = any(f.endswith((".bin", ".safetensors")) for f in files)
                        has_config = "config.json" in files

                        if has_model_files and has_config:
                            try:
                                model = AutoModelForCausalLM.from_pretrained(
                                    snapshot_path,
                                    device_map="balanced",
                                    torch_dtype=torch.bfloat16,
                                    local_files_only=local_files_only,
                                )
                                return model
                            except Exception as e:
                                logger.warning(f"Failed to load from snapshot {snapshot}: {e}")
                                continue

        if local_files_only:
            cache_dir = os.path.expanduser("~/.cache/huggingface")
        elif not is_base_model:
            cache_dir = create_finetuned_cache_dir()
        else:
            cache_dir = None

        kwargs = {
            "device_map": "balanced",
            "cache_dir": cache_dir,
            "torch_dtype": torch.bfloat16,
            "local_files_only": local_files_only,
        }
        if not local_files_only:
            kwargs["token"] = os.environ.get("HUGGINGFACE_TOKEN")

        return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    except RuntimeError as e:
        error_msg = str(e)
        if "size mismatch for" in error_msg and ("lm_head.weight" in error_msg or "model.embed_tokens.weight" in error_msg):
            pattern = re.search(r"shape torch\.Size\(\[(\d+), (\d+)\]\).*shape.*torch\.Size\(\[(\d+), \2\]\)", error_msg)
            if pattern and abs(int(pattern.group(1)) - int(pattern.group(3))) == 1:
                logger.info("Detected vocabulary size off-by-one error, attempting to load with ignore_mismatched_sizes=True")
                kwargs["ignore_mismatched_sizes"] = True
                return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        logger.error(f"Exception type: {type(e)}, message: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Exception type: {type(e)}, message: {str(e)}")
        raise


@retry_on_5xx()
def load_tokenizer(original_model: str, local_files_only: bool = False) -> AutoTokenizer:
    try:
        if local_files_only:
            cache_dir = os.path.expanduser("~/.cache/huggingface")
            cache_path = os.path.join(cache_dir, "hub", f"models--{original_model.replace('/', '--')}")

            if os.path.exists(cache_path):
                snapshots_dir = os.path.join(cache_path, "snapshots")
                if os.path.exists(snapshots_dir):
                    snapshots = sorted(os.listdir(snapshots_dir))

                    for snapshot in snapshots:
                        snapshot_path = os.path.join(snapshots_dir, snapshot)
                        files = os.listdir(snapshot_path)
                        tokenizer_files = [f for f in files if "tokenizer" in f.lower() or f.endswith(".model")]

                        if tokenizer_files:
                            try:
                                tokenizer = AutoTokenizer.from_pretrained(snapshot_path, local_files_only=True)
                                return tokenizer
                            except Exception as e:
                                logger.warning(f"Failed to load from snapshot {snapshot}: {e}")
                                continue

        kwargs = {
            "local_files_only": local_files_only,
            "cache_dir": os.path.expanduser("~/.cache/huggingface") if local_files_only else None,
        }
        if not local_files_only:
            kwargs["token"] = os.environ.get("HUGGINGFACE_TOKEN")

        return AutoTokenizer.from_pretrained(original_model, **kwargs)
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {str(e)}")
        logger.debug("Full traceback:", exc_info=True)
        raise


@retry_on_5xx()
def load_finetuned_model(repo: str, local_files_only: bool = False) -> AutoPeftModelForCausalLM:
    try:
        if local_files_only:
            cache_dir = os.path.expanduser("~/.cache/huggingface")
            cache_path = os.path.join(cache_dir, "hub", f"models--{repo.replace('/', '--')}")

            if os.path.exists(cache_path):
                snapshots_dir = os.path.join(cache_path, "snapshots")
                if os.path.exists(snapshots_dir):
                    snapshots = sorted(os.listdir(snapshots_dir))

                    for snapshot in snapshots:
                        snapshot_path = os.path.join(snapshots_dir, snapshot)
                        files = os.listdir(snapshot_path)

                        has_adapter = any("adapter" in f.lower() for f in files)

                        if has_adapter:
                            try:
                                adapter_config_path = os.path.join(snapshot_path, "adapter_config.json")
                                if os.path.exists(adapter_config_path):
                                    with open(adapter_config_path) as f:
                                        adapter_config = json.load(f)
                                        base_model_name = adapter_config.get("base_model_name_or_path")

                                        if base_model_name:
                                            patch_base_model_config_if_needed(base_model_name, cache_dir)

                                model = AutoPeftModelForCausalLM.from_pretrained(
                                    snapshot_path,
                                    is_trainable=False,
                                    device_map="balanced",
                                    torch_dtype=torch.bfloat16,
                                    local_files_only=True,
                                )
                                return model
                            except Exception as e:
                                logger.warning(f"Failed to load from snapshot {snapshot}: {e}", exc_info=True)
                                continue

        cache_dir = os.path.expanduser("~/.cache/huggingface") if local_files_only else create_finetuned_cache_dir()

        kwargs = {
            "is_trainable": False,
            "device_map": "balanced",
            "cache_dir": cache_dir,
            "torch_dtype": torch.bfloat16,
            "local_files_only": local_files_only,
        }
        if not local_files_only:
            kwargs["token"] = os.environ.get("HUGGINGFACE_TOKEN")

        return AutoPeftModelForCausalLM.from_pretrained(repo, **kwargs)
    except RuntimeError as e:
        error_msg = str(e)
        if "size mismatch for" in error_msg and ("lm_head.weight" in error_msg or "model.embed_tokens.weight" in error_msg):
            pattern = re.search(r"shape torch\.Size\(\[(\d+), (\d+)\]\).*shape.*torch\.Size\(\[(\d+), \2\]\)", error_msg)
            if pattern and abs(int(pattern.group(1)) - int(pattern.group(3))) == 1:
                logger.info("Detected vocabulary size off-by-one error, attempting to load with ignore_mismatched_sizes=True")
                kwargs["ignore_mismatched_sizes"] = True
                return AutoPeftModelForCausalLM.from_pretrained(repo, **kwargs)

        logger.error(f"Exception type: {type(e)}, message: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Exception type: {type(e)}, message: {str(e)}")
        raise


def count_model_parameters(model) -> int:
    """Count the total number of parameters in a model."""
    try:
        return sum(p.numel() for p in model.parameters())
    except Exception as e:
        logger.error(f"Failed to count model parameters: {e}")
        return 0


def check_and_log_base_model_size(original_model: str) -> None:
    """Check if base model size is logged in results; if not, load and log it."""
    results_dict = load_results_dict()

    if "model_params_count" not in results_dict:
        logger.info("Base model size not logged, loading base model to calculate size")
        base_model = load_model(original_model, is_base_model=True)
        results_dict["model_params_count"] = count_model_parameters(base_model)
        save_results_dict(results_dict)
        logger.info(f"Logged base model size: {results_dict['model_params_count']} parameters")
    else:
        logger.info(f"Base model size already logged: {results_dict['model_params_count']} parameters")
