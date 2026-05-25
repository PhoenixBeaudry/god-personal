import os
from math import ceil

import yaml
from axolotl.utils.dict import DictDefault
from transformers import AutoModelForCausalLM

from core.logging import get_logger
from core.training_config import create_dataset_entry
from validator.shared.models import EvaluationArgs


logger = get_logger(__name__)

def check_env_variables(required_vars):
    """Check for required environment variables."""
    env_vars = {var: os.environ.get(var, "") for var in required_vars}
    missing = [var for var, value in env_vars.items() if not value]

    if missing:
        logger.error("Missing required environment variables: " + ", ".join(missing))
        return False, env_vars

    return True, env_vars


def _load_and_update_evaluation_config(
    evaluation_args: EvaluationArgs,
    finetuned_model: AutoModelForCausalLM,
    config_path: str,
) -> DictDefault:
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)

    dataset_entry = create_dataset_entry(
        dataset=evaluation_args.dataset,
        dataset_type=evaluation_args.dataset_type,
        file_format=evaluation_args.file_format,
        is_eval=True,
    )
    config_dict["datasets"] = [dataset_entry]

    max_embeddings = getattr(finetuned_model.config, "max_position_embeddings", None)

    if max_embeddings and max_embeddings < 2 * config_dict["sequence_len"]:
        config_dict["sequence_len"] = ceil(max_embeddings / 2)

    return DictDefault(config_dict)
