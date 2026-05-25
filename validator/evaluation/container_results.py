import json
import os

from core.logging import get_logger
from validator.shared import constants as cst


logger = get_logger(__name__)

def load_results_dict():
    """Load existing evaluation results or create an empty dict if not found."""
    results_dict = {}
    output_dir = os.path.dirname(cst.CONTAINER_EVAL_RESULTS_PATH)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(cst.CONTAINER_EVAL_RESULTS_PATH):
        try:
            with open(cst.CONTAINER_EVAL_RESULTS_PATH, "r") as f:
                results_dict = json.load(f)
        except Exception as e:
            logger.error(f"Could not read existing results from {cst.CONTAINER_EVAL_RESULTS_PATH}, starting fresh: {e}")

    return results_dict


def save_results_dict(results_dict, model_id=None):
    """Save evaluation results to file."""
    with open(cst.CONTAINER_EVAL_RESULTS_PATH, "w") as f:
        json.dump(results_dict, f, indent=2)

    msg = "Saved evaluation results"
    if model_id:
        msg += f" for {model_id}"

    logger.info(msg)
    logger.info(json.dumps(results_dict, indent=2))
