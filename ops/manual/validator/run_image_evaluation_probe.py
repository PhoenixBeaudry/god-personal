import argparse
import asyncio

from core.logging import get_logger
from core.models.utility_models import ImageModelType
from validator.evaluation.local_evaluation import run_evaluation_docker_image


logger = get_logger(__name__)


async def run_probe(args: argparse.Namespace) -> None:
    results = await run_evaluation_docker_image(
        test_split_url=args.test_split_url,
        original_model_repo=args.original_model,
        models=args.models.split(","),
        model_type=ImageModelType(args.model_type),
        gpu_ids=[int(gpu_id) for gpu_id in args.gpu_ids.split(",")],
    )
    logger.info(f"Evaluation results: {results}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a manual image evaluation probe in Docker.")
    parser.add_argument("--test-split-url", required=True)
    parser.add_argument("--original-model", default="Qwen/Qwen-Image")
    parser.add_argument("--models", default="gradients-io-tournaments/qwenimage-test")
    parser.add_argument(
        "--model-type",
        default=ImageModelType.QWEN_IMAGE.value,
        choices=[model.value for model in ImageModelType],
    )
    parser.add_argument("--gpu-ids", default="0")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run_probe(parse_args()))
