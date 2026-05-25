import argparse
import asyncio

from core.logging import get_logger
from core.models.utility_models import ChatTemplateDatasetType
from core.models.utility_models import FileFormat
from validator.evaluation.local_evaluation import run_evaluation_docker_text


logger = get_logger(__name__)


async def run_probe(args: argparse.Namespace) -> None:
    custom_dataset_type = ChatTemplateDatasetType(
        chat_template=args.chat_template,
        chat_column=args.chat_column,
        chat_role_field=args.chat_role_field,
        chat_content_field=args.chat_content_field,
        chat_user_reference=args.chat_user_reference,
        chat_assistant_reference=args.chat_assistant_reference,
    )

    results = await run_evaluation_docker_text(
        dataset=args.dataset,
        models=args.models.split(","),
        original_model=args.original_model,
        dataset_type=custom_dataset_type,
        file_format=FileFormat.JSON,
        gpu_ids=[int(gpu_id) for gpu_id in args.gpu_ids.split(",")],
    )
    logger.info(f"Evaluation results: {results}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a manual text evaluation probe in Docker.")
    parser.add_argument("--dataset", default="/tmp/728c0ffac41d1699_test_data.json")
    parser.add_argument("--models", default="diagonalge/8ad2b90f-7b3e-4b67-9741-3f3c2ecc53eb")
    parser.add_argument("--original-model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--chat-template", default="chatml")
    parser.add_argument("--chat-column", default="conversations")
    parser.add_argument("--chat-role-field", default="from")
    parser.add_argument("--chat-content-field", default="value")
    parser.add_argument("--chat-user-reference", default="human")
    parser.add_argument("--chat-assistant-reference", default="gpt")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run_probe(parse_args()))
