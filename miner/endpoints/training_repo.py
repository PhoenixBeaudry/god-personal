import os

from fastapi import Depends
from fastapi import HTTPException
from fastapi.routing import APIRouter
from fiber.miner.dependencies import blacklist_low_stake
from fiber.miner.dependencies import verify_get_request

from core.models.payload_models import TrainingRepoResponse
from core.models.tournament_models import TournamentType
from core.service_paths import TRAINING_REPO_ENDPOINT


DEFAULT_TRAINING_REPO = "https://github.com/rayonlabs/G.O.D"
DEFAULT_TRAINING_COMMIT = "5f161f642cd578b829e72dedd8444a491b9bbca3"


def _task_env_prefix(task_type: TournamentType) -> str:
    return f"MINER_{task_type.value.upper()}"


def _env_for_task(task_type: TournamentType, name: str, default: str | None = None) -> str | None:
    task_specific = os.getenv(f"{_task_env_prefix(task_type)}_{name}")
    if task_specific:
        return task_specific
    return os.getenv(f"MINER_{name}", default)


def _parse_requested_datasets(raw_value: str | None) -> list[str] | None:
    if not raw_value:
        return None
    datasets = [dataset.strip() for dataset in raw_value.split(",") if dataset.strip()]
    return datasets or None


async def get_training_repo(task_type: TournamentType) -> TrainingRepoResponse:
    github_repo = _env_for_task(task_type, "TRAINING_REPO", DEFAULT_TRAINING_REPO)
    commit_hash = _env_for_task(task_type, "TRAINING_COMMIT", DEFAULT_TRAINING_COMMIT)
    github_token = _env_for_task(task_type, "GITHUB_TOKEN")
    requested_datasets = _parse_requested_datasets(_env_for_task(task_type, "REQUESTED_DATASETS"))

    if not github_repo or not commit_hash:
        raise HTTPException(
            status_code=503,
            detail="Miner training repository is not configured",
        )

    return TrainingRepoResponse(
        github_repo=github_repo,
        commit_hash=commit_hash,
        github_token=github_token,
        requested_datasets=requested_datasets,
    )


def factory_router() -> APIRouter:
    router = APIRouter(tags=["Miner"])

    router.add_api_route(
        f"{TRAINING_REPO_ENDPOINT}/{{task_type}}",
        get_training_repo,
        methods=["GET"],
        response_model=TrainingRepoResponse,
        summary="Get Training Repository",
        description="Return the miner training repository and commit hash for a tournament family.",
        dependencies=[Depends(blacklist_low_stake), Depends(verify_get_request)],
    )

    return router
