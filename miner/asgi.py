import os
import threading
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fiber.miner.core import configuration

from core.logging import get_logger
from miner.endpoints.training_repo import factory_router as training_repo_router


load_dotenv(os.getenv("ENV_FILE", ".miner.env"))

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = configuration.factory_config()
    metagraph = config.metagraph
    sync_thread = None

    if metagraph.substrate is not None:
        sync_thread = threading.Thread(target=metagraph.periodically_sync_nodes, daemon=True)
        sync_thread.start()

    yield

    logger.info("Shutting down miner...")
    metagraph.shutdown()
    if metagraph.substrate is not None and sync_thread is not None:
        sync_thread.join()


def factory() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.include_router(training_repo_router())
    return app


app = factory()


if __name__ == "__main__":
    logger.info("Starting miner")
    uvicorn.run(app, host="0.0.0.0", port=7999)
