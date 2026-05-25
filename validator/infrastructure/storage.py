import json
import os
import tempfile

import httpx
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_exponential

from core.logging import get_logger
from validator.infrastructure.minio_client import async_minio_client


logger = get_logger(__name__)

retry_http_with_backoff = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError)),
    reraise=True,
)

retry_with_backoff = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)

retry_http_fast = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.2, min=0.1, max=1),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError)),
    reraise=True,
)

async def save_json_to_temp_file(data: list[dict], prefix: str, dump_json: bool = True) -> tuple[str, int]:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json", prefix=prefix)
    if dump_json:
        with open(temp_file.name, "w") as f:
            json.dump(data, f)
    else:
        with open(temp_file.name, "w") as f:
            f.write(data)
    file_size = os.path.getsize(temp_file.name)
    return temp_file.name, file_size


async def upload_file_to_minio(file_path: str, bucket_name: str, object_name: str) -> str | None:
    """
    Uploads a file to MinIO and returns the presigned URL for the uploaded file.
    """
    result = await async_minio_client.upload_file(bucket_name, object_name, file_path)
    if result:
        return await async_minio_client.get_presigned_url(bucket_name, object_name)
    else:
        return None
