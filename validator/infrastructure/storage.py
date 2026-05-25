import json
import os
import tempfile

from validator.infrastructure.minio_client import async_minio_client


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
    result = await async_minio_client.upload_file(bucket_name, object_name, file_path)
    if result:
        return await async_minio_client.get_presigned_url(bucket_name, object_name)
    return None
