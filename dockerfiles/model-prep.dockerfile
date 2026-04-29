FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir safetensors huggingface_hub numpy requests pydantic transformers datasketch aiohttp sglang python-dotenv basilica-sdk "fiber @ git+https://github.com/besimray/fiber.git@v2.6.0" docker loguru astor minio

COPY trainer/model_prep/ trainer/model_prep/
COPY core/ core/

# Validator modules
COPY validator/ validator/

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "trainer/model_prep/entrypoint.py"]
