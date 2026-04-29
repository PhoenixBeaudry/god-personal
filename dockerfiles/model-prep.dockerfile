FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN pip install --no-cache-dir safetensors huggingface_hub numpy requests pydantic "transformers>=4.40,<5" datasketch aiohttp sglang python-dotenv

COPY trainer/model_prep/ trainer/model_prep/
COPY core/ core/

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "trainer/model_prep/entrypoint.py"]
