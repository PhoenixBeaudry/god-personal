FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN pip install --no-cache-dir safetensors huggingface_hub numpy requests pydantic transformers datasketch aiohttp sglang

COPY trainer/model_prep/ trainer/model_prep/
COPY core/ core/

# Validator modules needed for env stats (eval_environment reuse)
RUN mkdir -p validator/evaluation validator/core validator/utils
COPY validator/evaluation/eval_environment.py validator/evaluation/eval_environment.py
COPY validator/evaluation/utils.py validator/evaluation/utils.py
COPY validator/core/constants.py validator/core/constants.py
RUN touch validator/__init__.py validator/evaluation/__init__.py validator/core/__init__.py validator/utils/__init__.py

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "trainer/model_prep/entrypoint.py"]
