# Image for validator/evaluation/eval_swe.py. It runs a local SGLang
# server and a SWE-Infinite env server in the same Basilica deployment.
# Build manually:
#   docker build -f dockerfiles/validator-swe.dockerfile -t phoenixbeaudry/env-eval-swe:basilica .

FROM phoenixbeaudry/swe-infinite:context

WORKDIR /validator-app

COPY pyproject.toml README.md ./
RUN mkdir -p src && touch src/__init__.py

RUN pip install --no-cache-dir --upgrade-strategy only-if-needed .

RUN pip install --no-cache-dir --upgrade-strategy only-if-needed \
    "sglang[all]" peft==0.18.1 accelerate==1.6.0 aiohttp openai

COPY . /validator-app

ENV PYTHONPATH="/validator-app:/app"
ENV SGLANG_PORT=30000
ENV SGLANG_BASE_URL=http://127.0.0.1:30000
ENV SGLANG_HEALTH_PATH=/v1/models
ENV ENV_SERVER_BASE_URL=http://127.0.0.1:8001
ENV ENV_SERVER_HEALTH_PATH=/health

# The SWE image has historically exposed the FastAPI app from /app/server.py.
# eval_swe.py will also probe common alternatives if this command is unset.
ENV SWE_ENV_SERVER_CMD="cd /app && python -m uvicorn server:app --host 0.0.0.0 --port 8001 --workers 1 --loop asyncio"
