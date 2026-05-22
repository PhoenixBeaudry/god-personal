# Image for validator/evaluation/eval_swe.py. It runs a local SGLang
# server and the Affinetes SWE-INFINITE env server in the same Basilica deployment.
# Build manually:
#   docker build -f dockerfiles/validator-swe.dockerfile -t phoenixbeaudry/env-eval-swe:basilica .

ARG SWE_BASE_IMAGE=phoenixbeaudry/swe-infinite:v1
ARG SGLANG_BASE_IMAGE=lmsysorg/sglang:latest

FROM ${SWE_BASE_IMAGE} AS swe_runtime

FROM ${SGLANG_BASE_IMAGE}

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl docker.io ca-certificates libnuma1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=swe_runtime /app /app
COPY --from=swe_runtime /tmp/requirements.txt /tmp/swe-infinite-requirements.txt
COPY --from=swe_runtime /usr/local/bin/codex-static /usr/local/bin/codex-static

RUN chmod +x /usr/local/bin/codex-static \
    && pip install --no-cache-dir --upgrade-strategy only-if-needed -r /tmp/swe-infinite-requirements.txt

RUN pip install --no-cache-dir --upgrade-strategy only-if-needed \
    fastapi "uvicorn[standard]" httpx pydantic structlog \
    git+https://github.com/AffineFoundation/affinetes.git@main

RUN python -c "from importlib import resources; from pathlib import Path; d = Path('/app/_affinetes'); d.mkdir(parents=True, exist_ok=True); d.joinpath('__init__.py').write_text(''); d.joinpath('server.py').write_text(resources.files('affinetes').joinpath('templates/http_server.py').read_text()); Path('/app/request_logger.py').write_text(resources.files('affinetes').joinpath('templates/request_logger.py').read_text())" \
    && chmod -R 777 /app/_affinetes

WORKDIR /validator-app

COPY pyproject.toml README.md ./
RUN mkdir -p src && touch src/__init__.py

RUN pip install --no-cache-dir --upgrade-strategy only-if-needed .

RUN pip install --no-cache-dir --upgrade-strategy only-if-needed \
    peft==0.18.1 accelerate==1.6.0 aiohttp openai

COPY . /validator-app

ENV PYTHONPATH="/validator-app:/app"
ENV SGLANG_PORT=30000
ENV SGLANG_BASE_URL=http://127.0.0.1:30000
ENV SGLANG_HEALTH_PATH=/v1/models
ENV ENV_SERVER_BASE_URL=http://127.0.0.1:8001
ENV ENV_SERVER_HEALTH_PATH=/health
ENV AFFINETES_PORT=8001

ENV SWE_ENV_SERVER_CMD="cd /app && python -m uvicorn _affinetes.server:app --host 0.0.0.0 --port 8001 --workers 1 --loop asyncio"
