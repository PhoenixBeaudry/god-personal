# Image for validator/evaluation/eval_swe.py.
#
# This image is the Basilica SWE dispatcher and the reusable SGLang server
# runtime. Task repositories are evaluated by deploying their task-specific
# dockerhub_tag images separately and injecting the stdlib worker source from
# validator/evaluation/swe_basilica/worker_runtime.py.
#
# Build manually:
#   docker build -f dockerfiles/validator-swe.dockerfile -t phoenixbeaudry/env-eval-swe:basilica .

FROM lmsysorg/sglang:latest

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl ca-certificates libnuma1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /validator-app

COPY pyproject.toml README.md ./
RUN mkdir -p src && touch src/__init__.py

RUN pip install --no-cache-dir --upgrade-strategy only-if-needed .

RUN pip install --no-cache-dir --upgrade-strategy only-if-needed \
    peft==0.18.1 accelerate==1.6.0 aiohttp openai

COPY . /validator-app

ENV PYTHONPATH="/validator-app"
ENV SGLANG_PORT=30000
ENV SGLANG_HEALTH_PATH=/v1/models
ENV TRANSFORMERS_ALLOW_TORCH_LOAD=true
