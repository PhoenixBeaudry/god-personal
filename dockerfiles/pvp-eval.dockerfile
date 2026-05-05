FROM lmsysorg/sglang:latest

WORKDIR /app

COPY pyproject.toml README.md ./
RUN mkdir -p src && touch src/__init__.py

RUN pip install --no-cache-dir --upgrade-strategy only-if-needed .

RUN pip install --no-cache-dir --upgrade-strategy only-if-needed \
    open_spiel \
    peft==0.18.1 \
    accelerate==1.6.0

RUN apt-get update && apt-get install -y --no-install-recommends libnuma1 && rm -rf /var/lib/apt/lists/*

COPY . /app

ENV PVP_EVAL_CONFIG=""
ENV EVAL_LOG_LEVEL="INFO"
ENV CUDA_VISIBLE_DEVICES="0,1"

ENTRYPOINT ["python", "-m", "validator.evaluation.pvp"]
