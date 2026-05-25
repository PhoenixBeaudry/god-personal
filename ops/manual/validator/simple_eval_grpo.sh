#!/bin/bash
set -euo pipefail

# Manual GRPO evaluation probe.
# Usage:
#   ops/manual/validator/simple_eval_grpo.sh <dataset-url-or-local-json>

if [ "$#" -lt 1 ]; then
  echo "Usage: ops/manual/validator/simple_eval_grpo.sh <dataset-url-or-local-json>" >&2
  exit 2
fi

DATASET_SOURCE="$1"
TEMP_DIR=$(mktemp -d)
DATASET_FILE="$TEMP_DIR/dataset.json"
RESULTS_DIR="$PWD/grpo_results"
RESULTS_COPY="./grpo_eval_results_$(date +%s).json"

cleanup() {
  rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

mkdir -p "$RESULTS_DIR"

if [[ "$DATASET_SOURCE" == http://* || "$DATASET_SOURCE" == https://* ]]; then
  echo "Downloading dataset..."
  curl -L -o "$DATASET_FILE" "$DATASET_SOURCE"
else
  echo "Copying local dataset..."
  cp "$DATASET_SOURCE" "$DATASET_FILE"
fi

echo "Dataset ready at $DATASET_FILE"
echo "Dataset size: $(wc -c < "$DATASET_FILE") bytes"
echo "Dataset preview:"
head -n 5 "$DATASET_FILE"

echo "Starting GRPO evaluation..."
docker run --rm \
  -e DATASET="/workspace/input_data/dataset.json" \
  -e MODELS="robiual-awal/c621c6f1-40be-4a54-add1-38585b4e002f,Alphatao/3e05bf5e-0a8a-4c96-bf01-7a2d82bd333c" \
  -e ORIGINAL_MODEL="EleutherAI/pythia-70m" \
  -e DATASET_TYPE='{"field_prompt":"prompt","reward_functions":[{"reward_func":"def reward_func(completions, **kwargs):\n    return [text.count(\"e\") / (len(text) + 1) for text in completions]","reward_weight":0.7,"name":"e_counter"},{"reward_func":"def reward_func(completions, **kwargs):\n    return [min(len(text)/100, 1.0) for text in completions]","reward_weight":0.3,"name":"length_scorer"}]}' \
  -e FILE_FORMAT="json" \
  -v "$TEMP_DIR:/workspace/input_data:rw" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:rw" \
  -v "$RESULTS_DIR:/aplp:rw" \
  --runtime nvidia \
  -e CUDA_VISIBLE_DEVICES=0 \
  --gpus '"device=0"' \
  weightswandering/tuning_vali:latest \
  python -m validator.evaluation.eval_grpo

echo "Checking for result files..."
RESULTS_FILE=$(find "$TEMP_DIR" -type f -name "*evaluation_results.json" | head -n 1)

if [ -z "$RESULTS_FILE" ]; then
  RESULTS_FILE="$RESULTS_DIR/evaluation_results.json"
  if [ ! -f "$RESULTS_FILE" ]; then
    echo "No evaluation results found." >&2
    echo "TEMP_DIR contents:"
    find "$TEMP_DIR" -type f | sort
    echo "RESULTS_DIR contents:"
    find "$RESULTS_DIR" -type f | sort
    exit 1
  fi
fi

echo "Results content:"
cat "$RESULTS_FILE"
cp "$RESULTS_FILE" "$RESULTS_COPY"
echo "Results copied to $RESULTS_COPY"

echo ""
echo "=== VALIDATOR SCORING SIMULATION ==="
python3 - "$RESULTS_COPY" << 'ENDPYTHON'
import json
import sys

FIRST_PLACE_SCORE = 3.0

with open(sys.argv[1]) as f:
    results = json.load(f)

models_scores = []
for model_name, model_data in results.items():
    if isinstance(model_data, dict) and model_data.get("is_finetune", False):
        models_scores.append((model_name, model_data["eval_loss"]))

models_scores.sort(key=lambda item: -item[1])
print(f"\nFound {len(models_scores)} valid models")

for rank, (model, score) in enumerate(models_scores, 1):
    validator_score = FIRST_PLACE_SCORE if rank == 1 else 0.0
    print(f"\nRank {rank}: {model}")
    print(f"  GRPO Score: {score:.4f}")
    print(f"  Validator Score: {validator_score}")

if models_scores:
    top_model = models_scores[0][0]
    model_info = results[top_model]
    print(f"\nScoring Details (Top Model: {top_model})")
    if "raw_rewards" in model_info:
        print("  Raw Rewards:")
        for func_name, value in model_info["raw_rewards"].items():
            weight = model_info.get("reward_weights", {}).get(func_name, "?")
            print(f"    {func_name}: {value:.4f} (weight: {weight})")
    if "individual_rewards" in model_info and "wrapper" in model_info["individual_rewards"]:
        print(f"  Combined Reward: {model_info['individual_rewards']['wrapper']:.4f}")
    print(f"  GRPO Score (used for ranking): {model_info['eval_loss']:.4f}")
    print(f"  Validator Score: {FIRST_PLACE_SCORE} (top model only)")
ENDPYTHON

echo "Temporary files cleaned up. Results directory at: $RESULTS_DIR"
