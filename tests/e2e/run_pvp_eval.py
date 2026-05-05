#!/usr/bin/env python3
"""
End-to-end PvP evaluation test. Runs two real models through the full pipeline:
  build config → start SGLang servers → play games → verify results

Requires:
  - 2x GPU (one per model)
  - HuggingFace access (models downloaded by SGLang at startup)
  - open_spiel installed

Usage:
    python tests/e2e/run_pvp_eval.py
    python tests/e2e/run_pvp_eval.py --model Qwen/Qwen2.5-3B-Instruct --num-games 5
    python tests/e2e/run_pvp_eval.py --model-a org/lora-adapter --base-model Qwen/Qwen2.5-3B-Instruct

The default runs both models as the same base model (sanity check: results should
be roughly 50/50 since both models play identically). To test LoRA, pass --model-a
pointing to a LoRA adapter repo.
"""

import argparse
import json
import sys
import time

from core.constants import EnvironmentName
from core.models.pvp_models import (
    PvPEvalConfig,
    PvPMatchupConfig,
    PvPModelSpec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E2E PvP evaluation test")
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-3B-Instruct",
        help="Model repo for both players (symmetric test). Overridden by --model-a/--model-b.",
    )
    parser.add_argument("--model-a", default=None, help="Model A repo (overrides --model)")
    parser.add_argument("--model-b", default=None, help="Model B repo (overrides --model)")
    parser.add_argument(
        "--base-model", default="Qwen/Qwen2.5-3B-Instruct",
        help="Base model for LoRA detection",
    )
    parser.add_argument("--num-games", type=int, default=3, help="Games per environment (each played twice)")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--gpu-a", type=int, default=0, help="GPU for model A")
    parser.add_argument("--gpu-b", type=int, default=1, help="GPU for model B")
    parser.add_argument(
        "--envs", nargs="+", default=["liars_dice", "leduc_poker"],
        choices=[e.value for e in EnvironmentName],
        help="Environments to test",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PvPEvalConfig:
    model_a_repo = args.model_a or args.model
    model_b_repo = args.model_b or args.model

    matchups = {
        EnvironmentName(env): PvPMatchupConfig(num_games=args.num_games)
        for env in args.envs
    }

    return PvPEvalConfig(
        model_a=PvPModelSpec(
            repo=model_a_repo,
            original_model=args.base_model,
            gpu_id=args.gpu_a,
        ),
        model_b=PvPModelSpec(
            repo=model_b_repo,
            original_model=args.base_model,
            gpu_id=args.gpu_b,
        ),
        matchups=matchups,
        seed=args.seed,
        temperature=args.temperature,
    )


def validate_results(config: PvPEvalConfig, results_json: dict) -> list[str]:
    """Return list of validation errors (empty = pass)."""
    errors: list[str] = []

    if results_json["model_a"] != config.model_a.repo:
        errors.append(f"model_a mismatch: {results_json['model_a']} != {config.model_a.repo}")
    if results_json["model_b"] != config.model_b.repo:
        errors.append(f"model_b mismatch: {results_json['model_b']} != {config.model_b.repo}")

    for env_name, matchup_config in config.matchups.items():
        env_key = env_name.value
        if env_key not in results_json["results"]:
            errors.append(f"Missing results for {env_key}")
            continue

        env_result = results_json["results"][env_key]
        expected_total = matchup_config.num_games * 2

        if env_result["total_games"] != expected_total:
            errors.append(
                f"{env_key}: total_games={env_result['total_games']}, expected={expected_total}"
            )

        accounting = env_result["model_a_wins"] + env_result["model_b_wins"] + env_result["draws"]
        if accounting != env_result["total_games"]:
            errors.append(
                f"{env_key}: wins+losses+draws={accounting} != total_games={env_result['total_games']}"
            )

    metadata = results_json.get("metadata", {})
    if metadata.get("wall_time_seconds", 0) <= 0:
        errors.append("wall_time_seconds should be positive")

    return errors


def main() -> int:
    args = parse_args()
    config = build_config(args)

    print("=" * 60)
    print("PvP Evaluation E2E Test")
    print("=" * 60)
    print(f"Model A: {config.model_a.repo} (GPU {config.model_a.gpu_id})")
    print(f"Model B: {config.model_b.repo} (GPU {config.model_b.gpu_id})")
    print(f"Environments: {[e.value for e in config.matchups]}")
    print(f"Games per env: {args.num_games} (x2 for position swap)")
    print(f"Seed: {config.seed}, Temperature: {config.temperature}")
    print("=" * 60)

    # Import here so arg parsing / --help works without all deps installed
    from validator.evaluation.utils import configure_eval_logging
    from validator.evaluation.pvp.__main__ import _run_evaluation

    configure_eval_logging()

    start = time.time()
    try:
        results = _run_evaluation(config)
    except Exception as exc:
        print(f"\nFAILED: {exc}")
        import traceback
        traceback.print_exc()
        return 1

    elapsed = time.time() - start
    results_json = json.loads(results.model_dump_json())

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(json.dumps(results_json, indent=2))

    print("\n" + "=" * 60)
    print("Validation")
    print("=" * 60)

    errors = validate_results(config, results_json)
    if errors:
        for error in errors:
            print(f"  FAIL: {error}")
        return 1

    print("  All checks passed.")
    print(f"\n  Wall time: {elapsed:.1f}s")

    # Symmetric test check: same model on both sides should be roughly even
    is_symmetric = config.model_a.repo == config.model_b.repo
    if is_symmetric:
        print("\n  Symmetric test (same model both sides):")
        for env_key, env_result in results_json["results"].items():
            total = env_result["total_games"]
            a_pct = env_result["model_a_wins"] / total * 100 if total else 0
            b_pct = env_result["model_b_wins"] / total * 100 if total else 0
            d_pct = env_result["draws"] / total * 100 if total else 0
            print(f"    {env_key}: A={a_pct:.0f}% B={b_pct:.0f}% D={d_pct:.0f}%")
        print("  (With position swap, these should trend toward 50/50 over many games)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
