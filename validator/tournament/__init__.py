from importlib import import_module


__all__ = [
    "organise_tournament_round",
    "create_text_tournament_tasks",
    "create_image_tournament_tasks",
    "create_environment_tournament_tasks",
]


def __getattr__(name: str):
    if name in {
        "benchmark_utils",
        "brackets",
        "champions",
        "constants",
        "dstack_orchestrator",
        "environment_results",
        "notifications",
        "orchestrator",
        "participants",
        "performance_calculator",
        "performance_utils",
        "repo_diff_report",
        "repo_uploader",
        "reports",
        "resources",
        "round_results",
        "runner",
        "task_creator",
        "task_results",
        "thresholds",
        "tournament_manager",
        "trainer_client",
        "transfer_monitoring",
    }:
        return import_module(f"{__name__}.{name}")

    if name == "organise_tournament_round":
        from validator.tournament.tournament_manager import organise_tournament_round

        return organise_tournament_round

    if name in {"create_text_tournament_tasks", "create_image_tournament_tasks", "create_environment_tournament_tasks"}:
        from validator.tournament.task_creator import create_environment_tournament_tasks
        from validator.tournament.task_creator import create_image_tournament_tasks
        from validator.tournament.task_creator import create_text_tournament_tasks

        exports = {
            "create_text_tournament_tasks": create_text_tournament_tasks,
            "create_image_tournament_tasks": create_image_tournament_tasks,
            "create_environment_tournament_tasks": create_environment_tournament_tasks,
        }
        return exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
