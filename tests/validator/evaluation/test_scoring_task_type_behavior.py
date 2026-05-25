from core.models.utility_models import TaskType
from validator.evaluation.ranking import calculate_miner_ranking_and_scores
from validator.shared.models import MinerResultsText


def _result(hotkey: str, loss: float, task_type: TaskType) -> MinerResultsText:
    return MinerResultsText(
        hotkey=hotkey,
        test_loss=loss,
        synth_loss=loss,
        is_finetune=True,
        task_type=task_type,
    )


def test_text_loss_ranking_keeps_lower_loss_first():
    ranked = calculate_miner_ranking_and_scores([
        _result("alice", 0.2, TaskType.INSTRUCTTEXTTASK),
        _result("bob", 0.8, TaskType.INSTRUCTTEXTTASK),
    ])

    assert ranked[0].score == 3
    assert ranked[0].score_reason == "Ranked 1st by test_loss"
    assert ranked[1].score == 0.0


def test_grpo_ranking_keeps_higher_score_first():
    ranked = calculate_miner_ranking_and_scores([
        _result("alice", 0.2, TaskType.GRPOTASK),
        _result("bob", 0.8, TaskType.GRPOTASK),
    ])

    assert ranked[1].score == 3
    assert ranked[1].score_reason == "Ranked 1st by GRPO score (bigger is better)"
    assert ranked[0].score == 0.0


def test_environment_ranking_keeps_higher_score_first():
    ranked = calculate_miner_ranking_and_scores([
        _result("alice", 0.2, TaskType.ENVIRONMENTTASK),
        _result("bob", 0.8, TaskType.ENVIRONMENTTASK),
    ])

    assert ranked[1].score == 3
    assert ranked[1].score_reason == "Ranked 1st by Environment score (bigger is better)"
    assert ranked[0].score == 0.0
