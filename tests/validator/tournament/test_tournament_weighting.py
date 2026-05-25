import pytest

import validator.shared.constants as cts
from core.models.tournament_models import TournamentAuditData
from core.models.tournament_models import TournamentResultsWithWinners
from core.models.tournament_models import TournamentType
from validator.tournament.weighting import calculate_participation_scale
from validator.tournament.weighting import calculate_scaled_tournament_weights
from validator.tournament.weighting import get_audit_tournament_data
from validator.tournament.weighting import get_audit_tournament_weight
from validator.tournament.weighting import set_audit_tournament_data
from validator.tournament.weighting import set_audit_tournament_weight


def _results(winner_hotkey: str, base_winner_hotkey: str | None = None) -> TournamentResultsWithWinners:
    return TournamentResultsWithWinners(
        tournament_id="tournament",
        rounds=[],
        base_winner_hotkey=base_winner_hotkey,
        winner_hotkey=winner_hotkey,
    )


def test_audit_data_accessors_preserve_legacy_fields():
    audit_data = TournamentAuditData()
    text_results = _results("text-winner")

    set_audit_tournament_data(audit_data, TournamentType.TEXT, text_results)
    set_audit_tournament_weight(audit_data, TournamentType.IMAGE, 0.42)

    assert audit_data.text_tournament_data == text_results
    assert audit_data.image_tournament_weight == pytest.approx(0.42)
    assert get_audit_tournament_data(audit_data, TournamentType.TEXT) == text_results
    assert get_audit_tournament_weight(audit_data, TournamentType.IMAGE) == pytest.approx(0.42)


def test_calculate_scaled_tournament_weights_uses_specs_and_real_winners():
    audit_data = TournamentAuditData(
        participants=["alice", "bob"],
        text_tournament_data=_results(cts.EMISSION_BURN_HOTKEY, base_winner_hotkey="defending-champ"),
        image_tournament_data=_results("image-winner"),
        environment_tournament_data=None,
        text_tournament_weight=0.30,
        image_tournament_weight=0.20,
        environment_tournament_weight=0.10,
        burn_weight=0.40,
    )

    participation_total, scale_factor = calculate_participation_scale(audit_data.participants)
    scaled = calculate_scaled_tournament_weights(audit_data)

    assert participation_total == pytest.approx(2 * cts.TOURNAMENT_PARTICIPATION_WEIGHT)
    assert scaled.scale_factor == pytest.approx(scale_factor)
    assert scaled.tournament_weight(TournamentType.TEXT) == pytest.approx(0.30 * scale_factor)
    assert scaled.tournament_weight(TournamentType.IMAGE) == pytest.approx(0.20 * scale_factor)
    assert scaled.tournament_weight(TournamentType.ENVIRONMENT) == pytest.approx(0.10 * scale_factor)
    assert scaled.burn_weight == pytest.approx(0.40 * scale_factor)
    assert scaled.base_weight(TournamentType.TEXT) == pytest.approx(cts.TOURNAMENT_TEXT_WEIGHT * scale_factor)
    assert scaled.winner_hotkey(TournamentType.TEXT) == "defending-champ"
    assert scaled.winner_hotkey(TournamentType.IMAGE) == "image-winner"
    assert scaled.winner_hotkey(TournamentType.ENVIRONMENT) is None
