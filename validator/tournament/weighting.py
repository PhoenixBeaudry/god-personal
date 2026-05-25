from dataclasses import dataclass

import validator.shared.constants as cts
from core.models.tournament_models import TournamentAuditData
from core.models.tournament_models import TournamentResultsWithWinners
from core.models.tournament_models import TournamentType
from validator.tournament.champions import get_real_tournament_winner
from validator.tournament.specs import get_tournament_spec
from validator.tournament.specs import tournament_types


@dataclass(frozen=True)
class ScaledTournamentWeights:
    participation_total: float
    scale_factor: float
    tournament_weight_by_type: dict[TournamentType, float]
    base_weight_by_type: dict[TournamentType, float]
    winner_hotkey_by_type: dict[TournamentType, str | None]
    burn_weight: float

    def tournament_weight(self, tournament_type: TournamentType) -> float:
        return self.tournament_weight_by_type[tournament_type]

    def base_weight(self, tournament_type: TournamentType) -> float:
        return self.base_weight_by_type[tournament_type]

    def winner_hotkey(self, tournament_type: TournamentType) -> str | None:
        return self.winner_hotkey_by_type[tournament_type]


def get_audit_tournament_data(
    tournament_audit_data: TournamentAuditData,
    tournament_type: TournamentType,
) -> TournamentResultsWithWinners | None:
    return getattr(tournament_audit_data, get_tournament_spec(tournament_type).audit_data_field)


def set_audit_tournament_data(
    tournament_audit_data: TournamentAuditData,
    tournament_type: TournamentType,
    tournament_data: TournamentResultsWithWinners | None,
) -> None:
    setattr(tournament_audit_data, get_tournament_spec(tournament_type).audit_data_field, tournament_data)


def get_audit_tournament_weight(
    tournament_audit_data: TournamentAuditData,
    tournament_type: TournamentType,
) -> float:
    return getattr(tournament_audit_data, get_tournament_spec(tournament_type).audit_weight_field)


def set_audit_tournament_weight(
    tournament_audit_data: TournamentAuditData,
    tournament_type: TournamentType,
    weight: float,
) -> None:
    setattr(tournament_audit_data, get_tournament_spec(tournament_type).audit_weight_field, weight)


def calculate_participation_scale(participants: list[str]) -> tuple[float, float]:
    participation_total = len(participants) * cts.TOURNAMENT_PARTICIPATION_WEIGHT
    scale_factor = 1.0 - participation_total if participation_total > 0 else 1.0
    return participation_total, scale_factor


def calculate_scaled_tournament_weights(tournament_audit_data: TournamentAuditData) -> ScaledTournamentWeights:
    participation_total, scale_factor = calculate_participation_scale(tournament_audit_data.participants)

    tournament_weight_by_type = {}
    base_weight_by_type = {}
    winner_hotkey_by_type = {}
    for tournament_type in tournament_types():
        spec = get_tournament_spec(tournament_type)
        tournament_weight_by_type[tournament_type] = get_audit_tournament_weight(
            tournament_audit_data,
            tournament_type,
        ) * scale_factor
        base_weight_by_type[tournament_type] = spec.base_weight * scale_factor
        winner_hotkey_by_type[tournament_type] = get_real_tournament_winner(
            get_audit_tournament_data(tournament_audit_data, tournament_type)
        )

    return ScaledTournamentWeights(
        participation_total=participation_total,
        scale_factor=scale_factor,
        tournament_weight_by_type=tournament_weight_by_type,
        base_weight_by_type=base_weight_by_type,
        winner_hotkey_by_type=winner_hotkey_by_type,
        burn_weight=tournament_audit_data.burn_weight * scale_factor,
    )
