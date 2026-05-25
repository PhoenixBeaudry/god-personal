#!/usr/bin/env python3



from core.logging import get_logger
from core.models.tournament_models import TournamentData
from core.models.tournament_models import TournamentResultsWithWinners
from validator.shared.constants import EMISSION_BURN_HOTKEY


logger = get_logger(__name__)

def get_real_winner_hotkey(winner_hotkey: str | None, base_winner_hotkey: str | None) -> str | None:
    """
    Get the real hotkey of the tournament winner.

    If winner_hotkey is EMISSION_BURN_HOTKEY (defending champion defended),
    returns base_winner_hotkey (the real defending champion's hotkey).
    Otherwise returns winner_hotkey.

    This is needed because when a defending champion successfully defends,
    winner_hotkey is set to EMISSION_BURN_HOTKEY as a placeholder, and
    base_winner_hotkey contains their actual hotkey.

    Args:
        winner_hotkey: The tournament's winner_hotkey field
        base_winner_hotkey: The tournament's base_winner_hotkey field (defending champion snapshot)

    Returns:
        Real winner's hotkey, or None if no winner
    """
    if not winner_hotkey:
        return None

    if winner_hotkey == EMISSION_BURN_HOTKEY and base_winner_hotkey:
        return base_winner_hotkey

    return winner_hotkey


def get_real_tournament_winner(tournament: TournamentData | TournamentResultsWithWinners | None) -> str | None:
    """
    Get the real tournament winner hotkey, accounting for EMISSION_BURN_HOTKEY.

    When a defending champion wins, winner_hotkey is set to EMISSION_BURN_HOTKEY,
    and the actual winner hotkey is stored in base_winner_hotkey.
    """
    if not tournament or not tournament.winner_hotkey:
        return None

    winner = tournament.winner_hotkey
    if winner == EMISSION_BURN_HOTKEY:
        winner = tournament.base_winner_hotkey

    return winner


def did_winner_change(previous_tournament: TournamentData | None, latest_tournament: TournamentData) -> bool:
    """
    Determine if the tournament winner changed between two tournaments.

    Returns True if:
    - No previous tournament exists (first tournament)
    - Latest winner is a real hotkey (not EMISSION_BURN_HOTKEY)

    Returns:
        True if winner should be treated as a new winner, False if defending champion won via placeholder
    """
    if not previous_tournament:
        return True

    # EMISSION_BURN_HOTKEY explicitly marks a defending champion win.
    # Any real hotkey winner should be treated as "new winner" for fresh perf diff calc,
    # even if it's the same hotkey as a previous tournament.
    if latest_tournament.winner_hotkey != EMISSION_BURN_HOTKEY:
        return True

    return False
