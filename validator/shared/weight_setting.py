import asyncio
import os
from datetime import datetime

from dotenv import load_dotenv

from core.models.tournament_models import NodeWeightsResult
from core.models.tournament_models import TournamentAuditData
from core.models.tournament_models import TournamentBurnData
from core.models.tournament_models import TournamentData
from core.models.tournament_models import TournamentResults
from core.models.tournament_models import TournamentResultsWithWinners
from core.models.tournament_models import TournamentType
from validator.db.sql.auditing import store_latest_scores_url
from validator.db.sql.tournaments import count_champion_consecutive_wins
from validator.db.sql.tournaments import get_active_tournament_participants
from validator.db.sql.tournaments import get_latest_completed_tournament
from validator.db.sql.tournaments import get_tournament_full_results
from validator.db.sql.tournaments import get_tournament_where_champion_first_won
from validator.db.sql.tournaments import get_weekly_task_participation_data
from validator.evaluation.tournament_scoring import get_tournament_weights_from_data
from validator.tournament.champions import did_winner_change
from validator.tournament.champions import get_real_tournament_winner
from validator.tournament.performance_calculator import calculate_performance_difference
from validator.tournament.specs import get_tournament_base_weight
from validator.tournament.specs import get_tournament_max_weight
from validator.tournament.specs import get_tournament_spec
from validator.tournament.specs import tournament_types
from validator.tournament.weighting import calculate_scaled_tournament_weights
from validator.tournament.weighting import get_audit_tournament_data
from validator.tournament.weighting import get_audit_tournament_weight
from validator.tournament.weighting import set_audit_tournament_data
from validator.tournament.weighting import set_audit_tournament_weight


load_dotenv(os.getenv("ENV_FILE", ".vali.env"))

import json
from datetime import timezone
from uuid import UUID

from fiber.chain import fetch_nodes
from fiber.chain import weights
from fiber.chain.chain_utils import query_substrate
from fiber.chain.models import Node
from substrateinterface import SubstrateInterface

import validator.shared.constants as cts
from core import constants as ccst
from core.constants import BUCKET_NAME
from core.logging import get_logger
from validator.db.sql.nodes import get_vali_node_id
from validator.infrastructure.storage import save_json_to_temp_file
from validator.infrastructure.storage import upload_file_to_minio
from validator.shared.config import Config
from validator.shared.config import load_config
from validator.shared.connections import try_db_connections


logger = get_logger(__name__)


async def _upload_results_to_s3(config: Config, tournament_audit_data: TournamentAuditData) -> None:
    class DateTimeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, UUID):
                return str(obj)
            return super().default(obj)

    upload_data = {
        "tournament_audit_data": tournament_audit_data.model_dump(),
    }

    scores_json = json.dumps(upload_data, indent=2, cls=DateTimeEncoder)

    temp_file, _ = await save_json_to_temp_file(scores_json, "latest_scores", dump_json=False)
    datetime_of_upload = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    presigned_url = await upload_file_to_minio(temp_file, BUCKET_NAME, f"latest_scores_{datetime_of_upload}.json")
    os.remove(temp_file)
    await store_latest_scores_url(presigned_url, config)
    return presigned_url


def calculate_emission_boost_from_perf(performance_diff: float | None) -> float:
    if performance_diff is None:
        logger.warning("performance_diff is None, cannot calculate emission boost.")
        return 0.0
    if performance_diff <= cts.EMISSION_MULTIPLIER_THRESHOLD:
        return 0.0

    excess_performance = performance_diff - cts.EMISSION_MULTIPLIER_THRESHOLD
    emission_increase = excess_performance * cts.EMISSION_MULTIPLIER_RATE

    return emission_increase


def calculate_tournament_weight_with_decay(
    tournament_type: TournamentType,
    base_weight: float,
    emission_boost: float,
    old_decay: float,
    new_decay: float,
    apply_hybrid: bool,
    max_weight: float,
) -> float:
    """
    Apply hybrid decay logic and return final capped tournament weight.
    """
    if apply_hybrid:
        # Pre-cutoff: old_decay only affects emission_boost (emission doesn't go below base), then apply cumulative/max logic
        boost_after_old = max(0.0, emission_boost - old_decay)
        if boost_after_old == 0.0:
            final_weight = max(0.0, base_weight - new_decay)
        else:
            final_weight = max(0.0, base_weight + boost_after_old)
    else:
        # Old regime purely
        if old_decay > 0.0:
            boost_after_old = max(0.0, emission_boost - old_decay)
            final_weight = max(0.0, base_weight + boost_after_old)
        # New regime purely, we will default to this after a while or if both winners change after cutoff
        elif new_decay > 0.0:
            final_weight = max(0.0, base_weight + emission_boost - new_decay)
        else:
            final_weight = base_weight + emission_boost

    final_weight = min(final_weight, max_weight)

    logger.info(
        f"{tournament_type}: base={base_weight:.4f} + boost={emission_boost:.4f}, "
        f"old_decay={old_decay:.4f}, new_decay={new_decay:.4f}, "
        f"apply_hybrid={apply_hybrid} → final={final_weight:.4f}"
    )

    return final_weight


def calculate_hybrid_decays(
    first_championship_time: datetime, consecutive_wins: int, current_time: datetime | None = None
) -> tuple[float, float, bool]:
    """
    Calculate time-based decay & previous consecutive wins decay for backwards compatibility.
    Returns a tuple of (old_decay, new_decay, apply_hybrid).
    """
    if first_championship_time is None:
        logger.error("First championship time is None, cannot calculate time-based decay.")
        return (1.0, 1.0, False)

    # timezone alignment
    current_time_utc = current_time if current_time else datetime.now(timezone.utc)
    if current_time_utc.tzinfo is None:
        current_time_utc = current_time_utc.replace(tzinfo=timezone.utc)
    cutoff_date = datetime.combine(cts.EMISSION_TIME_DECAY_START_DATE, datetime.min.time(), tzinfo=timezone.utc)
    first_championship_time_utc = (
        first_championship_time.replace(tzinfo=timezone.utc)
        if first_championship_time.tzinfo is None
        else first_championship_time
    )

    if current_time_utc < cutoff_date:
        old_decay = max(0, consecutive_wins - 1) * cts.EMISSION_BOOST_DECAY_PER_WIN
        return (old_decay, 0.0, False)

    if first_championship_time_utc < cutoff_date:
        old_decay = max(0, consecutive_wins - 1) * cts.EMISSION_BOOST_DECAY_PER_WIN
        days_since_cutoff = (current_time_utc - cutoff_date).total_seconds() / cts.SECONDS_PER_DAY
        new_decay = days_since_cutoff * cts.EMISSION_DAILY_TIME_DECAY_RATE

        logger.debug(
            f"Pre-cutoff champion: old_decay={old_decay:.4f}, new_decay={new_decay:.4f}, will apply hybrid logic downstream"
        )
        return (old_decay, new_decay, True)
    else:
        # Champion won AFTER cutoff - only new time-based decay
        days_as_champion = (current_time_utc - first_championship_time_utc).total_seconds() / cts.SECONDS_PER_DAY
        new_decay = days_as_champion * cts.EMISSION_DAILY_TIME_DECAY_RATE
        logger.debug(f"Post-cutoff champion: new_decay={new_decay:.4f} (reign={days_as_champion:.1f} days)")
        return (0.0, new_decay, False)


def get_base_weight_by_tournament_type(tournament_type: TournamentType) -> float:
    """Get the base weight for a tournament type."""
    return get_tournament_base_weight(tournament_type)


def get_max_weight_by_tournament_type(tournament_type: TournamentType) -> float:
    """Get the max weight for a tournament type."""
    return get_tournament_max_weight(tournament_type)


def calculate_innovation_incentive(performance_diff: float | None) -> float:
    """
    Calculate innovation incentive based on performance difference.

    Args:
        performance_diff: Performance difference for the current champion

    Returns:
        Innovation incentive (emission boost based on performance)
    """
    return calculate_emission_boost_from_perf(performance_diff)


async def _get_tournament_performance_diff(psql_db, tournament_type: TournamentType) -> float | None:
    performance_diff = None
    latest_tournament = await get_latest_completed_tournament(psql_db, tournament_type)
    if not latest_tournament:
        return None

    logger.info(f"Found latest {tournament_type} tournament: {latest_tournament.tournament_id}")

    previous_tournament = await get_latest_completed_tournament(
        psql_db,
        tournament_type,
        exclude_tournament_id=latest_tournament.tournament_id,
    )

    if did_winner_change(previous_tournament, latest_tournament):
        performance_diff = await calculate_performance_difference(latest_tournament.tournament_id, psql_db)
        logger.info(f"NEW winner - calculated performance difference for {tournament_type}: {performance_diff}")
    else:
        champion_hotkey = latest_tournament.base_winner_hotkey
        if champion_hotkey:
            champion_win_tournament = await get_tournament_where_champion_first_won(
                psql_db,
                tournament_type,
                champion_hotkey,
            )
            if champion_win_tournament:
                performance_diff = champion_win_tournament.winning_performance_difference
                if performance_diff is None:
                    performance_diff = 0.0
                logger.info(
                    f"SAME winner - using stored performance difference from when {champion_hotkey} first won "
                    f"(tournament {champion_win_tournament.tournament_id}): {performance_diff:.4f}"
                )
            else:
                logger.warning(f"Could not find tournament where {champion_hotkey} first won for {tournament_type}")
                performance_diff = 0.0
        else:
            logger.warning(f"No base_winner_hotkey found for defending champion in {tournament_type}")
            performance_diff = 0.0

    if performance_diff is None:
        if latest_tournament.winner_hotkey == cts.EMISSION_BURN_HOTKEY:
            logger.info(
                f"No performance data available for {tournament_type} tournament, "
                f"burn account won - assuming worst performance (100% difference)"
            )
            return 1.0

        logger.info(
            f"No performance data available for {tournament_type} tournament, "
            f"assuming perfect performance (0% difference)"
        )
        return 0.0

    return performance_diff


async def _get_champion_decay(psql_db, tournament_type: TournamentType) -> tuple[float, float, bool]:
    latest_tournament = await get_latest_completed_tournament(psql_db, tournament_type)
    champion_hotkey = get_real_tournament_winner(latest_tournament)
    if not champion_hotkey:
        return (0.0, 0.0, False)

    consecutive_wins = await count_champion_consecutive_wins(psql_db, tournament_type, champion_hotkey)
    first_win_tournament = await get_tournament_where_champion_first_won(psql_db, tournament_type, champion_hotkey)
    if not first_win_tournament or not first_win_tournament.updated_at:
        logger.warning(f"Could not calculate decay for {tournament_type} champion {champion_hotkey[:8]}...")
        return (0.0, 0.0, False)

    old_decay, new_decay, apply_hybrid = calculate_hybrid_decays(first_win_tournament.updated_at, consecutive_wins)
    logger.info(
        f"{tournament_type.value.title()} champion {champion_hotkey[:8]}... has {consecutive_wins} consecutive wins, "
        f"first won at {first_win_tournament.updated_at}, "
        f"old_decay={old_decay:.4f}, new_decay={new_decay:.4f}, apply_hybrid={apply_hybrid}"
    )
    return (old_decay, new_decay, apply_hybrid)


async def get_tournament_burn_details(psql_db) -> TournamentBurnData:
    """
    Calculate detailed tournament burn data with calculations for TEXT, IMAGE, and ENVIRONMENT tournaments.

    This function calculates burn proportions for TEXT, IMAGE, and ENVIRONMENT tournaments,
    then applies them based on each hotkey's tournament participation.

    Returns:
        TournamentBurnData with performance metrics and weight distributions
    """
    logger.info("=== CALCULATING TOURNAMENT BURN DATA ===")

    performance_diffs = {}
    tournament_weights = {}
    burn_proportions = {}

    for tournament_type in tournament_types():
        logger.info(f"Processing {tournament_type} tournament type")
        spec = get_tournament_spec(tournament_type)
        performance_diff = await _get_tournament_performance_diff(psql_db, tournament_type)
        innovation_incentive = calculate_innovation_incentive(performance_diff)
        old_decay, new_decay, apply_hybrid = await _get_champion_decay(psql_db, tournament_type)

        performance_diffs[tournament_type] = performance_diff
        tournament_weights[tournament_type] = calculate_tournament_weight_with_decay(
            tournament_type=tournament_type,
            base_weight=spec.base_weight,
            emission_boost=innovation_incentive,
            old_decay=old_decay,
            new_decay=new_decay,
            apply_hybrid=apply_hybrid,
            max_weight=spec.max_weight,
        )
        burn_proportions[tournament_type] = (spec.max_weight - tournament_weights[tournament_type]) / spec.max_weight

        logger.info(
            f"[{tournament_type.value.upper()}] innovation_incentive={innovation_incentive:.4f} "
            f"(perf_diff={performance_diff})"
        )

    burn_weight = 1.0 - sum(tournament_weights.values())

    logger.info(
        "Weights - "
        + ", ".join(
            f"{tournament_type.value.title()} tournament: {tournament_weights[tournament_type]}"
            for tournament_type in tournament_types()
        )
    )
    logger.info(f"Total burn weight: {burn_weight}")

    return TournamentBurnData(
        text_performance_diff=performance_diffs[TournamentType.TEXT],
        image_performance_diff=performance_diffs[TournamentType.IMAGE],
        environment_performance_diff=performance_diffs[TournamentType.ENVIRONMENT],
        text_burn_proportion=burn_proportions[TournamentType.TEXT],
        image_burn_proportion=burn_proportions[TournamentType.IMAGE],
        environment_burn_proportion=burn_proportions[TournamentType.ENVIRONMENT],
        text_tournament_weight=tournament_weights[TournamentType.TEXT],
        image_tournament_weight=tournament_weights[TournamentType.IMAGE],
        environment_tournament_weight=tournament_weights[TournamentType.ENVIRONMENT],
        burn_weight=burn_weight,
    )


def _apply_single_tournament_weights(
    label: str,
    tournament_weights: dict[str, float],
    hotkey_to_node_id: dict[str, int],
    all_node_weights: list[float],
    scaled_tournament_weight: float,
    scaled_base_weight: float,
    winner_hotkey: str | None,
) -> float:
    distributed = 0.0
    label_lower = label.lower()
    logger.info(f"Processing {len(tournament_weights)} {label_lower} tournament winners")

    for hotkey, weight in tournament_weights.items():
        node_id = hotkey_to_node_id.get(hotkey)
        if node_id is None:
            continue

        scaled_weight = scaled_tournament_weight if hotkey == winner_hotkey else scaled_base_weight
        contribution = weight * scaled_weight
        all_node_weights[node_id] += contribution
        distributed += contribution

        logger.info(
            f"Node ID {node_id} (hotkey: {hotkey[:8]}...): "
            f"{label.upper()} TOURNAMENT - weight={weight:.6f}, "
            f"scaled_{label_lower}_weight={scaled_weight:.6f}, "
            f"{label_lower}_contribution={contribution:.6f}, "
            f"total_weight={all_node_weights[node_id]:.6f}"
        )

    undistributed = scaled_tournament_weight - distributed
    logger.info(
        f"{label} tournament: allocated={scaled_tournament_weight:.10f}, "
        f"distributed={distributed:.10f}, undistributed={undistributed:.10f}"
    )
    return undistributed


def apply_tournament_weights(
    text_tournament_weights: dict[str, float],
    image_tournament_weights: dict[str, float],
    environment_tournament_weights: dict[str, float],
    hotkey_to_node_id: dict[str, int],
    all_node_weights: list[float],
    scaled_text_tournament_weight: float,
    scaled_image_tournament_weight: float,
    scaled_environment_tournament_weight: float,
    scaled_text_base_weight: float,
    scaled_image_base_weight: float,
    scaled_environment_base_weight: float,
    text_winner_hotkey: str | None,
    image_winner_hotkey: str | None,
    environment_winner_hotkey: str | None,
) -> float:
    """Apply tournament weights. Returns the total undistributed weight that should go to burn."""
    logger.info("=== TOURNAMENT WEIGHT CALCULATIONS ===")

    text_undistributed = _apply_single_tournament_weights(
        "Text",
        text_tournament_weights,
        hotkey_to_node_id,
        all_node_weights,
        scaled_text_tournament_weight,
        scaled_text_base_weight,
        text_winner_hotkey,
    )
    image_undistributed = _apply_single_tournament_weights(
        "Image",
        image_tournament_weights,
        hotkey_to_node_id,
        all_node_weights,
        scaled_image_tournament_weight,
        scaled_image_base_weight,
        image_winner_hotkey,
    )
    environment_undistributed = _apply_single_tournament_weights(
        "Environment",
        environment_tournament_weights,
        hotkey_to_node_id,
        all_node_weights,
        scaled_environment_tournament_weight,
        scaled_environment_base_weight,
        environment_winner_hotkey,
    )

    total_undistributed = text_undistributed + image_undistributed + environment_undistributed
    logger.info(f"Total undistributed weight to add to burn: {total_undistributed:.10f}")

    return total_undistributed


async def get_node_weights_from_tournament_audit_data(
    substrate: SubstrateInterface,
    netuid: int,
    tournament_audit_data: TournamentAuditData,
) -> NodeWeightsResult:
    all_nodes: list[Node] = fetch_nodes.get_nodes_for_netuid(substrate, netuid)
    hotkey_to_node_id: dict[str, int] = {node.hotkey: node.node_id for node in all_nodes}

    all_node_ids: list[int] = [node.node_id for node in all_nodes]
    all_node_weights: list[float] = [0.0 for _ in all_nodes]

    logger.info("=== USING BURN DATA FROM AUDIT ===")

    for tournament_type in tournament_types():
        tournament_weight = get_audit_tournament_weight(tournament_audit_data, tournament_type)
        logger.info(f"{tournament_type.value.title()} tournament weight: {tournament_weight:.6f}")
    logger.info(f"Total burn weight: {tournament_audit_data.burn_weight:.6f}")

    # Check that base weights sum to 1.0
    base_weight_sum = sum(
        get_audit_tournament_weight(tournament_audit_data, tournament_type) for tournament_type in tournament_types()
    ) + tournament_audit_data.burn_weight
    logger.info(f"Base weights sum (text + image + environment + burn): {base_weight_sum:.10f}")
    logger.info(f"Base weights sum to 1.0? {abs(base_weight_sum - 1.0) < 0.0001}")

    scaled_weights = calculate_scaled_tournament_weights(tournament_audit_data)
    participants: list[str] = tournament_audit_data.participants

    logger.info(f"Number of participants: {len(participants)}")
    logger.info(f"Participation total weight: {scaled_weights.participation_total:.10f}")
    logger.info(f"Scale factor (1.0 - participation_total): {scaled_weights.scale_factor:.10f}")

    # Check that scaled weights + participation still sum to 1.0
    scaled_weight_sum = (
        sum(scaled_weights.tournament_weight(tournament_type) for tournament_type in tournament_types())
        + scaled_weights.burn_weight
        + scaled_weights.participation_total
    )
    logger.info(
        "Scaled weights sum (scaled_text + scaled_image + scaled_environment + scaled_burn + participation): "
        f"{scaled_weight_sum:.10f}"
    )
    logger.info(f"Scaled weights sum to 1.0? {abs(scaled_weight_sum - 1.0) < 0.0001}")

    text_tournament_weights, image_tournament_weights, environment_tournament_weights = get_tournament_weights_from_data(
        get_audit_tournament_data(tournament_audit_data, TournamentType.TEXT),
        get_audit_tournament_data(tournament_audit_data, TournamentType.IMAGE),
        get_audit_tournament_data(tournament_audit_data, TournamentType.ENVIRONMENT),
    )

    undistributed_weight = apply_tournament_weights(
        text_tournament_weights,
        image_tournament_weights,
        environment_tournament_weights,
        hotkey_to_node_id,
        all_node_weights,
        scaled_weights.tournament_weight(TournamentType.TEXT),
        scaled_weights.tournament_weight(TournamentType.IMAGE),
        scaled_weights.tournament_weight(TournamentType.ENVIRONMENT),
        scaled_weights.base_weight(TournamentType.TEXT),
        scaled_weights.base_weight(TournamentType.IMAGE),
        scaled_weights.base_weight(TournamentType.ENVIRONMENT),
        scaled_weights.winner_hotkey(TournamentType.TEXT),
        scaled_weights.winner_hotkey(TournamentType.IMAGE),
        scaled_weights.winner_hotkey(TournamentType.ENVIRONMENT),
    )

    # Check sum after tournament weights applied
    weight_sum_after_tournament = sum(all_node_weights)
    logger.info(f"Weight sum after tournament weights applied: {weight_sum_after_tournament:.10f}")

    for hotkey in participants:
        node_id = hotkey_to_node_id.get(hotkey)
        if node_id is not None:
            all_node_weights[node_id] += cts.TOURNAMENT_PARTICIPATION_WEIGHT

    # Check sum after participation weights added
    weight_sum_after_participation = sum(all_node_weights)
    logger.info(f"Weight sum after participation weights added: {weight_sum_after_participation:.10f}")

    # Add undistributed tournament weight to burn.
    # Undistributed weight comes from the gap between the boosted allocation and what's
    # actually distributed (winner gets boost, non-winners capped at base weight).
    # This ensures total weights sum to exactly 1.0.
    burn_node_id: int | None = hotkey_to_node_id.get(cts.EMISSION_BURN_HOTKEY)
    if burn_node_id is not None:
        all_node_weights[burn_node_id] = scaled_weights.burn_weight + undistributed_weight
        logger.info(
            f"Burn weight: base={scaled_weights.burn_weight:.10f} + undistributed={undistributed_weight:.10f} = "
            f"total={all_node_weights[burn_node_id]:.10f}"
        )

    # Final weight sum check
    final_weight_sum = sum(all_node_weights)
    logger.info("=== FINAL WEIGHT SUM CHECK ===")
    logger.info(f"Total weight sum (before normalization): {final_weight_sum:.10f}")
    logger.info("Expected: 1.0")
    logger.info(f"Difference from 1.0: {abs(final_weight_sum - 1.0):.10f}")
    logger.info(f"Weights sum to 1.0? {abs(final_weight_sum - 1.0) < 0.0001}")
    logger.info(f"Number of non zero node weights: {sum(1 for weight in all_node_weights if weight != 0)}")

    if abs(final_weight_sum - 1.0) >= 0.0001:
        logger.warning(f"⚠️  WARNING: Weights DO NOT sum to 1.0! Sum is {final_weight_sum:.10f}")
    else:
        logger.info("✅ Weights correctly sum to 1.0")

    return NodeWeightsResult(node_ids=all_node_ids, node_weights=all_node_weights)


async def build_tournament_audit_data(psql_db) -> TournamentAuditData:
    """
    Build TournamentAuditData with all necessary tournament information.

    This is the central function for gathering tournament data used by both
    the validator (for weight setting) and auditor (for verification).

    Args:
        psql_db: Database connection

    Returns:
        TournamentAuditData with all tournament information populated
    """
    tournament_audit_data = TournamentAuditData()

    for tournament_type in tournament_types():
        tournament: TournamentData = await get_latest_completed_tournament(psql_db, tournament_type)
        if not tournament:
            continue

        tournament_results: TournamentResults = await get_tournament_full_results(tournament.tournament_id, psql_db)
        set_audit_tournament_data(
            tournament_audit_data,
            tournament_type,
            TournamentResultsWithWinners(
                tournament_id=tournament_results.tournament_id,
                rounds=tournament_results.rounds,
                base_winner_hotkey=tournament.base_winner_hotkey,
                winner_hotkey=tournament.winner_hotkey,
            ),
        )

    # Fetch participants
    tournament_audit_data.participants = await get_active_tournament_participants(psql_db)

    # Fetch burn weights
    burn_data: TournamentBurnData = await get_tournament_burn_details(psql_db)
    for tournament_type in tournament_types():
        set_audit_tournament_weight(
            tournament_audit_data,
            tournament_type,
            getattr(burn_data, get_tournament_spec(tournament_type).audit_weight_field),
        )
    tournament_audit_data.burn_weight = burn_data.burn_weight

    # Fetch weekly participation data
    tournament_audit_data.weekly_participation = await get_weekly_task_participation_data(psql_db)

    return tournament_audit_data


async def set_weights(config: Config, all_node_ids: list[int], all_node_weights: list[float], validator_node_id: int) -> bool:
    try:
        success = await asyncio.to_thread(
            weights.set_node_weights,
            substrate=config.substrate,
            keypair=config.keypair,
            node_ids=all_node_ids,
            node_weights=all_node_weights,
            netuid=config.netuid,
            version_key=ccst.VERSION_KEY,
            validator_node_id=int(validator_node_id),
            wait_for_inclusion=False,
            wait_for_finalization=False,
            max_attempts=3,
        )
    except Exception as e:
        logger.error(f"Failed to set weights: {e}")
        return False

    if success:
        logger.info("Weights set successfully.")

        return True
    else:
        logger.error("Failed to set weights :(")
        return False


async def _get_and_set_weights(config: Config, validator_node_id: int) -> bool:
    # Build tournament audit data using the centralized function
    tournament_audit_data: TournamentAuditData = await build_tournament_audit_data(config.psql_db)

    result = await get_node_weights_from_tournament_audit_data(config.substrate, config.netuid, tournament_audit_data)
    all_node_ids = result.node_ids
    all_node_weights = result.node_weights
    logger.info("Weights calculated, about to set...")

    success = await set_weights(config, all_node_ids, all_node_weights, validator_node_id)
    if success:
        # Upload both task results and tournament data
        url = await _upload_results_to_s3(config, tournament_audit_data)
        logger.info(f"Uploaded the scores and tournament data to s3 for auditing - url: {url}")

    return success


async def _set_metagraph_weights(config: Config) -> None:
    nodes: list[Node] = fetch_nodes.get_nodes_for_netuid(config.substrate, config.netuid)
    node_ids = [node.node_id for node in nodes]
    node_weights = [node.incentive for node in nodes]
    validator_node_id = await get_vali_node_id(config.substrate, config.keypair.ss58_address)
    if validator_node_id is None:
        raise ValueError("Validator node id not found")

    await asyncio.to_thread(
        weights.set_node_weights,
        substrate=config.substrate,
        keypair=config.keypair,
        node_ids=node_ids,
        node_weights=node_weights,
        netuid=config.netuid,
        version_key=ccst.VERSION_KEY,
        validator_node_id=int(validator_node_id),
        wait_for_inclusion=False,
        wait_for_finalization=False,
        max_attempts=3,
    )


# To improve: use activity cutoff & The epoch length to set weights at the perfect times
async def set_weights_periodically(config: Config, just_once: bool = False) -> None:
    substrate = config.substrate
    substrate, uid = query_substrate(
        substrate,
        "SubtensorModule",
        "Uids",
        [config.netuid, config.keypair.ss58_address],
        return_value=True,
    )

    if uid is None:
        raise ValueError(f"Can't find hotkey {config.keypair.ss58_address} for our keypair on netuid: {config.netuid}.")

    consecutive_failures = 0
    while True:
        substrate, current_block = query_substrate(substrate, "System", "Number", [], return_value=True)
        substrate, last_updated_value = query_substrate(
            substrate, "SubtensorModule", "LastUpdate", [config.netuid], return_value=False
        )
        updated: int = current_block - last_updated_value[uid]
        substrate, weights_set_rate_limit = query_substrate(
            substrate, "SubtensorModule", "WeightsSetRateLimit", [config.netuid], return_value=True
        )
        logger.info(
            f"My Validator Node ID: {uid}. Last updated {updated} blocks ago. Weights set rate limit: {weights_set_rate_limit}."
        )

        if updated < weights_set_rate_limit:
            logger.info("Sleeping for a bit as we set recently...")
            await asyncio.sleep((weights_set_rate_limit - updated + 1) * 12)
            continue

        if os.getenv("ENV", "prod").lower() == "dev":
            success = await _get_and_set_weights(config, uid)
        else:
            try:
                success = await _get_and_set_weights(config, uid)
            except Exception as e:
                logger.error(f"Failed to set weights with error: {e}")
                logger.exception(e)
                success = False

        if success:
            consecutive_failures = 0
            logger.info("Successfully set weights! Sleeping for 25 blocks before next check...")
            if just_once:
                return
            await asyncio.sleep(12 * 25)
            continue

        consecutive_failures += 1
        if just_once:
            logger.info("Failed to set weights, will try again...")
            await asyncio.sleep(12 * 1)
        else:
            logger.info(f"Failed to set weights {consecutive_failures} times in a row - sleeping for a bit...")
            await asyncio.sleep(12 * 25)  # Try again in 25 blocks

        if consecutive_failures == 1 or updated < 3000:
            continue

        if just_once or config.set_metagraph_weights_with_high_updated_to_not_dereg:
            logger.warning("Setting metagraph weights as our updated value is getting too high!")
            if just_once:
                logger.warning("Please exit if you do not want to do this!!!")
                await asyncio.sleep(4)
            try:
                success = await _set_metagraph_weights(config)
            except Exception as e:
                logger.error(f"Failed to set metagraph weights: {e}")
                success = False

            if just_once:
                return

            if success:
                consecutive_failures = 0
                continue


async def main():
    config = load_config()
    await try_db_connections(config)
    await set_weights_periodically(config)


if __name__ == "__main__":
    asyncio.run(main())
