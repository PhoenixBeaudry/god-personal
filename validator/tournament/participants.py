"""Tournament participant repository and eligibility helpers."""

import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
import httpx

from core.git import build_authenticated_git_url
from core.git import sanitize_git_text
from core.logging import get_logger
from core.models.tournament_models import GitHubOwnerRepo
from core.models.tournament_models import RespondingNode
from core.models.tournament_models import RoundType
from core.models.tournament_models import TournamentData
from core.models.tournament_models import TournamentParticipant
from core.models.tournament_models import TournamentRoundData
from core.models.tournament_models import TournamentType
from validator.db.database import PSQLDB
from validator.db.sql.tournaments import get_latest_completed_tournament
from validator.db.sql.tournaments import get_tournament_pairs
from validator.db.sql.tournaments import get_tournament_participant
from validator.shared.config import Config
from validator.shared.constants import DEFAULT_PARTICIPANT_COMMIT
from validator.shared.constants import DEFAULT_PARTICIPANT_REPO
from validator.shared.constants import EMISSION_BURN_HOTKEY
from validator.tournament import constants as t_cst


logger = get_logger(__name__)


async def _get_final_round_participants(completed_round: TournamentRoundData, psql_db: PSQLDB) -> tuple[str, str]:
    if completed_round.round_type != RoundType.KNOCKOUT:
        raise ValueError(f"Expected a knockout round, got {completed_round.round_type}")

    pairs = await get_tournament_pairs(completed_round.round_id, psql_db)
    if not pairs:
        raise ValueError(f"No pairs found for final round {completed_round.round_id}")

    pair = pairs[0]
    return pair.hotkey1, pair.hotkey2


async def get_challenger_participant_for_retained_boss(
    tournament: TournamentData,
    completed_round: TournamentRoundData,
    winners: list[str],
    psql_db: PSQLDB,
) -> TournamentParticipant | None:
    challenger_hotkey = next((hotkey for hotkey in winners if hotkey != EMISSION_BURN_HOTKEY), None)
    if not challenger_hotkey and completed_round.round_type == RoundType.KNOCKOUT:
        try:
            participant1, participant2 = await _get_final_round_participants(completed_round, psql_db)
            challenger_hotkey = participant2 if participant1 == EMISSION_BURN_HOTKEY else participant1
        except Exception as exc:
            logger.warning(f"Could not determine retained-boss challenger from final round participants: {exc}")

    if not challenger_hotkey:
        logger.warning("Could not determine retained-boss challenger; diff report will not include challenger repo")
        return None

    challenger = await get_tournament_participant(tournament.tournament_id, challenger_hotkey, psql_db)
    if not challenger or not challenger.training_repo:
        logger.warning(f"Challenger {challenger_hotkey} has no training repository in DB")
        return None
    return challenger


async def get_latest_commit_hash_from_github(repo_url: str) -> str | None:
    """Fetch the latest commit hash from a GitHub repository."""
    # Extract owner/repo from URL: https://github.com/owner/repo
    repo_path = repo_url.split("github.com/")[1].replace(".git", "")
    api_url = f"https://api.github.com/repos/{repo_path}/commits/main"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("sha", "")
                else:
                    logger.error(f"Failed to fetch commit hash from {repo_url}: HTTP {response.status}")
                    return None
    except Exception as e:
        logger.error(f"Error fetching commit hash from {repo_url}: {e}")
        return None


async def get_base_contestant(psql_db: PSQLDB, tournament_type: TournamentType, config: Config) -> TournamentParticipant | None:
    """Get a BASE contestant as the last tournament winner."""

    latest_winner = await get_latest_tournament_winner_participant(psql_db, tournament_type, config)
    if latest_winner:
        logger.info(f"Using latest tournament winner as BASE: {latest_winner.hotkey}")

        if latest_winner.backup_repo:
            logger.info(f"Previous winner has backup repo: {latest_winner.backup_repo}")
            commit_hash = await get_latest_commit_hash_from_github(latest_winner.backup_repo)
            if not commit_hash:
                logger.warning(f"Could not fetch commit hash for {latest_winner.backup_repo}, setting to None")

            return TournamentParticipant(
                tournament_id="",
                hotkey=EMISSION_BURN_HOTKEY,
                training_repo=latest_winner.backup_repo,
                training_commit_hash=commit_hash,
            )
        else:
            logger.warning("Could not determine tournament ID for uploaded repo, falling back to original training_repo")
            # Fallback to original training_repo if we can't determine the uploaded repo
            return TournamentParticipant(
                tournament_id="",
                hotkey=EMISSION_BURN_HOTKEY,
                training_repo=latest_winner.training_repo,
                training_commit_hash=latest_winner.training_commit_hash,
            )

    logger.info(
        f"No previous tournament winner found for type {tournament_type.value}, "
        f"using hardcoded base winner: {EMISSION_BURN_HOTKEY}"
    )

    hardcoded_participant = TournamentParticipant(
        tournament_id="",
        hotkey=EMISSION_BURN_HOTKEY,
        training_repo=DEFAULT_PARTICIPANT_REPO,
        training_commit_hash=DEFAULT_PARTICIPANT_COMMIT,
    )

    return hardcoded_participant


async def get_latest_tournament_winner_participant(
    psql_db: PSQLDB, tournament_type: TournamentType, config: Config
) -> TournamentParticipant | None:
    """Get the winner participant from the latest completed tournament of the given type."""
    latest_tournament = await get_latest_completed_tournament(psql_db, tournament_type)
    if not latest_tournament:
        logger.warning(f"No completed tournaments found for type {tournament_type.value}")
        return None

    winner_hotkey = latest_tournament.winner_hotkey
    if not winner_hotkey:
        logger.warning(f"Tournament {latest_tournament.tournament_id} is completed but has no winner_hotkey stored")
        return None

    logger.info(f"Found latest tournament winner: {winner_hotkey}")
    winner_participant = await get_tournament_participant(latest_tournament.tournament_id, winner_hotkey, psql_db)

    # If we can't find the winner's participant record, check if they were the defending champion
    # who entered as EMISSION_BURN_HOTKEY
    if not winner_participant:
        logger.warning(
            f"Could not find participant record for winner {winner_hotkey} in tournament {latest_tournament.tournament_id}"
        )

        # If the winner was the base_winner (defending champion), try to get their record from EMISSION_BURN_HOTKEY
        if winner_hotkey == latest_tournament.base_winner_hotkey:
            logger.info(f"Winner {winner_hotkey} was the defending champion, checking EMISSION_BURN_HOTKEY participant record")
            emission_participant = await get_tournament_participant(
                latest_tournament.tournament_id, EMISSION_BURN_HOTKEY, psql_db
            )
            if emission_participant:
                # Use the EMISSION_BURN_HOTKEY participant's training info but with the actual winner's hotkey
                emission_participant.hotkey = winner_hotkey
                return emission_participant

        # If still no participant record found, return None to use default
        logger.warning(f"No participant record found for winner {winner_hotkey}, will use default")
        return None

    # If the participant is EMISSION_BURN_HOTKEY but we have a real winner, use the real winner's hotkey
    if winner_participant.hotkey == EMISSION_BURN_HOTKEY and latest_tournament.base_winner_hotkey:
        winner_participant.hotkey = latest_tournament.base_winner_hotkey

    return winner_participant





async def validate_repo_obfuscation(
    repo_url: str, commit_hash: str | None = None, github_token: str | None = None
) -> bool:
    """
    Validate that a repository is not obfuscated using the obfuscation detection.

    Args:
        repo_url: The repository URL to validate
        commit_hash: Optional commit hash to validate instead of the default branch

    Returns:
        bool: True if repo is not obfuscated, False if obfuscated
    """
    try:
        clone_url = build_authenticated_git_url(repo_url, github_token)
        cmd = [t_cst.OBFUSCATION_DETECTION_PATH, "--repo", clone_url]
        if commit_hash:
            cmd += ["--commit", commit_hash]

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        logger.info(f"Obfuscation detection output: {proc.stdout}")

        if proc.returncode == 0:
            logger.info(f"Repo {repo_url} is not obfuscated (exit code 0)")
            return True
        else:
            logger.warning(f"Repo {repo_url} is obfuscated (exit code {proc.returncode})")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"Obfuscation detection timed out for repo {repo_url}")
        return False
    except Exception as e:
        logger.error(f"Obfuscation detection failed for repo {repo_url}: {str(e)}")
        return False


async def validate_repo_license(repo_url: str, github_token: str | None = None) -> bool:
    """
    Validate that a repository has verbatim LICENSE and NOTICE files matching the current repository.

    Args:
        repo_url: The repository URL to validate

    Returns:
        bool: True if repo has valid LICENSE and NOTICE files, False otherwise
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Cloning repository {repo_url} for license validation")
            clone_url = build_authenticated_git_url(repo_url, github_token)

            clone_proc = subprocess.run(
                ["git", "clone", clone_url, temp_dir],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if clone_proc.returncode != 0:
                sanitized_stderr = sanitize_git_text(clone_proc.stderr, github_token)
                logger.error(f"Failed to clone repository {repo_url}: {sanitized_stderr}")
                return False

            temp_path = Path(temp_dir)
            current_file_path = Path(__file__).resolve()
            repo_root = current_file_path.parent.parent.parent

            expected_license_path = repo_root / "LICENSE.md"
            if not expected_license_path.exists():
                expected_license_path = repo_root / "LICENSE"
                if not expected_license_path.exists():
                    logger.warning(
                        f"Expected LICENSE file not found in validator repository at "
                        f"{repo_root / 'LICENSE.md'} or {repo_root / 'LICENSE'}. "
                        f"Skipping license validation for {repo_url}"
                    )
                    return True

            expected_notice_path = None
            for notice_filename in ["NOTICE", "NOTICE.txt", "notice.txt", "Notice.txt", "notice", "Notice"]:
                potential_path = repo_root / notice_filename
                if potential_path.exists():
                    expected_notice_path = potential_path
                    break

            if not expected_notice_path:
                logger.warning(
                    f"Expected NOTICE file not found in validator repository at {repo_root} "
                    f"(checked NOTICE, NOTICE.txt, notice.txt, Notice.txt, notice, Notice). "
                    f"Skipping license validation for {repo_url}"
                )
                return True

            license_file_path = None
            for license_filename in ["LICENSE.md", "LICENSE", "license.md", "license", "License.md", "License"]:
                potential_path = temp_path / license_filename
                if potential_path.exists():
                    license_file_path = potential_path
                    break

            if not license_file_path:
                logger.warning(
                    f"License file not found in repository {repo_url} "
                    f"(checked LICENSE.md, LICENSE, license.md, license, License.md, License)"
                )
                return False

            license_content = license_file_path.read_text(encoding="utf-8")
            expected_license = expected_license_path.read_text(encoding="utf-8")

            expected_license_normalized = "\n".join(line.rstrip() for line in expected_license.splitlines())
            actual_license_normalized = "\n".join(line.rstrip() for line in license_content.splitlines())

            if expected_license_normalized != actual_license_normalized:
                logger.warning(f"LICENSE file content does not match verbatim for repository {repo_url}")
                return False

            notice_file_path = None
            for notice_filename in ["NOTICE", "NOTICE.txt", "notice.txt", "Notice.txt", "notice", "Notice"]:
                potential_path = temp_path / notice_filename
                if potential_path.exists():
                    notice_file_path = potential_path
                    break

            if not notice_file_path:
                logger.warning(
                    f"NOTICE file not found in repository {repo_url} "
                    f"(checked NOTICE, NOTICE.txt, notice.txt, Notice.txt, notice, Notice)"
                )
                return False

            notice_content = notice_file_path.read_text(encoding="utf-8")
            expected_notice = expected_notice_path.read_text(encoding="utf-8")

            expected_notice_normalized = "\n".join(line.rstrip() for line in expected_notice.splitlines())
            actual_notice_normalized = "\n".join(line.rstrip() for line in notice_content.splitlines())

            if expected_notice_normalized != actual_notice_normalized:
                logger.warning(f"NOTICE file content does not match verbatim for repository {repo_url}")
                return False

            logger.info(f"Repository {repo_url} passed license validation")
            return True

    except subprocess.TimeoutExpired:
        logger.error(f"Repository validation timed out for repo {repo_url}")
        return False
    except Exception as e:
        logger.error(f"Repository validation failed for repo {repo_url}: {str(e)}")
        return False


def parse_github_owner_repo(repo_url: str) -> GitHubOwnerRepo | None:
    path = urlparse(repo_url).path.strip("/")
    parts = path.split("/")
    if len(parts) >= 2 and parts[0] and parts[1]:
        owner, repo_name = parts[0], parts[1].removesuffix(".git")
        return GitHubOwnerRepo(owner=owner, repo=repo_name)
    return None


async def validate_github_tokens(nodes: list[RespondingNode]) -> None:
    async with httpx.AsyncClient(timeout=10) as client:
        for node in nodes:
            token = node.training_repo_response.github_token
            if not token:
                continue

            parsed = parse_github_owner_repo(node.training_repo_response.github_repo)
            if not parsed:
                node.training_repo_response.github_token = None
                continue

            try:
                resp = await client.get(
                    f"https://api.github.com/repos/{parsed.owner}/{parsed.repo}",
                    headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
                )
                if resp.status_code != 200:
                    logger.warning(
                        f"Token for {node.node.hotkey} does not grant access to "
                        f"{parsed.owner}/{parsed.repo} (HTTP {resp.status_code}) — ignoring token"
                    )
                    node.training_repo_response.github_token = None
            except Exception as e:
                logger.warning(f"Token validation failed for {node.node.hotkey}: {e} — ignoring token")
                node.training_repo_response.github_token = None


def deduplicate_by_github_account(nodes: list[RespondingNode]) -> list[RespondingNode]:
    by_account: defaultdict[str, list[RespondingNode]] = defaultdict(list)
    no_account: list[RespondingNode] = []

    for node in nodes:
        parsed = parse_github_owner_repo(node.training_repo_response.github_repo)
        if parsed:
            by_account[parsed.owner.lower()].append(node)
        else:
            no_account.append(node)

    kept: list[RespondingNode] = list(no_account)
    for account, group in by_account.items():
        if len(group) == 1:
            kept.append(group[0])
            continue

        with_token = [n for n in group if n.training_repo_response.github_token]
        without_token = [n for n in group if not n.training_repo_response.github_token]

        if with_token:
            winner = with_token[0]
            rejected = with_token[1:] + without_token
        else:
            winner = without_token[0]
            rejected = without_token[1:]

        kept.append(winner)
        for r in rejected:
            logger.warning(
                f"Rejecting {r.node.hotkey} — duplicate GitHub account '{account}' "
                f"(kept {winner.node.hotkey})"
            )

    return kept
