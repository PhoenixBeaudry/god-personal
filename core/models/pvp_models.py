"""
Pydantic models for PvP (Player-vs-Player) environment evaluation.
Defines input configuration and output result contracts.
"""

from enum import Enum

from pydantic import BaseModel
from pydantic import Field

from core.constants import EnvironmentName


class GameOutcome(str, Enum):
    """Outcome of a single game from a player's perspective."""

    WIN = "win"
    LOSS = "loss"
    DRAW = "draw"


class GameScoringContext(BaseModel):
    """Extracted game metadata needed to compute win/loss/draw from returns."""

    returns: list[float] = Field(description="Terminal returns from state.returns(), one per player")
    player_id: int = Field(description="Index of the player whose outcome we're computing")
    is_zero_sum: bool = Field(description="Whether the game is zero-sum")
    min_utility: float = Field(description="Minimum possible return value")
    max_utility: float = Field(description="Maximum possible return value")


class GameInstance(BaseModel):
    """Configuration for a single game to be played."""

    game_name: str = Field(description="OpenSpiel game identifier (e.g. 'liars_dice')")
    game_params: dict[str, int] = Field(description="Parameters passed to pyspiel.load_game()")
    model_a_player_id: int = Field(description="Player index assigned to model A (0 or 1)")
    seed: int = Field(description="Random seed for this game instance")
    is_zero_sum: bool = Field(description="Whether the game is zero-sum")
    min_utility: float = Field(description="Game's minimum utility value")
    max_utility: float = Field(description="Game's maximum utility value")


class PvPModelSpec(BaseModel):
    """Specification for a model participating in PvP evaluation."""

    model_config = {"protected_namespaces": ()}

    repo: str = Field(description="HuggingFace model repository (e.g. 'org/model-name')")
    original_model: str = Field(
        description="Base model repository, used for LoRA detection and merging"
    )
    gpu_id: int | None = Field(default=None, ge=0, description="GPU device ID. Defaults to 0 for model_a, 1 for model_b")
    port: int | None = Field(default=None, gt=0, description="SGLang server port. Defaults to 30000 for model_a, 30001 for model_b")


class PvPMatchupConfig(BaseModel):
    """Configuration for a single environment matchup."""

    num_games: int = Field(
        gt=0,
        description="Number of seeds to play. Each seed is played twice (position swap), so total games = num_games * 2",
    )


class PvPEvalConfig(BaseModel):
    """Top-level input configuration for a PvP evaluation run.

    Loaded from PVP_EVAL_CONFIG env var or /config/pvp_eval.json.
    """

    model_config = {"protected_namespaces": ()}

    model_a: PvPModelSpec
    model_b: PvPModelSpec
    matchups: dict[EnvironmentName, PvPMatchupConfig] = Field(
        description="Map of environment name to matchup configuration"
    )
    seed: int = Field(default=42, description="Base seed for deterministic game generation")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class PvPEnvironmentResult(BaseModel):
    """Win/loss/draw result for a single environment."""

    model_config = {"protected_namespaces": ()}

    model_a_wins: int = 0
    model_b_wins: int = 0
    draws: int = 0
    total_games: int = 0


class ChatRole(str, Enum):
    """OpenAI-compatible message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """A single message in an OpenAI-compatible conversation."""

    role: ChatRole
    content: str


class ChatCompletionConfig(BaseModel):
    """Configuration for calling an OpenAI-compatible chat endpoint."""

    model_config = {"protected_namespaces": ()}

    model: str = Field(description="Model name as registered in the inference server")
    base_url: str = Field(description="OpenAI-compatible API base (e.g. http://localhost:30000/v1)")
    api_key: str = Field(default="dummy", description="API key (SGLang ignores but SDK requires)")
    temperature: float | None = Field(default=None, description="Sampling temperature, None uses server default")
    seed: int | None = Field(default=None, description="Random seed for reproducibility")
    max_tokens: int = Field(default=20, gt=0, description="Max tokens to generate per response")
    max_retries: int = Field(default=10, ge=0, description="Retry attempts on transient failures")
    read_timeout: float = Field(default=30.0, gt=0, description="HTTP read timeout in seconds")


class ChatResult(BaseModel):
    """Result from an LLM chat completion during PvP game play."""

    content: str | None = None
    usage: dict[str, int] | None = None




class PvPEvalMetadata(BaseModel):
    """Metadata about the evaluation run."""

    seed: int
    temperature: float
    position_swapped: bool = True
    wall_time_seconds: float = 0.0


class PvPEvalResults(BaseModel):
    """Complete output of a PvP evaluation run.

    Written to /app/pvp_results.json.
    """

    model_config = {"protected_namespaces": ()}

    model_a: str
    model_b: str
    results: dict[EnvironmentName, PvPEnvironmentResult]
    metadata: PvPEvalMetadata
