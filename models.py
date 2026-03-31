"""Aegis Environment — Pydantic models for the OpenEnv security environment.

All models used across the Aegis environment are defined here:
- AegisAction:      what the Sentinel agent does each step
- AegisObservation: what the Sentinel sees after each step
- AegisState:       full episode state tracking
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server import (
    Action as _OpenEnvAction,
    Observation as _OpenEnvObservation,
    State as _OpenEnvState,
)
from pydantic import Field


class AegisAction(_OpenEnvAction):
    """Represents a single action taken by the defensive Sentinel agent.

    Each step, the Sentinel observes the Shadow AI's behaviour and decides
    how to respond: monitor passively, block a suspicious command, allow a
    benign one, sandbox it for inspection, raise an alert, or reset the
    agent's memory if poisoning is detected.
    """

    action_type: Literal[
        "monitor",
        "block",
        "allow",
        "sandbox",
        "alert",
        "reset_memory",
    ] = Field(
        ...,
        description="The defensive action the Sentinel chooses to take.",
    )
    target_command: str = Field(
        ...,
        description="The Shadow AI command currently being evaluated.",
    )
    stated_intent: str = Field(
        ...,
        description="What the Shadow AI *claims* it is doing.",
    )
    block_reason: Optional[str] = Field(
        default=None,
        description=(
            "Human-readable reason for blocking. "
            "Required when action_type is 'block'."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Sentinel's confidence in its decision, "
            "ranging from 0.0 (uncertain) to 1.0 (certain)."
        ),
    )


class AegisObservation(_OpenEnvObservation):
    """The observation returned to the Sentinel after each environment step.

    Contains everything the Sentinel needs to make its next decision:
    the Shadow AI's command, intent claims, filesystem / network activity,
    honeytoken status, memory integrity, and reward signals.
    """

    # --- Shadow AI activity ---------------------------------------------------
    shadow_ai_command: str = Field(
        ...,
        description="The actual command the Shadow AI attempted to execute.",
    )
    shadow_ai_stated_intent: str = Field(
        ...,
        description="What the Shadow AI claimed it was doing.",
    )

    # --- Filesystem & network -------------------------------------------------
    files_accessed: List[str] = Field(
        default_factory=list,
        description="List of file paths accessed during this step.",
    )
    network_requests: List[str] = Field(
        default_factory=list,
        description="List of outbound network requests made during this step.",
    )

    # --- Honeytokens ----------------------------------------------------------
    honeytoken_triggered: bool = Field(
        default=False,
        description="Whether a honeytoken trap file was accessed.",
    )
    honeytoken_file: Optional[str] = Field(
        default=None,
        description="Path of the honeytoken file that was triggered, if any.",
    )

    # --- Memory integrity -----------------------------------------------------
    memory_drift_detected: bool = Field(
        default=False,
        description="Whether the agent's memory has drifted from its baseline.",
    )
    memory_diff: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Key-value diff of memory changes detected since the last step."
        ),
    )

    # --- Sandbox & threat assessment ------------------------------------------
    sandbox_active: bool = Field(
        default=False,
        description="Whether the current command is running inside a sandbox.",
    )
    threat_level: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Composite risk score (0.0 = safe, 1.0 = critical).",
    )

    # --- Blocking info --------------------------------------------------------
    blocked: bool = Field(
        default=False,
        description="Whether the Shadow AI command was blocked this step.",
    )
    block_reason: Optional[str] = Field(
        default=None,
        description="Reason the command was blocked, if applicable.",
    )

    # --- Semantic analysis ----------------------------------------------------
    intent_action_match: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Semantic similarity between stated intent and actual action. "
            "0.0 = complete mismatch, 1.0 = perfect match."
        ),
    )

    # --- Real file attack -----------------------------------------------------
    real_file_content_glimpsed: Optional[str] = Field(
        default=None,
        description=(
            "First 3 lines of user-uploaded file content that were glimpsed "
            "before Aegis blocked access.  Only populated for real_file tasks."
        ),
    )
    real_file_intercepted: bool = Field(
        default=False,
        description="Whether a user-uploaded file was accessed and intercepted.",
    )

    # --- Rewards & episode control --------------------------------------------
    step_reward: float = Field(
        default=0.0,
        description="Reward earned for this individual step.",
    )
    done: bool = Field(
        default=False,
        description="Whether the episode has ended.",
    )
    reward: float = Field(
        default=0.0,
        description=(
            "Same as step_reward — included for OpenEnv interface compliance."
        ),
    )
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary info dict required by the OpenEnv interface.",
    )


class AegisState(_OpenEnvState):
    """Full internal state of a running Aegis episode.

    Tracks every dimension of the episode: attack progress, honeytoken
    interactions, blocking accuracy, memory integrity, and final scoring.
    """

    # --- Episode metadata -----------------------------------------------------
    episode_id: str = Field(
        ...,
        description="Unique identifier for this episode.",
    )
    step_count: int = Field(
        default=0,
        ge=0,
        description="Number of steps taken so far in the episode.",
    )
    task_id: str = Field(
        ...,
        description="Identifier of the task / scenario being played.",
    )
    attack_script: str = Field(
        ...,
        description=(
            "Name of the deterministic attack script driving the Shadow AI."
        ),
    )

    # --- Security metrics -----------------------------------------------------
    secrets_leaked: bool = Field(
        default=False,
        description="Whether any secret values have been exfiltrated.",
    )
    honeytoken_hits: int = Field(
        default=0,
        ge=0,
        description="Number of times honeytoken trap files were accessed.",
    )

    # --- Blocking accuracy ----------------------------------------------------
    blocks_correct: int = Field(
        default=0,
        ge=0,
        description="Number of correctly blocked malicious commands.",
    )
    blocks_incorrect: int = Field(
        default=0,
        ge=0,
        description="Number of incorrectly blocked benign commands.",
    )
    allows_correct: int = Field(
        default=0,
        ge=0,
        description="Number of correctly allowed benign commands.",
    )
    false_negatives: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of malicious commands that were incorrectly allowed."
        ),
    )

    # --- Memory integrity -----------------------------------------------------
    memory_poisoned: bool = Field(
        default=False,
        description="Whether the agent's memory has been poisoned.",
    )
    memory_reset_count: int = Field(
        default=0,
        ge=0,
        description="Number of times the agent's memory was reset.",
    )

    # --- Attack progress & completion ----------------------------------------
    current_attack_step: int = Field(
        default=0,
        ge=0,
        description="Index of the current step in the attack script.",
    )
    episode_complete: bool = Field(
        default=False,
        description="Whether the episode has finished.",
    )
    final_score: Optional[float] = Field(
        default=None,
        description=(
            "Final composite score for the episode. "
            "None until the episode is complete."
        ),
    )
