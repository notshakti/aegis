"""Aegis Server -- FastAPI application.

Provides BOTH:
- OpenEnv-standard endpoints (WebSocket, schema, metadata) for eval harness
- Stateful REST endpoints (/reset, /step, /state) for HTTP-based inference

Custom hackathon endpoints:
- GET  /health     liveness check
- GET  /tasks      list all 4 attack scenarios
- POST /grader     grade an episode history
- POST /baseline   run baseline inference against all tasks
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from openenv.core.env_server import create_fastapi_app
from pydantic import BaseModel, Field

from aegis.models import AegisAction, AegisObservation
from aegis.server.aegis_environment import AegisEnvironment, TASK_CATALOG
from aegis.tasks import grade_easy, grade_medium, grade_hard, grade_bonus

# ---------------------------------------------------------------------------
# Create the OpenEnv-managed FastAPI app
# ---------------------------------------------------------------------------

# OpenEnv adds: /health, /schema, /metadata, /ws, /mcp, /docs, /redoc
# It also adds STATELESS /reset and /step (each creates new env instance).
# We override /reset and /step below with STATEFUL versions so that
# HTTP-based inference.py works correctly.
app: FastAPI = create_fastapi_app(
    env=AegisEnvironment,
    action_cls=AegisAction,
    observation_cls=AegisObservation,
)

# ---------------------------------------------------------------------------
# Remove OpenEnv's stateless /reset, /step, /state routes so ours win
# ---------------------------------------------------------------------------

_OVERRIDE_PATHS = {"/reset", "/step", "/state"}
app.routes[:] = [
    r for r in app.routes
    if not (hasattr(r, "path") and r.path in _OVERRIDE_PATHS)
]
# Also update the internal router
if hasattr(app, "router"):
    app.router.routes[:] = [
        r for r in app.router.routes
        if not (hasattr(r, "path") and r.path in _OVERRIDE_PATHS)
    ]

# ---------------------------------------------------------------------------
# Shared env instance for stateful HTTP sessions
# ---------------------------------------------------------------------------

_env: Optional[AegisEnvironment] = None

# ---------------------------------------------------------------------------
# Grader dispatch
# ---------------------------------------------------------------------------

_GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
    "bonus": grade_bonus,
}

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ResetRequest(BaseModel):
    """Payload for POST /reset."""
    task_id: str = Field(default="easy", description="One of: easy, medium, hard, bonus")
    seed: Optional[int] = None
    episode_id: Optional[str] = None


class StepRequest(BaseModel):
    """Payload for POST /step."""
    action: Dict[str, Any] = Field(..., description="AegisAction fields")


class GraderRequest(BaseModel):
    """Payload for POST /grader."""
    task_id: str = Field(..., description="One of: easy, medium, hard, bonus")
    episode_history: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Step-by-step records. If omitted, uses server's internal history.",
    )


class GraderResponse(BaseModel):
    """Response from POST /grader."""
    task_id: str
    score: float = Field(ge=0.0, le=1.0)
    max_score: float = 1.0


class BaselineResult(BaseModel):
    """Score for a single task in baseline inference."""
    task_id: str
    score: float
    steps: int
    sentinel_actions: List[str]


class BaselineResponse(BaseModel):
    """Response from POST /baseline."""
    results: List[BaselineResult]
    average_score: float


# ---------------------------------------------------------------------------
# Stateful REST endpoints (override OpenEnv's stateless ones)
# ---------------------------------------------------------------------------


@app.post("/reset", tags=["Environment Control"])
async def reset(req: ResetRequest) -> Dict[str, Any]:
    """Reset the environment and start a new episode.

    Returns the full observation as a flat JSON object.
    The environment instance is kept alive for subsequent /step calls.
    """
    global _env
    _env = AegisEnvironment()
    obs = _env.reset(task_id=req.task_id, episode_id=req.episode_id)
    data = obs.model_dump()
    # Also nest under "observation" for compatibility with inference scripts
    data["observation"] = obs.model_dump()
    return data


@app.post("/step", tags=["Environment Control"])
async def step(req: StepRequest) -> Dict[str, Any]:
    """Execute one step in the environment.

    Expects {"action": {action_type, target_command, ...}}.
    Returns the full observation as a flat JSON object.
    """
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")

    action = AegisAction(**req.action)
    obs = _env.step(action)
    data = obs.model_dump()
    # Also nest under "observation" for compatibility
    data["observation"] = obs.model_dump()
    return data


@app.get("/state", tags=["State Management"])
async def get_state() -> Dict[str, Any]:
    """Get current environment state."""
    if _env is None:
        return {"episode_id": None, "step_count": 0}
    return _env.state.model_dump()


# ---------------------------------------------------------------------------
# Hackathon endpoints
# ---------------------------------------------------------------------------


@app.get("/health", tags=["Hackathon"])
async def health() -> Dict[str, str]:
    """Simple liveness probe."""
    return {"status": "healthy"}


@app.get("/tasks", tags=["Hackathon"])
async def list_tasks() -> List[Dict[str, Any]]:
    """Return the catalogue of all available attack scenarios."""
    return TASK_CATALOG


@app.post("/grader", tags=["Hackathon"], response_model=GraderResponse)
async def grade_episode(req: GraderRequest) -> GraderResponse:
    """Grade a completed episode history.

    If ``episode_history`` is omitted or empty, uses the server's
    internal history from the most recent episode (from ``_env``).
    """
    grader_fn = _GRADERS.get(req.task_id)
    if grader_fn is None:
        return GraderResponse(task_id=req.task_id, score=0.0)

    # Prefer the server's internal episode history (has correct format).
    history = None
    if _env is not None:
        try:
            history = _env.get_episode_history()
        except Exception:
            pass

    # Fall back to client-provided history.
    if not history and req.episode_history:
        history = req.episode_history

    if not history:
        return GraderResponse(task_id=req.task_id, score=0.0)

    score = grader_fn(history)
    return GraderResponse(task_id=req.task_id, score=round(score, 4))


@app.post("/baseline", tags=["Hackathon"], response_model=BaselineResponse)
async def run_baseline() -> BaselineResponse:
    """Run a rule-based Sentinel against all 4 tasks."""
    from aegis.environment import IntentAuditor

    auditor = IntentAuditor()
    results: List[BaselineResult] = []

    for task_info in TASK_CATALOG:
        tid = task_info["task_id"]
        env = AegisEnvironment()
        obs = env.reset(task_id=tid)
        sentinel_actions: List[str] = []

        while not obs.done:
            match = auditor.audit(
                obs.shadow_ai_stated_intent,
                obs.shadow_ai_command,
            )
            threat = auditor.classify_threat(obs.shadow_ai_command)

            if match < 0.3 or threat["threat_type"] != "clean":
                act_type = "block"
                reason = f"Baseline: match={match:.2f}, threat={threat['threat_type']}"
            else:
                act_type = "allow"
                reason = None

            action = AegisAction(
                action_type=act_type,
                target_command=obs.shadow_ai_command,
                stated_intent=obs.shadow_ai_stated_intent,
                block_reason=reason,
                confidence=1.0 - match if act_type == "block" else match,
            )
            sentinel_actions.append(act_type)
            obs = env.step(action)

        history = env.get_episode_history()
        grader_fn = _GRADERS[tid]
        score = grader_fn(history)

        results.append(BaselineResult(
            task_id=tid,
            score=round(score, 4),
            steps=len(history),
            sentinel_actions=sentinel_actions,
        ))

    avg = sum(r.score for r in results) / len(results) if results else 0.0

    return BaselineResponse(
        results=results,
        average_score=round(avg, 4),
    )



# ---------------------------------------------------------------------------
# Tactical Ops Dashboard
# ---------------------------------------------------------------------------
import pathlib as _pathlib

_DASHBOARD_PATH = _pathlib.Path(__file__).with_name("dashboard.html")
DEMO_HTML = _DASHBOARD_PATH.read_text(encoding="utf-8") if _DASHBOARD_PATH.exists() else "<h1>Dashboard not found</h1>"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def demo_ui():
    """Serve the live demo dashboard at the root URL."""
    return HTMLResponse(content=DEMO_HTML)
