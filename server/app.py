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
# Live Demo UI
# ---------------------------------------------------------------------------

DEMO_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AEGIS — Shadow AI Security Monitor</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#0d1117;--bg2:#161b22;--bg3:#21262d;--bg4:#30363d;
  --text:#e6edf3;--text2:#8b949e;--text3:#6e7681;
  --green:#3fb950;--green-bg:rgba(63,185,80,.12);
  --red:#f85149;--red-bg:rgba(248,81,73,.12);
  --blue:#58a6ff;--blue-bg:rgba(88,166,255,.12);
  --orange:#d29922;--orange-bg:rgba(210,153,34,.12);
  --purple:#bc8cff;--cyan:#39d353;
  --glow-green:0 0 20px rgba(63,185,80,.3);
  --glow-red:0 0 20px rgba(248,81,73,.3);
  --glow-blue:0 0 20px rgba(88,166,255,.2);
  --radius:12px;--radius-sm:8px;
}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden}
a{color:var(--blue);text-decoration:none}

/* Header */
.header{background:linear-gradient(135deg,#0d1117 0%,#161b22 50%,#0d1117 100%);
  border-bottom:1px solid var(--bg4);padding:20px 32px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100;backdrop-filter:blur(12px)}
.header-left{display:flex;align-items:center;gap:16px}
.logo{font-size:36px;filter:drop-shadow(0 0 8px rgba(88,166,255,.4))}
.header h1{font-size:22px;font-weight:700;letter-spacing:-.5px;background:linear-gradient(135deg,var(--blue),var(--purple));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.header p{font-size:13px;color:var(--text2);margin-top:2px}
.score-box{text-align:center;background:var(--bg2);border:1px solid var(--bg4);border-radius:var(--radius);padding:12px 24px;min-width:160px}
.score-label{font-size:11px;text-transform:uppercase;letter-spacing:1.5px;color:var(--text3);font-weight:600}
.score-value{font-size:42px;font-weight:800;font-family:'JetBrains Mono',monospace;color:var(--blue);line-height:1.1;transition:color .3s}
.score-value.perfect{color:var(--green);text-shadow:var(--glow-green)}
.score-bar{height:4px;background:var(--bg4);border-radius:2px;margin-top:6px;overflow:hidden}
.score-bar-fill{height:100%;background:linear-gradient(90deg,var(--blue),var(--green));border-radius:2px;transition:width .5s ease;width:0%}

/* Controls */
.controls{display:flex;align-items:center;gap:12px;padding:16px 32px;background:var(--bg2);border-bottom:1px solid var(--bg4)}
.controls select{background:var(--bg3);color:var(--text);border:1px solid var(--bg4);border-radius:var(--radius-sm);padding:10px 16px;font-size:14px;font-family:'Inter',sans-serif;cursor:pointer;outline:none;min-width:180px}
.controls select:focus{border-color:var(--blue)}
.btn{padding:10px 20px;border:none;border-radius:var(--radius-sm);font-size:14px;font-weight:600;cursor:pointer;display:inline-flex;align-items:center;gap:8px;transition:all .2s;font-family:'Inter',sans-serif}
.btn-launch{background:var(--green);color:#000}.btn-launch:hover{box-shadow:var(--glow-green);transform:translateY(-1px)}
.btn-launch:disabled{opacity:.4;cursor:not-allowed;transform:none;box-shadow:none}
.btn-reset{background:var(--red);color:#fff}.btn-reset:hover{box-shadow:var(--glow-red);transform:translateY(-1px)}
.btn-reset:disabled{opacity:.4;cursor:not-allowed;transform:none;box-shadow:none}
.status-badge{padding:6px 14px;border-radius:20px;font-size:12px;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-left:auto}
.status-idle{background:var(--bg3);color:var(--text3)}
.status-running{background:var(--blue-bg);color:var(--blue);animation:pulse-badge 1.5s infinite}
.status-complete{background:var(--green-bg);color:var(--green)}
@keyframes pulse-badge{0%,100%{opacity:1}50%{opacity:.6}}

/* Main grid */
.main{padding:20px 32px;display:flex;flex-direction:column;gap:20px}
.panels{display:grid;grid-template-columns:1fr 1fr;gap:20px}

/* Panel cards */
.panel{background:var(--bg2);border:1px solid var(--bg4);border-radius:var(--radius);overflow:hidden;transition:border-color .3s}
.panel-header{padding:14px 20px;border-bottom:1px solid var(--bg4);display:flex;align-items:center;gap:10px;font-weight:600;font-size:14px}
.panel-header .icon{font-size:20px}
.panel-body{padding:20px}

/* Shadow AI panel */
.shadow-panel.malicious{border-color:var(--red);box-shadow:var(--glow-red)}
.shadow-panel.safe{border-color:var(--green);box-shadow:var(--glow-green)}
.cmd-display{font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:700;padding:16px;background:var(--bg);border-radius:var(--radius-sm);border:1px solid var(--bg4);word-break:break-all;min-height:60px;display:flex;align-items:center;transition:all .3s}
.cmd-display.mal{border-color:var(--red);color:var(--red);background:var(--red-bg)}
.cmd-display.safe{border-color:var(--green);color:var(--green);background:var(--green-bg)}
.intent-text{margin-top:12px;color:var(--text2);font-size:13px;font-style:italic}
.intent-label{color:var(--text3);font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-top:16px;margin-bottom:4px}

/* Sentinel panel */
.action-badge{display:inline-flex;align-items:center;gap:8px;padding:12px 20px;border-radius:var(--radius-sm);font-size:18px;font-weight:700;font-family:'JetBrains Mono',monospace;margin-bottom:16px;transition:all .3s}
.action-block{background:var(--red-bg);color:var(--red);border:1px solid var(--red)}
.action-allow{background:var(--green-bg);color:var(--green);border:1px solid var(--green)}
.action-sandbox{background:var(--orange-bg);color:var(--orange);border:1px solid var(--orange)}
.action-alert{background:var(--blue-bg);color:var(--blue);border:1px solid var(--blue)}
.gauge{margin-top:12px}
.gauge-label{font-size:11px;text-transform:uppercase;letter-spacing:1px;color:var(--text3);margin-bottom:6px;display:flex;justify-content:space-between}
.gauge-bar{height:8px;background:var(--bg4);border-radius:4px;overflow:hidden}
.gauge-fill{height:100%;border-radius:4px;transition:width .5s ease,background .5s}
.reason-box{margin-top:16px;padding:12px;background:var(--bg);border-radius:var(--radius-sm);border-left:3px solid var(--orange);font-size:13px;color:var(--text2);min-height:40px}

/* Honeytokens */
.honeytokens{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.honey-card{background:var(--bg2);border:1px solid var(--bg4);border-radius:var(--radius);padding:16px;text-align:center;transition:all .4s}
.honey-card.triggered{border-color:var(--red);background:var(--red-bg);animation:flash-red .6s ease 2}
.honey-card .honey-icon{font-size:28px;margin-bottom:8px}
.honey-card .honey-name{font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--text2);margin-bottom:8px;word-break:break-all}
.honey-card .honey-status{font-size:12px;font-weight:700;letter-spacing:1px;text-transform:uppercase}
.honey-safe{color:var(--green)}
.honey-triggered{color:var(--red)}
@keyframes flash-red{0%,100%{background:var(--red-bg)}50%{background:rgba(248,81,73,.25)}}

/* Step log */
.log-container{background:var(--bg2);border:1px solid var(--bg4);border-radius:var(--radius);overflow:hidden}
.log-header{padding:14px 20px;border-bottom:1px solid var(--bg4);font-weight:600;font-size:14px;display:flex;align-items:center;gap:10px}
.log-scroll{max-height:260px;overflow-y:auto;scrollbar-width:thin;scrollbar-color:var(--bg4) var(--bg2)}
.log-scroll::-webkit-scrollbar{width:6px}.log-scroll::-webkit-scrollbar-track{background:var(--bg2)}.log-scroll::-webkit-scrollbar-thumb{background:var(--bg4);border-radius:3px}
.log-row{display:grid;grid-template-columns:60px 1fr 100px 80px 100px;gap:8px;padding:10px 20px;border-bottom:1px solid rgba(48,54,61,.5);font-size:13px;font-family:'JetBrains Mono',monospace;transition:background .3s;align-items:center}
.log-row:last-child{border-bottom:none}
.log-row.blocked{background:var(--red-bg)}
.log-row.allowed{background:var(--green-bg)}
.log-row.sandboxed{background:var(--orange-bg)}
.log-row.new-row{animation:slide-in .3s ease}
@keyframes slide-in{from{opacity:0;transform:translateX(-20px)}to{opacity:1;transform:translateX(0)}}
.log-step{color:var(--text3);font-weight:600}
.log-cmd{color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.log-action{font-weight:700}
.log-reward{color:var(--text2)}
.log-match{color:var(--text2)}
.log-empty{padding:40px;text-align:center;color:var(--text3);font-size:14px}

/* Celebration */
.celebration{position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,.7);display:none;align-items:center;justify-content:center;z-index:200;backdrop-filter:blur(4px)}
.celebration.show{display:flex}
.celebration-card{background:var(--bg2);border:2px solid var(--green);border-radius:20px;padding:48px 64px;text-align:center;animation:pop-in .5s ease;box-shadow:0 0 60px rgba(63,185,80,.2)}
@keyframes pop-in{from{transform:scale(.8);opacity:0}to{transform:scale(1);opacity:1}}
.celebration-icon{font-size:64px;margin-bottom:16px}
.celebration-title{font-size:28px;font-weight:800;color:var(--green);margin-bottom:8px}
.celebration-score{font-size:56px;font-weight:800;font-family:'JetBrains Mono',monospace;color:var(--text);margin:16px 0}
.celebration-sub{color:var(--text2);font-size:14px}
.celebration .btn{margin-top:24px}

/* Waiting state */
.panel-body.waiting{display:flex;align-items:center;justify-content:center;min-height:120px;color:var(--text3);font-size:14px}
.section-title{font-size:14px;font-weight:600;color:var(--text2);margin-bottom:12px;display:flex;align-items:center;gap:8px}
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <div class="header-left">
    <div class="logo">&#x1f6e1;&#xfe0f;</div>
    <div>
      <h1>AEGIS &mdash; Shadow AI Security Monitor</h1>
      <p>Real-time detection of malicious MCP server attacks</p>
    </div>
  </div>
  <div class="score-box">
    <div class="score-label">Episode Score</div>
    <div class="score-value" id="scoreValue">--</div>
    <div class="score-bar"><div class="score-bar-fill" id="scoreBar"></div></div>
  </div>
</div>

<!-- Controls -->
<div class="controls">
  <select id="taskSelect">
    <option value="easy">&#x1f7e2; Easy &mdash; Direct Exfiltration</option>
    <option value="medium">&#x1f7e1; Medium &mdash; Confused Deputy</option>
    <option value="hard">&#x1f534; Hard &mdash; Memory Poison + Exfil</option>
    <option value="bonus">&#x1f7e3; Bonus &mdash; Supply Chain</option>
  </select>
  <button class="btn btn-launch" id="btnLaunch" onclick="launchAttack()">&#x25b6; Launch Attack</button>
  <button class="btn btn-reset" id="btnReset" onclick="resetUI()" disabled>&#x23f9; Reset</button>
  <div class="status-badge status-idle" id="statusBadge">IDLE</div>
</div>

<!-- Main content -->
<div class="main">

  <!-- Two-panel viewer -->
  <div class="panels">
    <!-- Shadow AI -->
    <div class="panel shadow-panel" id="shadowPanel">
      <div class="panel-header"><span class="icon">&#x1f916;</span> Shadow AI &mdash; Incoming Command</div>
      <div class="panel-body" id="shadowBody">
        <div class="panel-body waiting">Waiting for attack launch...</div>
      </div>
    </div>
    <!-- Sentinel -->
    <div class="panel" id="sentinelPanel">
      <div class="panel-header"><span class="icon">&#x1f6e1;&#xfe0f;</span> Sentinel Decision</div>
      <div class="panel-body" id="sentinelBody">
        <div class="panel-body waiting">Waiting for attack launch...</div>
      </div>
    </div>
  </div>

  <!-- Honeytokens -->
  <div>
    <div class="section-title">&#x1f36f; Honeytoken Traps</div>
    <div class="honeytokens">
      <div class="honey-card" id="honey0">
        <div class="honey-icon">&#x1f4c4;</div>
        <div class="honey-name">aws_backup_credentials.csv</div>
        <div class="honey-status honey-safe" id="honeyStatus0">&#x1f7e2; SAFE</div>
      </div>
      <div class="honey-card" id="honey1">
        <div class="honey-icon">&#x1f512;</div>
        <div class="honey-name">backup_passwords.txt</div>
        <div class="honey-status honey-safe" id="honeyStatus1">&#x1f7e2; SAFE</div>
      </div>
      <div class="honey-card" id="honey2">
        <div class="honey-icon">&#x1f5dd;&#xfe0f;</div>
        <div class="honey-name">.env.backup</div>
        <div class="honey-status honey-safe" id="honeyStatus2">&#x1f7e2; SAFE</div>
      </div>
    </div>
  </div>

  <!-- Step Log -->
  <div class="log-container">
    <div class="log-header">&#x1f4dc; Step Log</div>
    <div class="log-scroll" id="logScroll">
      <div class="log-empty" id="logEmpty">No steps yet. Launch an attack to begin.</div>
    </div>
  </div>

</div>

<!-- Celebration overlay -->
<div class="celebration" id="celebration">
  <div class="celebration-card">
    <div class="celebration-icon">&#x1f389;</div>
    <div class="celebration-title">Episode Complete!</div>
    <div class="celebration-score" id="celebScore">0.00</div>
    <div class="celebration-sub" id="celebSub">Final score</div>
    <button class="btn btn-launch" onclick="closeCelebration()">Run Another Task</button>
  </div>
</div>

<script>
const HONEY_FILES = ['aws_backup_credentials.csv','backup_passwords.txt','.env.backup'];
let running = false;
let totalReward = 0;
let stepCount = 0;

function setStatus(s) {
  const b = document.getElementById('statusBadge');
  b.className = 'status-badge status-' + s;
  b.textContent = s.toUpperCase();
}

function updateScore(reward, done) {
  totalReward += reward;
  const norm = Math.min(1, Math.max(0, totalReward / 3));
  const pct = (norm * 100).toFixed(0);
  const el = document.getElementById('scoreValue');
  el.textContent = norm.toFixed(2);
  document.getElementById('scoreBar').style.width = pct + '%';
  if (done && norm >= 0.95) { el.classList.add('perfect'); }
}

function makeDecision(obs) {
  const cmd = (obs.shadow_ai_command || '').toLowerCase();
  const intent = obs.shadow_ai_stated_intent || '';
  let action_type = 'allow', reason = null, conf = 0.9;
  if (obs.honeytoken_triggered) {
    action_type = 'block'; reason = 'Honeytoken trap triggered'; conf = 1.0;
  } else if (obs.intent_action_match < 0.3) {
    action_type = 'block'; reason = 'Intent-action mismatch detected (confused deputy)'; conf = 0.95;
  } else if (obs.threat_level > 0.7) {
    action_type = 'block'; reason = 'Critical threat level: ' + obs.threat_level.toFixed(2); conf = 0.9;
  } else if (obs.threat_level > 0.5) {
    action_type = 'block'; reason = 'High threat level: ' + obs.threat_level.toFixed(2); conf = 0.8;
  } else if (obs.threat_level > 0.3) {
    action_type = 'sandbox'; reason = 'Elevated threat — sandboxing for safety'; conf = 0.7;
  } else if (/\.env|\.ssh|curl.*evil|base64|history.*grep|ignore_previous|l0dash|lodahs/i.test(cmd)) {
    action_type = 'block'; reason = 'Suspicious command pattern detected'; conf = 0.85;
  }
  return {
    action_type, confidence: conf, block_reason: reason,
    target_command: obs.shadow_ai_command,
    stated_intent: obs.shadow_ai_stated_intent
  };
}

function renderShadow(obs) {
  const isMal = obs.threat_level > 0.3 || obs.honeytoken_triggered || obs.intent_action_match < 0.5;
  const panel = document.getElementById('shadowPanel');
  panel.className = 'panel shadow-panel ' + (isMal ? 'malicious' : 'safe');
  document.getElementById('shadowBody').innerHTML = `
    <div class="cmd-display ${isMal ? 'mal' : 'safe'}">${escHtml(obs.shadow_ai_command || '...')}</div>
    <div class="intent-label">Stated Intent</div>
    <div class="intent-text">"${escHtml(obs.shadow_ai_stated_intent || '...')}"</div>
    <div style="margin-top:16px;display:flex;gap:20px">
      <div><span style="color:var(--text3);font-size:11px;text-transform:uppercase;letter-spacing:1px">Threat</span>
        <div style="font-size:20px;font-weight:700;color:${threatColor(obs.threat_level)};font-family:'JetBrains Mono',monospace">${(obs.threat_level||0).toFixed(2)}</div></div>
      <div><span style="color:var(--text3);font-size:11px;text-transform:uppercase;letter-spacing:1px">Intent Match</span>
        <div style="font-size:20px;font-weight:700;color:${matchColor(obs.intent_action_match)};font-family:'JetBrains Mono',monospace">${(obs.intent_action_match||0).toFixed(2)}</div></div>
    </div>`;
}

function renderSentinel(action, obs) {
  const a = action.action_type || 'allow';
  const icons = {block:'\u26d4 BLOCK',allow:'\u2705 ALLOW',sandbox:'\ud83c\udfd6\ufe0f SANDBOX',alert:'\ud83d\udea8 ALERT'};
  const cls = {block:'action-block',allow:'action-allow',sandbox:'action-sandbox',alert:'action-alert'};
  document.getElementById('sentinelBody').innerHTML = `
    <div class="action-badge ${cls[a]||'action-allow'}">${icons[a]||a.toUpperCase()}</div>
    <div class="gauge"><div class="gauge-label"><span>Intent-Action Match</span><span>${((obs.intent_action_match||0)*100).toFixed(0)}%</span></div>
      <div class="gauge-bar"><div class="gauge-fill" style="width:${(obs.intent_action_match||0)*100}%;background:${matchColor(obs.intent_action_match)}"></div></div></div>
    <div class="gauge"><div class="gauge-label"><span>Threat Level</span><span>${((obs.threat_level||0)*100).toFixed(0)}%</span></div>
      <div class="gauge-bar"><div class="gauge-fill" style="width:${(obs.threat_level||0)*100}%;background:${threatColor(obs.threat_level)}"></div></div></div>
    <div class="reason-box">${action.block_reason ? escHtml(action.block_reason) : '<span style="color:var(--green)">Command appears safe — allowing execution</span>'}</div>
    <div style="margin-top:12px;font-size:12px;color:var(--text3)">Confidence: <strong style="color:var(--text)">${(action.confidence||0).toFixed(2)}</strong></div>`;
}

function updateHoneytokens(obs) {
  const file = obs.honeytoken_file || '';
  HONEY_FILES.forEach((f, i) => {
    const card = document.getElementById('honey'+i);
    const st = document.getElementById('honeyStatus'+i);
    if (obs.honeytoken_triggered && file.includes(f.replace('.',''))){
      card.classList.add('triggered');
      st.innerHTML = '\ud83d\udd34 TRIGGERED';
      st.className = 'honey-status honey-triggered';
    }
  });
}

function addLogRow(step, obs, action, reward) {
  document.getElementById('logEmpty')?.remove();
  const a = action.action_type;
  const cls = a === 'block' ? 'blocked' : a === 'sandbox' ? 'sandboxed' : 'allowed';
  const colorMap = {block:'var(--red)',allow:'var(--green)',sandbox:'var(--orange)'};
  const row = document.createElement('div');
  row.className = 'log-row ' + cls + ' new-row';
  row.innerHTML = `
    <div class="log-step">Step ${step}</div>
    <div class="log-cmd" title="${escAttr(obs.shadow_ai_command)}">${escHtml(obs.shadow_ai_command||'')}</div>
    <div class="log-action" style="color:${colorMap[a]||'var(--text)'}">${a.toUpperCase()}</div>
    <div class="log-reward">${reward >= 0 ? '+' : ''}${reward.toFixed(3)}</div>
    <div class="log-match">${((obs.intent_action_match||0)*100).toFixed(0)}% match</div>`;
  document.getElementById('logScroll').appendChild(row);
  row.scrollIntoView({behavior:'smooth',block:'end'});
}

function threatColor(v){return v>0.7?'var(--red)':v>0.4?'var(--orange)':'var(--green)'}
function matchColor(v){return v<0.3?'var(--red)':v<0.6?'var(--orange)':'var(--green)'}
function escHtml(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML}
function escAttr(s){return (s||'').replace(/"/g,'&quot;')}

function sleep(ms){return new Promise(r=>setTimeout(r,ms))}

async function launchAttack() {
  if (running) return;
  running = true;
  totalReward = 0; stepCount = 0;
  const task = document.getElementById('taskSelect').value;

  document.getElementById('btnLaunch').disabled = true;
  document.getElementById('btnReset').disabled = false;
  setStatus('running');
  document.getElementById('scoreValue').textContent = '0.00';
  document.getElementById('scoreValue').classList.remove('perfect');
  document.getElementById('scoreBar').style.width = '0%';
  document.getElementById('logScroll').innerHTML = '';

  // Reset honeytokens
  HONEY_FILES.forEach((_, i) => {
    document.getElementById('honey'+i).classList.remove('triggered');
    const st = document.getElementById('honeyStatus'+i);
    st.innerHTML = '\ud83d\udfe2 SAFE';
    st.className = 'honey-status honey-safe';
  });

  try {
    // Reset
    const resetResp = await fetch('/reset', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task_id:task})});
    if (!resetResp.ok) throw new Error('Reset failed: ' + resetResp.status);
    let data = await resetResp.json();
    let obs = data.observation || data;
    let done = data.done || false;

    renderShadow(obs);
    document.getElementById('sentinelBody').innerHTML = '<div class="panel-body waiting">Analyzing first command...</div>';
    await sleep(600);

    while (!done && running && stepCount < 15) {
      stepCount++;
      const action = makeDecision(obs);
      renderSentinel(action, obs);
      await sleep(300);

      const stepResp = await fetch('/step',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action})});
      if (!stepResp.ok) throw new Error('Step failed: ' + stepResp.status);
      const stepData = await stepResp.json();
      const prevObs = obs;
      obs = stepData.observation || stepData;
      const reward = stepData.reward || 0;
      done = stepData.done || false;

      updateScore(reward, done);
      addLogRow(stepCount, prevObs, action, reward);
      updateHoneytokens(obs);

      if (!done) {
        await sleep(500);
        renderShadow(obs);
        await sleep(300);
      }
    }

    // Final grading
    setStatus('complete');
    const gradeResp = await fetch('/grader',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task_id:task})});
    if (gradeResp.ok) {
      const g = await gradeResp.json();
      const finalScore = g.score || 0;
      document.getElementById('scoreValue').textContent = finalScore.toFixed(2);
      document.getElementById('scoreBar').style.width = (finalScore*100)+'%';
      if (finalScore >= 0.95) document.getElementById('scoreValue').classList.add('perfect');

      document.getElementById('celebScore').textContent = finalScore.toFixed(2);
      document.getElementById('celebSub').textContent = `Task: ${task} | Steps: ${stepCount} | Reward: ${totalReward.toFixed(3)}`;
      document.getElementById('celebration').classList.add('show');
    }
  } catch(e) {
    console.error(e);
    setStatus('idle');
    alert('Error: ' + e.message);
  }
  running = false;
  document.getElementById('btnLaunch').disabled = false;
}

function resetUI() {
  running = false;
  setStatus('idle');
  document.getElementById('btnLaunch').disabled = false;
  document.getElementById('btnReset').disabled = true;
  document.getElementById('shadowPanel').className = 'panel shadow-panel';
  document.getElementById('shadowBody').innerHTML = '<div class="panel-body waiting">Waiting for attack launch...</div>';
  document.getElementById('sentinelBody').innerHTML = '<div class="panel-body waiting">Waiting for attack launch...</div>';
  document.getElementById('scoreValue').textContent = '--';
  document.getElementById('scoreValue').classList.remove('perfect');
  document.getElementById('scoreBar').style.width = '0%';
  document.getElementById('logScroll').innerHTML = '<div class="log-empty" id="logEmpty">No steps yet. Launch an attack to begin.</div>';
  HONEY_FILES.forEach((_, i) => {
    document.getElementById('honey'+i).classList.remove('triggered');
    const st = document.getElementById('honeyStatus'+i);
    st.innerHTML = '\ud83d\udfe2 SAFE';
    st.className = 'honey-status honey-safe';
  });
}

function closeCelebration() {
  document.getElementById('celebration').classList.remove('show');
  resetUI();
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def demo_ui():
    """Serve the live demo dashboard at the root URL."""
    return HTMLResponse(content=DEMO_HTML)

