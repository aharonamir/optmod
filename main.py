import hashlib
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from optmod.config import load_config, Config
from optmod.registry import ModelRegistry
from optmod.features import FeatureExtractor
from optmod.routing import BaseRouter, build_router
from optmod.mutators import BaseContextMutator
from optmod.mutators.noop import NoopMutator
from optmod.mutators.thinking_mode import ThinkingModeMutator
from optmod.escalation import EscalationPolicy
from optmod.forwarder import ModelForwarder
from optmod.log import RoutingLog
from optmod.schemas import OpenAIChatRequest, RoutingContext, RoutingDecision, LogEntry
from optmod.stats import stats_router

_registry:   ModelRegistry       | None = None
_extractor:  FeatureExtractor    | None = None
_router:     BaseRouter          | None = None
_mutators:   dict[str, BaseContextMutator] = {}
_escalation: EscalationPolicy    | None = None
_forwarder:  ModelForwarder      | None = None
_log:        RoutingLog          | None = None
_config:     Config              | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _registry, _extractor, _router, _mutators
    global _escalation, _forwarder, _log, _config

    _config     = load_config("config.yaml")
    _registry   = ModelRegistry(_config.models, _config.primary_model)
    _extractor  = FeatureExtractor()
    _router     = build_router(_config.router, _config.dict())
    _mutators   = {
        "noop":          NoopMutator(),
        "thinking_mode": ThinkingModeMutator(),
    }
    _escalation = EscalationPolicy(max_escalations=_config.escalation.max_escalations)
    _forwarder  = ModelForwarder()
    _log        = RoutingLog(_config.log_path)

    yield

    await _forwarder.close()
    _log.flush()


app = FastAPI(title="optmod", lifespan=lifespan)
app.include_router(stats_router)
app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")


def _derive_session_id(req: OpenAIChatRequest) -> str:
    first = next(
        (m.content for m in req.messages if m.role == "user" and isinstance(m.content, str)),
        str(id(req)),
    )
    return hashlib.sha1(str(first).encode()).hexdigest()[:16]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _completion_to_sse(resp: dict) -> Generator[str, None, None]:
    """Wrap a non-streaming chat.completion response in SSE so streaming clients work."""
    msg_id  = resp.get("id", "")
    created = resp.get("created", 0)
    model   = resp.get("model", "")
    usage   = resp.get("usage", {})

    choices = resp.get("choices", [])
    if choices:
        choice       = choices[0]
        message      = choice.get("message", {})
        finish_reason = choice.get("finish_reason", "stop")
        tool_calls   = message.get("tool_calls")

        delta: dict = {"role": message.get("role", "assistant")}
        if tool_calls:
            delta["content"]    = None
            delta["tool_calls"] = [
                {"index": i, **{k: v for k, v in tc.items() if k != "index"}}
                for i, tc in enumerate(tool_calls)
            ]
        else:
            delta["content"] = message.get("content") or ""

        # Propagate non-standard fields (reasoning, etc.) so clients see them
        for key in ("reasoning", "reasoning_details"):
            if key in message:
                delta[key] = message[key]

        chunk = {"id": msg_id, "object": "chat.completion.chunk",
                 "created": created, "model": model,
                 "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
        yield f"data: {json.dumps(chunk)}\n\n"

        final = {"id": msg_id, "object": "chat.completion.chunk",
                 "created": created, "model": model,
                 "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                 "usage": usage}
        yield f"data: {json.dumps(final)}\n\n"

    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(raw: Request) -> JSONResponse:
    body     = await raw.json()
    req      = OpenAIChatRequest(**body)
    features = _extractor.extract(req)
    session  = _derive_session_id(req)
    t0       = time.perf_counter()

    ctx = RoutingContext(
        request=req,
        features=features,
        session_id=session,
        registry=_registry,
    )

    response:   dict             = {}
    error_type: str | None       = None
    decision:   RoutingDecision | None = None

    for attempt in range(_escalation.max_escalations + 1):
        ctx.attempt_number = attempt

        decision = _router.route(ctx)
        mutator  = _mutators.get(decision.mutator, _mutators["noop"])
        msgs     = mutator.mutate(req.messages, decision)

        response, error_type = await _forwarder.forward(decision.model, req, msgs)
        ctx.models_tried.append(decision.model.name)

        if error_type is None:
            break

        ctx.last_error_type = error_type
        if not _escalation.should_escalate(error_type, attempt, ctx):
            break

    latency_ms = (time.perf_counter() - t0) * 1000
    usage      = response.get("usage", {})

    _log.append(LogEntry(
        ts=                _now_iso(),
        session_id=        session,
        task_type=         features.task_type,
        difficulty=        features.difficulty,
        token_count=       features.token_count,
        has_tools=         features.has_tools,
        language=          features.language,
        router=            _router.name,
        decision_model=    decision.model.name if decision else "unknown",
        decision_reason=   decision.reason if decision else "",
        confidence=        decision.confidence if decision else 0.0,
        mutator=           decision.mutator if decision else "noop",
        models_tried=      ctx.models_tried,
        escalation_count=  len(ctx.models_tried) - 1,
        final_model=       ctx.models_tried[-1] if ctx.models_tried else "unknown",
        ok=                error_type is None,
        error_type=        error_type,
        latency_ms=        round(latency_ms, 2),
        prompt_tokens=     usage.get("prompt_tokens", 0),
        completion_tokens= usage.get("completion_tokens", 0),
    ))

    if error_type and not response:
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"optmod: all models failed ({error_type})", "type": "proxy_error"}},
        )
    if req.stream:
        return StreamingResponse(
            _completion_to_sse(response),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return JSONResponse(content=response)


@app.get("/optmod/status")
async def status() -> JSONResponse:
    return JSONResponse(content={
        "router":  _router.name,
        "primary": _registry.primary.name,
        "models": [
            {"name": m.name, "tier": m.tier_name, "cost_per_1k": m.cost_per_1k}
            for m in _registry.all()
        ],
    })


@app.post("/optmod/log/clear")
async def clear_log() -> JSONResponse:
    _log.clear()
    return JSONResponse(content={"ok": True})


@app.post("/optmod/restart")
async def restart_server() -> JSONResponse:
    # Touch main.py so uvicorn --reload picks up the change and restarts the worker.
    # Without --reload this is a no-op; restart the process manually in that case.
    from optmod.stats import reload_tier_map
    reload_tier_map()
    Path("main.py").touch()
    return JSONResponse(content={"ok": True, "message": "reloading…"})


@app.post("/optmod/router/{name}")
async def set_router(name: str) -> JSONResponse:
    global _router
    valid = {"passthrough", "rule_based", "decision_tree", "trouter"}
    if name not in valid:
        return JSONResponse(status_code=400, content={"error": f"unknown router: {name}"})
    _router = build_router(name, _config.dict())
    return JSONResponse(content={"router": name, "ok": True})
