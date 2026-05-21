# 08 — FastAPI App (main.py)

## Module-level singletons

Declare all as `None` at module level, initialized in `lifespan`:

```python
_registry:   ModelRegistry       | None = None
_extractor:  FeatureExtractor    | None = None
_router:     BaseRouter          | None = None
_mutators:   dict[str, BaseContextMutator] = {}
_escalation: EscalationPolicy    | None = None
_forwarder:  ModelForwarder      | None = None
_log:        RoutingLog          | None = None
_config:     Config              | None = None
```

## Lifespan

```python
from contextlib import asynccontextmanager

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

    yield   # ← app runs here

    await _forwarder.close()
    _log.flush()
```

## App creation

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="optmod", lifespan=lifespan)
app.include_router(stats_router)           # from stats.py
app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")
```

## Main endpoint: POST /v1/chat/completions

```python
import time, hashlib

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

    response:   dict       = {}
    error_type: str | None = None
    decision:   RoutingDecision | None = None

    for attempt in range(_escalation.max_escalations + 1):
        ctx.attempt_number = attempt

        decision  = _router.route(ctx)
        mutator   = _mutators.get(decision.mutator, _mutators["noop"])
        msgs      = mutator.mutate(req.messages, decision)

        response, error_type = await _forwarder.forward(decision.model, req, msgs)
        ctx.models_tried.append(decision.model.name)

        if error_type is None:
            break

        ctx.last_error_type = error_type
        if not _escalation.should_escalate(error_type, attempt, ctx):
            break

    latency_ms = (time.perf_counter() - t0) * 1000

    # Extract token counts from response
    usage = response.get("usage", {})

    _log.append(LogEntry(
        ts=               _now_iso(),
        session_id=       session,
        task_type=        features.task_type,
        difficulty=       features.difficulty,
        token_count=      features.token_count,
        has_tools=        features.has_tools,
        language=         features.language,
        router=           _router.name,
        decision_model=   decision.model.name if decision else "unknown",
        decision_reason=  decision.reason if decision else "",
        confidence=       decision.confidence if decision else 0.0,
        mutator=          decision.mutator if decision else "noop",
        models_tried=     ctx.models_tried,
        escalation_count= len(ctx.models_tried) - 1,
        final_model=      ctx.models_tried[-1] if ctx.models_tried else "unknown",
        ok=               error_type is None,
        error_type=       error_type,
        latency_ms=       round(latency_ms, 2),
        prompt_tokens=    usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
    ))

    if error_type and not response:
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"optmod: all models failed ({error_type})", "type": "proxy_error"}},
        )
    return JSONResponse(content=response)
```

## Helper: session ID

```python
def _derive_session_id(req: OpenAIChatRequest) -> str:
    """Hash first user message for session continuity."""
    first = next(
        (m.content for m in req.messages if m.role == "user" and isinstance(m.content, str)),
        str(id(req)),
    )
    return hashlib.sha1(str(first).encode()).hexdigest()[:16]

def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
```

## Control endpoints

```python
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

@app.post("/optmod/router/{name}")
async def set_router(name: str) -> JSONResponse:
    global _router
    valid = {"passthrough", "rule_based", "decision_tree"}
    if name not in valid:
        return JSONResponse(status_code=400, content={"error": f"unknown router: {name}"})
    _router = build_router(name, _config.dict())
    return JSONResponse(content={"router": name, "ok": True})
```
