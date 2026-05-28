# optmod — Claude Code Build Instructions

## What You Are Building

`optmod` is a **local OpenAI-compatible proxy server** written in Python 3.11+.

Hermes Agent (and any other OpenAI-compatible agent framework) points its
`base_url` at `http://localhost:8765/v1`. Every LLM call goes through optmod,
which classifies the request, selects the optimal model, optionally mutates
context, forwards to the real model, handles retry/escalation internally, and
returns one clean response. The calling agent never knows a proxy is in the middle.

## Read Order

Read these spec files in order before writing any code:

1. `SPEC/00_overview.md`       — architecture, constraints, non-goals
2. `SPEC/01_schemas.md`        — all Pydantic models and dataclasses
3. `SPEC/02_registry.md`       — ModelConfig and ModelRegistry
4. `SPEC/03_features.md`       — FeatureExtractor (the classifier)
5. `SPEC/04_routers.md`        — BaseRouter, PassthroughRouter, RuleBasedRouter, DecisionTreeRouter
6. `SPEC/05_mutators.md`       — BaseContextMutator, NoopMutator, ThinkingModeMutator
7. `SPEC/06_escalation.md`     — EscalationPolicy
8. `SPEC/07_forwarder.md`      — ModelForwarder (httpx async)
9. `SPEC/08_main.md`           — FastAPI app, lifespan, /v1/chat/completions endpoint
10. `SPEC/09_stats.md`         — /api/stats endpoint, JSONL reader
11. `SPEC/10_tests.md`         — full test plan with exact test cases
12. `SPEC/11_config.md`        — config.yaml schema and example
13. `SPEC/12_ui.md`            — dashboard HTML (copy as-is from assets/)

## Build Rules (Non-Negotiable)

- **Python 3.11+** only. Use `X | Y` union types, not `Optional[X]`.
- **Never raise** inside a router's `route()` method. Catch all exceptions, return passthrough.
- **Never modify messages in-place** inside a mutator. Always return a new list.
- **Feature extraction must be <1ms**. No ML, no file I/O, no network calls inside `FeatureExtractor.extract()`.
- **All regex patterns compiled at module import time**, not inside functions.
- **One httpx.AsyncClient per base_url**, reused across requests. Never create a new client per request.
- **JSONL log**: one JSON line per request, appended atomically. Never rewrite the file.
- **`pyproject.toml`** is the only dependency file. No `requirements.txt`.
- **Tests use `respx`** to mock httpx calls. No real network calls in tests.
- **`asyncio_mode = "auto"`** in pytest config.

## Directory Structure to Create

```
optmod/
├── main.py
├── config.py
├── config.yaml
├── registry.py
├── features.py
├── forwarder.py
├── escalation.py
├── log.py
├── schemas.py
├── routing/
│   ├── __init__.py      ← BaseRouter ABC
│   ├── context.py       ← RoutingContext dataclass
│   ├── passthrough.py
│   ├── rule_based.py
│   └── decision_tree.py
├── mutators/
│   ├── __init__.py      ← BaseContextMutator ABC
│   ├── noop.py
│   └── thinking_mode.py
├── stats.py             ← /api/stats FastAPI router
├── ui/
│   └── index.html       ← copy from assets/dashboard.html
├── tests/
│   ├── conftest.py
│   ├── test_features.py
│   ├── test_routers.py
│   ├── test_escalation.py
│   └── test_proxy_e2e.py
└── pyproject.toml
```

## Build Order

Build in this exact sequence. After each step, run the relevant tests before continuing.

1. `pyproject.toml` → `schemas.py` → `config.py` → `config.yaml`
2. `registry.py` → `features.py` → `log.py`
3. `routing/__init__.py` → `routing/context.py` → `routing/passthrough.py`
4. `mutators/__init__.py` → `mutators/noop.py` → `mutators/thinking_mode.py`
5. `escalation.py` → `forwarder.py`
6. `routing/rule_based.py` → `routing/decision_tree.py`
7. `main.py` → `stats.py`
8. `ui/index.html` (copy from assets/dashboard.html)
9. All tests
10. Smoke test: `uvicorn main:app --port 8765` → `curl http://localhost:8765/optmod/status`

## Do Not

- Do not add any dependency not in `pyproject.toml`
- Do not create a `requirements.txt`
- Do not use `Optional[X]` — use `X | None`
- Do not call any external service during tests
- Do not use `threading` — everything is async
- Do not add logging beyond what the spec defines
- Do not create any file not listed in the directory structure above
