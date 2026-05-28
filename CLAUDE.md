# optmod — Claude Code Context

This file is read automatically at the start of every Claude Code session.

## What This Project Is

`optmod` is a **local OpenAI-compatible routing proxy** (FastAPI, Python 3.11+).
Any agent pointing at `http://localhost:8765/v1` gets transparent multi-model routing:
classify → route → mutate context → forward → escalate on error → log → respond.

The calling agent never knows a proxy is in the middle.

## Running the Server

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8765 --reload
```

API keys are loaded from `.env` at startup (via `api_key_env` in `config.yaml`).
**Important:** uvicorn `--reload` only watches `.py` files. After changing `config.yaml`,
do a hard restart (Ctrl+C → re-run).

Dashboard: `http://localhost:8765/ui`

## Current Model Pool (all via OpenRouter)

| Tier | Model | Cost/1k | Notes |
|------|-------|---------|-------|
| fast (0) | `openai/gpt-oss-120b:free` | $0.000 | Free tier |
| reasoning (1) | `arcee-ai/trinity-large-thinking` | $0.00022 | Thinking model |
| oracle (2) | `deepseek/deepseek-v4-flash` | $0.00014 | Primary / quality ceiling |

Escalation path: `fast → reasoning → oracle → 502`

## Active Routers (swap with `POST /optmod/router/{name}`)

| Name | Description |
|------|-------------|
| `rule_based` | **Default.** 9 deterministic rules (task type, difficulty, language, tokens) |
| `trouter` | Neural net: sentence-BERT (all-MiniLM-L6-v2) → MLP → model index. Loads `routing/trouter_weights.pt` |
| `passthrough` | Always uses oracle (deepseek-v4-flash). Useful for baseline |
| `decision_tree` | scikit-learn tree; falls back to `rule_based` if `routing_policy.pkl` absent |

## Key Files

```
main.py                   FastAPI app, lifespan, all /optmod/* endpoints
config.yaml               Model pool + active router (edit here to change models)
config.py                 Config loader — reads config.yaml + resolves env vars
registry.py               ModelConfig, ModelRegistry
features.py               FeatureExtractor — <1ms regex classifier, no I/O
forwarder.py              httpx async forwarder, one AsyncClient per base_url
escalation.py             EscalationPolicy
log.py                    RoutingLog — append-only JSONL, threading.Lock
stats.py                  /api/stats aggregation; _TIER_MAP loaded from config.yaml at import
routing/__init__.py        BaseRouter ABC + build_router() factory
routing/rule_based.py      9-rule deterministic router
routing/trouter_router.py  TRouterRouter — neural net routing
routing/train_trouter.py   TRouter training code + route() inference function
mutators/thinking_mode.py  Prepends /think to trigger CoT in thinking models
ui/index.html             Self-contained dashboard (no build step)
tests/test_live_e2e.py    Live tests hitting real OpenRouter — skipped if no API key
```

## Architecture Rules (Non-Negotiable)

- **Never raise** inside `router.route()` — catch everything, return passthrough
- **Never mutate messages in-place** — mutators return a new list
- **Feature extraction must be <1ms** — no ML, no I/O inside `FeatureExtractor.extract()`
- **All regex compiled at module import time**, not inside functions
- **One `httpx.AsyncClient` per `base_url`** — never create per request
- **JSONL log**: one line per request, appended atomically — never rewrite
- **`pyproject.toml`** is the only dependency file — no `requirements.txt`
- **`asyncio_mode = "auto"`** in pytest config

## Testing

```bash
# Mock tests (fast, no network)
uv run pytest tests/test_proxy_e2e.py tests/test_routers.py \
              tests/test_features.py tests/test_escalation.py -v

# Live e2e (requires OPENROUTER_API_KEY in .env, hits real providers)
uv run pytest tests/test_live_e2e.py -v -s
```

Mock tests use `respx` to intercept `httpx` calls. All three models route
through OpenRouter (`https://openrouter.ai/api/v1`), so mock tests intercept
that single base URL.

## Dashboard Endpoints

| Method | Path | Notes |
|--------|------|-------|
| `POST` | `/optmod/log/clear` | Truncates `routing.log.jsonl` |
| `POST` | `/optmod/restart` | Re-reads `_TIER_MAP` + touches `main.py` for `--reload` |
| `POST` | `/optmod/router/{name}` | Hot-swap router |
| `GET` | `/optmod/status` | Current router + model list with costs |
| `GET` | `/api/stats?range=N` | Aggregated stats (1h, 6h, 24h, last-N, all) |
| `GET` | `/ui` | Live dashboard |

## Common Gotchas

- **`config.yaml` not picked up after edit** — uvicorn `--reload` ignores `.yaml`.
  Hard-restart the server, or call `POST /optmod/restart` (re-reads `_TIER_MAP` in-process).
- **All tier badges show ORACLE** — `_TIER_MAP` in `stats.py` not loaded with current config.
  Hard-restart the server.
- **Thinking models need large `max_tokens`** — `arcee-ai/trinity-large-thinking` and
  `deepseek/deepseek-v4-flash` consume token budget on internal reasoning before
  producing the answer. Use `max_tokens >= 1500` in tests.
- **Old log entries from a previous model pool** appear in the dashboard.
  Use the **CLEAR LOG** button or `POST /optmod/log/clear`.
- **TRouter optional deps** — `routing/trouter_router.py` imports `torch` and
  `sentence_transformers`. Install with `uv pip install -e '.[trouter]'`.
  If absent, TRouter falls back to passthrough silently.

## What's Phase 2 (Not Yet Built)

- `routing_policy.pkl` — WildClawBench-trained scikit-learn tree; needed to enable `decision_tree` router without falling back to `rule_based`
- Hermes plugin — thin wrapper calling `/optmod/*` control endpoints
- Feedback loop CLI — retrains TRouter / decision tree from `routing.log.jsonl`
