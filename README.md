# optmod

A local OpenAI-compatible routing proxy that selects the optimal LLM for each request — transparently, within a single HTTP call.

Point any OpenAI-compatible agent (Hermes, LangChain, etc.) at `http://localhost:8765/v1` and optmod classifies each request, picks the right model, escalates on failure, and returns one clean response. The calling agent never knows a proxy is in the middle.

## How it works

```
Agent  →  POST /v1/chat/completions  →  optmod
                                           │
                                           ├─ extract features  (<1ms, regex only)
                                           ├─ route → pick model
                                           ├─ mutate context (e.g. /think prefix)
                                           ├─ forward to real model (httpx async)
                                           │     └─ on error: escalate up one tier
                                           └─ log to JSONL → return response
```

## Model tiers

| Tier | Model | Provider | Use |
|---|---|---|---|
| fast (0) | qwen2.5:7b | Ollama (local) | Simple tasks, extract, summarize |
| reasoning (1) | qwen3:8b | Ollama (local) | Multi-step reasoning, tool calls, Hebrew |
| oracle (2) | deepseek-v4 | DeepSeek API | Hard tasks, long context, quality ceiling |

Escalation path: `fast → reasoning → oracle → 502`

## Quickstart

```bash
# Prerequisites
ollama serve &
ollama pull qwen2.5:7b
ollama pull qwen3:8b
export DEEPSEEK_API_KEY=your_key_here

# Install and run
uv venv && uv pip install -e '.[dev]'
source .venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8765 --reload
```

```bash
# Check status
curl http://localhost:8765/optmod/status

# Open dashboard
open http://localhost:8765/ui

# Send a request
curl -X POST http://localhost:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"optmod","messages":[{"role":"user","content":"why does quicksort fail on sorted input?"}]}'

# Swap router at runtime
curl -X POST http://localhost:8765/optmod/router/passthrough
curl -X POST http://localhost:8765/optmod/router/rule_based

# Stats
curl "http://localhost:8765/api/stats?range=1h"
```

## Use with Hermes

Add to `~/.hermes/config.yaml`:

```yaml
provider: custom
model: optmod-router
base_url: http://localhost:8765/v1
api_key: optmod
```

## API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | Main proxy — OpenAI wire format |
| `GET` | `/optmod/status` | Router name, primary model, model list |
| `POST` | `/optmod/router/{name}` | Swap router: `passthrough`, `rule_based`, `decision_tree` |
| `GET` | `/api/stats?range=N` | Aggregated stats from JSONL log |
| `GET` | `/api/stats/live` | Lightweight live counts |
| `GET` | `/ui` | Dashboard |

## Routers

- **passthrough** — always uses the primary model (deepseek-v4), no routing logic
- **rule_based** — 9 deterministic rules (difficulty, task type, language, token count)
- **decision_tree** — scikit-learn tree trained on WildClawBench (Phase 2); falls back to rule_based if `routing_policy.pkl` is absent

## Project layout

Flat layout — source lives at the project root, importable as `optmod.*`.

```
main.py        FastAPI app, lifespan, proxy endpoint
schemas.py     Pydantic + dataclass types
config.py      Config loader (config.yaml)
registry.py    ModelConfig + ModelRegistry
features.py    FeatureExtractor (<1ms regex classifier)
forwarder.py   Async httpx forwarder, one client per base_url
escalation.py  EscalationPolicy
log.py         Append-only JSONL log
stats.py       /api/stats aggregation
routing/       BaseRouter, PassthroughRouter, RuleBasedRouter, DecisionTreeRouter
mutators/      BaseContextMutator, NoopMutator, ThinkingModeMutator
tests/         31 tests (features, routers, escalation, e2e with respx)
config.yaml    Model pool + router config
ui/index.html  Self-contained stats dashboard
.venv/         uv virtual environment (not committed)
```

## Running tests

```bash
uv run pytest tests/ -v
```

## What's not built yet (Phase 2)

- `routing_policy.pkl` — produced by running WildClawBench
- Hermes plugin (thin wrapper calling `/optmod/*` endpoints)
- GLM-4.7-Flash model tier
- Feedback loop calibration CLI (retrains tree from `routing.log.jsonl`)
