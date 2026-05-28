# optmod

A local OpenAI-compatible routing proxy that selects the optimal LLM for each request — transparently, within a single HTTP call.

Point any OpenAI-compatible agent (Hermes, LangChain, etc.) at `http://localhost:8765/v1` and optmod classifies each request, picks the right model, escalates on failure, and returns one clean response. The calling agent never knows a proxy is in the middle.

## How it works

```
Agent  →  POST /v1/chat/completions  →  optmod
                                           │
                                           ├─ extract features  (<1ms, regex only)
                                           ├─ route → pick model + strategy
                                           ├─ mutate context (e.g. /think prefix)
                                           ├─ forward via OpenRouter (httpx async)
                                           │     └─ on error: escalate up one tier
                                           └─ log to JSONL → return response
```

## Model pool

All models are served through [OpenRouter](https://openrouter.ai). Set `OPENROUTER_API_KEY` in `.env`.

| Tier | Model | Cost / 1k tokens | Use |
|---|---|---|---|
| fast (0) | `openai/gpt-oss-120b:free` | $0.000 | Simple tasks, extract, summarize |
| reasoning (1) | `arcee-ai/trinity-large-thinking` | $0.00022 | Multi-step reasoning, tool calls, Hebrew |
| oracle (2) | `deepseek/deepseek-v4-flash` | $0.00014 | Hard tasks, long context, quality ceiling |

Escalation path: `fast → reasoning → oracle → 502`

## Quickstart

```bash
# 1. Install
uv venv && uv pip install -e '.[dev]'

# 2. Add your OpenRouter key
echo "OPENROUTER_API_KEY=sk-or-..." >> .env

# 3. Run
uv run uvicorn main:app --host 0.0.0.0 --port 8765 --reload

# 4. Verify
curl http://localhost:8765/optmod/status

# 5. Open the dashboard
open http://localhost:8765/ui

# 6. Send a request
curl -X POST http://localhost:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"optmod","messages":[{"role":"user","content":"why does quicksort fail on sorted input?"}]}'

# 7. Swap router at runtime (no restart needed)
curl -X POST http://localhost:8765/optmod/router/rule_based
curl -X POST http://localhost:8765/optmod/router/trouter
curl -X POST http://localhost:8765/optmod/router/passthrough
```

### TRouter — neural net routing (optional)

```bash
# Install PyTorch + sentence-transformers extras
uv pip install -e '.[dev,trouter]'

# Activate (loads trouter_weights.pt + all-MiniLM-L6-v2 encoder at startup)
curl -X POST http://localhost:8765/optmod/router/trouter
```

## Routers

| Name | Description |
|---|---|
| `passthrough` | Always routes to the primary model; no classification |
| `rule_based` | 9 deterministic rules on task type, difficulty, language, token count |
| `decision_tree` | scikit-learn tree trained on WildClawBench; falls back to `rule_based` if `routing_policy.pkl` is absent |
| `trouter` | Neural network router: sentence-BERT encodes the query, a lightweight MLP picks the model weighting quality and cost |

All routers swap at runtime — no restart required.

## Dashboard

Open `http://localhost:8765/ui` for a live dashboard with 5s auto-refresh:

- **Stat cards** — total requests, success rate, escalation rate, error rate
- **Model distribution** — tier-colored usage bars, per-model cost estimate for the window, savings vs. always routing to oracle
- **Task types** — distribution of task classifications
- **Latency histogram** — bucketed with P50/P90/P99/avg
- **Escalation flow** — which models are being escalated to and why
- **Recent requests table** — per-request model (with full name tooltip), models-tried chain, cost, latency, status, routing reason (truncated; hover for full text), confidence. New rows slide in animated; a pause/resume button freezes the table without stopping stat updates.

## API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | Main proxy — OpenAI wire format |
| `GET` | `/optmod/status` | Router name, primary model, full model list with costs |
| `POST` | `/optmod/router/{name}` | Swap router: `passthrough`, `rule_based`, `decision_tree`, `trouter` |
| `GET` | `/api/stats?range=N` | Aggregated stats from JSONL log (1h, 6h, 24h, last-N, all) |
| `GET` | `/api/stats/live` | Lightweight live counts for polling |
| `GET` | `/ui` | Live routing dashboard |

## Project layout

Flat layout — source lives at the project root, importable as `optmod.*`.

```
main.py               FastAPI app, lifespan, proxy endpoint
schemas.py            Pydantic + dataclass types
config.py             Config loader (config.yaml)
config.yaml           Model pool + router config (all via OpenRouter)
registry.py           ModelConfig + ModelRegistry
features.py           FeatureExtractor (<1ms regex classifier)
forwarder.py          Async httpx forwarder, one client per base_url
escalation.py         EscalationPolicy
log.py                Append-only JSONL log
stats.py              /api/stats aggregation (model_tokens, cost tracking)
routing/
  __init__.py         BaseRouter ABC + build_router() factory
  passthrough.py      PassthroughRouter
  rule_based.py       RuleBasedRouter (9 rules)
  decision_tree.py    DecisionTreeRouter (scikit-learn)
  trouter_router.py   TRouterRouter (neural net, sentence-BERT encoder)
  train_trouter.py    TRouter training code + route() function
  trouter_weights.pt  Trained checkpoint
mutators/             BaseContextMutator, NoopMutator, ThinkingModeMutator
tests/
  test_features.py    Unit tests — FeatureExtractor
  test_routers.py     Unit tests — all routers
  test_escalation.py  Unit tests — EscalationPolicy
  test_proxy_e2e.py   Mock e2e tests (respx, no network)
  test_live_e2e.py    Live e2e tests (real OpenRouter calls, skipped in CI)
ui/index.html         Self-contained stats dashboard
.env                  API keys — git-ignored
```

## Running tests

```bash
# Mock tests only (no network, fast)
uv run pytest tests/test_proxy_e2e.py tests/test_routers.py \
              tests/test_features.py tests/test_escalation.py -v

# All tests including live e2e (requires OPENROUTER_API_KEY in .env)
uv run pytest tests/ -v -s
```

## Use with Hermes

Add to `~/.hermes/config.yaml`:

```yaml
provider: custom
model: optmod-router
base_url: http://localhost:8765/v1
api_key: optmod
```

## What's not built yet (Phase 2)

- `routing_policy.pkl` — produced by running WildClawBench; needed to enable the `decision_tree` router
- Hermes plugin (thin wrapper calling `/optmod/*` endpoints)
- Feedback loop CLI (retrains TRouter / decision tree from `routing.log.jsonl`)
