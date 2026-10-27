# optmod

A local OpenAI-compatible routing proxy that selects the optimal LLM for each request — transparently, within a single HTTP call.

<img width="1834" height="852" alt="image" src="https://github.com/user-attachments/assets/1ee67d2c-4a20-43ab-b3cc-4c13ad62f555" />

Point any OpenAI-compatible agent (Hermes, LangChain, etc.) at `http://localhost:8765/v1` and optmod classifies each request, picks the right model, escalates on failure, and returns one clean response. The calling agent never knows a proxy is in the middle.

## How it works

```
Agent  →  POST /v1/chat/completions  →  optmod
                                           │
                                           ├─ extract features  (<1ms, regex only)
                                           ├─ route → pick model + strategy
                                           ├─ mutate context (e.g. /think prefix)
                                           ├─ forward via OpenRouter (httpx async)
                                           │     └─ on error: escalate to the best-scoring healthy model that fits
                                           └─ log to JSONL → return response
```

## Model pool

The current pool is multi-provider: DeepSeek direct API for the flagship models, [OpenRouter](https://openrouter.ai) for everything else. Set both `DEEPSEEK_API_KEY` and `OPENROUTER_API_KEY` in `.env`; each model declares which env var to use via `api_key_env` in `config.yaml`.

| Tier | Model | Provider | Cost / 1k input |
|---|---|---|---|
| fast (0) | `openai/gpt-oss-120b` | OpenRouter | $0.000037 |
| fast (0) | `nvidia/nemotron-3-super-120b-a12b:free` | OpenRouter | $0.0 |
| fast (0) | `openai/gpt-4o-mini` | OpenRouter | $0.00015 |
| reasoning (1) | `deepseek-v4-flash` | DeepSeek | $0.00014 |
| reasoning (1) | `google/gemini-3.1-flash-lite` | OpenRouter | $0.0001 |
| reasoning (1) | `tencent/hy3-preview` | OpenRouter | $0.000063 |
| oracle (2) | `deepseek-v4-pro` | DeepSeek | $0.000435 |
| oracle (2) | `anthropic/claude-sonnet-4.6` | OpenRouter | $0.003 |
| oracle (2) | `xiaomi/mimo-v2.5-pro` | OpenRouter | $0.000435 |
| oracle (2) | `moonshotai/kimi-k2.6` | OpenRouter | $0.000684 |

`perf_router.model_set` chooses the pool: `curated` (the IDs in `curated_models`) or `full` (every model in the PerfRouter data, currently ~80). On failure a request escalates to the best-scoring healthy model that can take the prompt — see [Escalation and model health](#escalation-and-model-health); after the attempts are used up the proxy returns 502. Edit `config.yaml` and hard-restart (or `POST /optmod/restart`) to change the pool.

## Install and run

### Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- The sibling [`perfrouter`](../perfrouter) repository checked out next to this one (`../perfrouter`). The default router
  loads its data (`../perfrouter/data`), checkpoint (`../perfrouter/models/perf_router.pkl`) and `models.yaml` from there
  (`perf_router.perfrouter_data_dir` in `config.yaml`).
- An OpenRouter API key. A DeepSeek key is also needed for the default primary model (`deepseek-v4-flash`); keys for other
  providers (Google, Anthropic, OpenAI, Moonshot) are optional, since models without a native key go through OpenRouter.
- Optional: a GPU makes the Laya tier faster; everything also runs on CPU.

### Install

```bash
cd ~/workspace
git clone <optmod-url> optmod && git clone <perfrouter-url> perfrouter   # side by side

cd optmod
uv venv
uv pip install -e '.[dev,perfrouter]'     # perfrouter extra = the sibling package (xgboost, sentence-transformers)
uv pip install -e '.[laya]'               # optional: Laya difficulty tier (perf_router.laya.mode != off)
uv pip install -e '.[trouter]'            # optional: only for the `trouter` router (torch)
```

The first start downloads the `all-MiniLM-L6-v2` encoder (and the Laya checkpoint if enabled) from Hugging Face, which takes a
minute; later starts load from the local cache. Set `HF_TOKEN` to avoid anonymous-download rate limits.

### Configure keys

```bash
cat > .env <<'EOF2'
OPENROUTER_API_KEY=sk-or-...
DEEPSEEK_API_KEY=sk-...
# GOOGLE_API_KEY= ANTHROPIC_API_KEY= OPENAI_API_KEY= MOONSHOT_API_KEY=   (optional)
EOF2
```

`.env` is git-ignored. `uv run` does not load it by itself, so always pass `--env-file .env` (the helper scripts do).

### Run

```bash
./start.sh            # background process; pid in .optmod.pid, log in optmod.out.log (--foreground, --reload for dev)
./stop.sh

# or directly
uv run --env-file .env uvicorn main:app --host 0.0.0.0 --port 8765 --reload
```

`--reload` only watches `.py` files: after editing `config.yaml`, restart (`./stop.sh && ./start.sh`). Startup takes about
10 seconds while the models load; `start.sh` waits for it.

### Check it works

```bash
curl http://localhost:8765/optmod/status          # router, pool, model health, pins, laya
open http://localhost:8765/ui                     # dashboard (use xdg-open on Linux)

curl -X POST http://localhost:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"optmod","messages":[{"role":"user","content":"why does quicksort fail on sorted input?"}]}'
```

Point any OpenAI-compatible client at `http://localhost:8765/v1` (any API key string; the model name is ignored). The routing
decision of each request is logged to `routing.log.jsonl` and shown in the dashboard.

Swap the router at runtime, no restart needed:

```bash
curl -X POST http://localhost:8765/optmod/router/rule_based      # also: perf_router, trouter, passthrough, decision_tree
```

### See every task type on the dashboard

`scripts/send_task_types.py` sends one agent-style prompt per PerfRouter task type (33) through the proxy, so you can
review the classification, routing mode and chosen model in `/ui`. The prompts are not the taxonomy's training examples.

```bash
uv run --env-file .env python scripts/send_task_types.py                    # all 33 types
uv run --env-file .env python scripts/send_task_types.py --only code,reasoning   # families or type ids
uv run --env-file .env python scripts/send_task_types.py --dry-run          # list prompts, send nothing
```

Options: `--base-url` (default `http://localhost:8765`), `--max-tokens` (32), `--pause` (0.5 s). Each prompt is its own
session (a run id is prefixed, so reruns don't hit the session pin). The requests are real and cost a few cents. After each
call it prints the expected type, the logged `pr_task_type`, `pr_routing_mode`, top similarity and final model, then a match
total. Prompts that mention an attachment, image or very long document (multimodal, long-context) are expected to miss, since
no file is sent.

### Troubleshooting

| Symptom | Fix |
|---|---|
| "no API key" warnings at startup | the key is not in `.env`, or you started without `--env-file .env` |
| PerfRouter "failed to load" in the log; every request goes to the primary model | `../perfrouter` is missing, or `perf_router.pkl` / the data files are not where `perfrouter_data_dir` points |
| `config.yaml` change has no effect | restart the server; `--reload` ignores `.yaml` |
| Dashboard shows old models or tiers | restart, and use CLEAR LOG for rows from a previous pool |
| Slow first request after start | encoder and Laya warm-up; later requests take a few milliseconds of routing time |

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
| `perf_router` | **Default.** Sentence-BERT classifies the query into task types, XGBoost predicts per-model quality, then `quality − α·cost` picks the winner. Tunable via `perf_router_cost_weight` (α), `perf_router_baseline`, `perf_router_degradation_threshold`, `perf_router_min_similarity` in `config.yaml`. |

All routers swap at runtime — `POST /optmod/router/{name}`, no restart required.

## Session pinning

Multi-turn conversations stick to the same model while the provider-side prompt cache (DeepSeek auto-cache, Anthropic explicit cache) is still warm — preserving cached-read pricing rather than paying full input cost on every turn.

Three phases keyed on time since the last turn in the session:

| Phase | Window | Behaviour |
|---|---|---|
| Hard pin | 0 – 300 s | Bypass the router entirely; force the previous model |
| Soft bonus | 300 – 1800 s | Router scores normally, but the previously-used model is credited with its observed cache savings |
| Fresh | > 1800 s | Pin dropped, normal routing |

Escapes that always override the pin:

- Pinned model errors / rate-limits — drop pin, fall through to the router on the next attempt
- Request includes images and the pinned model isn't vision-capable
- Pinned model removed from the registry
- Conversation grew past the pinned model's `context_window` → **rehome** to the closest larger-context model (same tier preferred, tiebreak by cost); if no candidate fits, drop the pin

Session identity is the SHA1 of the first user message in the request. Pin state is in-memory only; restarting the server clears it.

Config knobs in `config.yaml`:

```yaml
session_pin_hard_window_s: 300       # 5 min — provider cache TTL
session_pin_soft_window_s: 1800      # 30 min — soft-bonus tail
session_pin_soft_bonus_weight: 0.5   # 0.0 disables the soft bonus
```

Per-model `supports_vision: true` declares vision capability for the escape clause.

Each log entry carries a `pin_state` field — one of `fresh`, `hard`, `soft`, `rehomed_context`, `evicted_vision`, `evicted_missing_model`, `evicted_context`, `evicted_post_failure` — so you can audit what the pin did on every request.

## Laya difficulty tier

[Laya](https://github.com/nandhakishorm/laya) (optional extra: `uv pip install -e '.[laya]'`) classifies each routed query as `easy` / `medium` / `hard` with one typed question. The tier picks the per-request degradation threshold δ and cost cap passed to PerfRouter's `route()`. `perf_router` only.

| `perf_router.laya.mode` | Behaviour |
|---|---|
| `off` | Identical to having no `laya` block |
| `shadow` (default in `config.yaml`) | Route with the static values; if the tier's values would pick a different model, a second `route()` call records it as `laya_shadow_model` |
| `active` | Tier values are used when Laya answered `ok` with confidence ≥ `min_confidence` |

- Laya runs after the hard-pin gate, on the last 3 user messages (left-truncated to `max_chars`). Below PerfRouter's `min_similarity` it is **observe-only**: `route()` takes its cheapest-model fallback and ignores δ, so the tier is logged but never applied.
- `backend: embedded` runs `laya.load` in-process (warm-up at startup, ~11 ms per call on a GPU). `backend: remote` (`laya-serve` over HTTP, localhost or any URL) is planned and currently reports `unavailable`. A missing package or failed load reports `unavailable`; it never breaks routing.
- Guards: a wall-time check per call, a circuit breaker (`breaker_failures` consecutive errors/timeouts/slow calls open it for `breaker_cooldown_s`), a 512-entry memo, one classifier instance per process.
- δ = 0 is **not** "best quality" (PerfRouter switches to `argmax(quality − α·cost)`); keep every tier's δ above 0.
- Toggle at runtime: `POST /optmod/laya/{off|shadow|active}`; state is under `laya` in `GET /optmod/status`.

```yaml
perf_router:
  laya:
    mode: shadow
    backend: embedded
    embedded: { checkpoint: convaiinnovations/laya, device: null }   # device: cuda, cuda:0, cpu
    timeout_ms: 80
    max_chars: 1500
    min_confidence: 0.5
    breaker_failures: 3
    breaker_cooldown_s: 60
    tiers:                       # placeholders — tune from shadow data
      easy:   { delta: 0.25, cost_cap: 2.0 }
      medium: { delta: 0.15, cost_cap: 2.0 }
      hard:   { delta: 0.03, cost_cap: 4.0 }   # cost_cap: null = no cap
```

Each log entry carries `pr_task_type`, `pr_routing_mode`, `pr_top_similarity`, `laya_status`, `laya_tier`, `laya_confidence`, `laya_ms`, `laya_cached` (memo hit, so `laya_ms` ≈ 0), `laya_applied`, `laya_shadow_model`, `effective_delta`, `effective_cost_cap`. Query text is never logged. `curl_examples.txt` has easy/medium/hard requests to try it; the plan is in `optmod-spec/plans/laya-difficulty-tier.md`.

## Escalation and model health

When an attempt fails, the proxy retries (up to `escalation.max_escalations`) on a different model. Three pieces decide which:

**Error types** — 429/5xx/timeouts/404 (`not_found`)/401-403 (`auth`) and context-overflow 400s (`context_overflow`) are retryable; any other 400 (`bad_request`) stops the request, since resending the same bad request elsewhere will not help.

**Model health (failure TTL)** — a model that fails is marked unhealthy and skipped for a while, so later requests do not walk into the same failure:

```yaml
model_health:
  enabled: true
  max_ttl_s: 86400        # cap when the provider advertises a reset
  ttl_s:                  # seconds by error type; 0 = never mark that error
    not_found: 3600       # 404 — slug gone / unavailable
    server: 120
    timeout: 120
    rate_limit: 30
    auth: 600             # 401/403 — one model's key/access rejected
```

If the failure response carries `Retry-After` or `X-RateLimit-Reset` (headers, or OpenRouter's `error.metadata.headers`), that advertised wait replaces the configured TTL, capped by `max_ttl_s` — so an exhausted free-tier model stays out until its daily reset. State is in memory (cleared on restart) and listed under `model_health` in `GET /optmod/status`.

**Escalation target** — after routing, `_avoid_failed()` replaces the router's pick if that model already failed in this request or is marked unhealthy (routers re-score from scratch and can keep proposing a dead model). Replacements must be healthy, untried, vision-capable when the request has images, and have a context window that fits the prompt.

```yaml
escalation:
  max_escalations: 2
  strategy: best_quality       # best_quality | next_tier
  max_cost_multiplier: null    # null = off; e.g. 5 = only targets up to 5x the primary's price per 1k
```

- `best_quality` (default): the model with the highest predicted quality for this request (`PerfRouterRouter.rank_models`, cost as tie-break), from any tier. With `max_cost_multiplier` set, the best-ranked model within that budget — not necessarily the highest overall; if nothing affordable fits, the cap is relaxed for that swap rather than failing the request.
- `next_tier`: the cheapest model in the tier above (the legacy rule). Routers without `rank_models` use this too.

A swapped decision logs `decision_reason` as `escalated from=… to=… (failed this request | marked unhealthy; best predicted quality=… | next tier up[; cost<=Nx primary])`.

## Prompt size and context windows

`token_count` is estimated as words × 1.3, which undercounts large code/JSON prompts (logged: ~308k estimated vs ~489k billed, a ratio of 1.6–2.2). Context-window checks — PerfRouter's eligibility filter, the escalation target, session-pin rehoming — therefore use the estimate scaled up:

- `token_estimate_margin` (top-level key, default `1.7`) applies to the first turn of a session.
- Once a turn succeeds, that session's own billed/estimated ratio (plus 5% headroom) replaces it.
- A provider "maximum context length … you requested about N tokens" 400 becomes `context_overflow`: N (or the failed model's window + 1) becomes a floor, and the next pick must have a window that holds it.

## Dashboard

Open `http://localhost:8765/ui` for a live dashboard with 5s auto-refresh:

- **Stat cards** — total requests, success rate, escalation rate, error rate, cache hit rate (with active session-pin count)
- **Model distribution** — tier-colored usage bars, per-model cost estimate for the window, and savings vs. always using a baseline model (`dashboard.baseline_model`, default the primary; the label names it and the figure can go negative). The pane resets to "no data yet" after **CLEAR LOG**
- **Task types** — distribution of task classifications
- **Latency histogram** — bucketed with P50/P90/P99/avg
- **Escalation flow** — which models are being escalated to and why
- **Recent requests table** — per-request model (with full name tooltip), models-tried chain, cost, latency, status, routing reason with a Laya chip (tier, confidence, ● applied / ○ observed; hover for details and the full reason), confidence. New rows slide in animated; a pause/resume button freezes the table without stopping stat updates.

## API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | Main proxy — OpenAI wire format |
| `GET` | `/optmod/status` | Router name, primary and dashboard baseline model, full model list with costs, active session-pin count, pin window config, tool-compressor state, unhealthy models (`model_health`), Laya state |
| `POST` | `/optmod/router/{name}` | Swap router: `passthrough`, `rule_based`, `decision_tree`, `trouter`, `perf_router` |
| `POST` | `/optmod/laya/{mode}` | Laya tier mode: `off`, `shadow`, `active` (409 if the active router is not `perf_router`) |
| `POST` | `/optmod/compressor/{state}` | Toggle the `tool_result_compressor` mutator (`on` / `off`) |
| `POST` | `/optmod/log/clear` | Truncate `routing.log.jsonl` |
| `POST` | `/optmod/restart` | Re-read `config.yaml` tier map and touch `main.py` so `uvicorn --reload` picks up changes |
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
model_health.py       ModelHealth — per-model failure TTL (by error type or provider reset time)
start.sh / stop.sh    Run the server in the background with uv (pid + log files)
curl_examples.txt     Easy / medium / hard request examples and log-check commands
scripts/send_task_types.py   One prompt per PerfRouter task type, sent through the proxy for dashboard review
log.py                Append-only JSONL log
stats.py              /api/stats aggregation (model_tokens, cost tracking)
routing/
  __init__.py              BaseRouter ABC + build_router() factory
  passthrough.py           PassthroughRouter
  rule_based.py            RuleBasedRouter (9 rules)
  decision_tree.py         DecisionTreeRouter (scikit-learn)
  trouter/                 TRouter package
    router.py              TRouterRouter (sentence-BERT + MLP)
    trouter_weights.pt     Trained TRouter checkpoint
    train_trouter.py       TRouter training code
  perfrouter/              PerfRouter package
    router.py              PerfRouterRouter — config plumbing, session pin, Laya tier, rank_models()
    inference.py           PerfRouter inference engine (sentence-BERT + XGBoost)
    laya_tier.py           LayaTierClassifier (embedded backend, breaker, memo)
    perf_router.pkl        Fallback PerfRouter checkpoint (the live data is read from perf_router.perfrouter_data_dir)
mutators/
  noop.py                  NoopMutator — passthrough
  thinking_mode.py         ThinkingModeMutator — prepends /think for CoT-capable models
  tool_result_compressor.py ToolResultCompressorMutator — RTK-style compression of tool outputs (git diff/status, grep, ls/tree, build/test output, log dedup, smart-truncate). Runtime toggle via POST /optmod/compressor/{on|off}.
tests/
  test_features.py              Unit tests — FeatureExtractor
  test_routers.py               Unit tests — all routers
  test_escalation.py            Unit tests — EscalationPolicy
  test_model_health.py          Unit tests — ModelHealth TTLs
  test_best_quality_escalation.py Unit tests — reset-header TTL, best-score swap, cost ceiling, overflow recovery, rank_models
  test_laya_tier.py             Unit tests — Laya classifier, tier resolution, router + log plumbing
  test_proxy_e2e.py             Mock e2e tests (respx, no network)
  test_session_pin.py           Unit tests — session pin helpers, escapes, rehoming, cache-token extraction
  test_tool_result_compressor.py Unit tests — every compression filter and edge case
  test_live_e2e.py              Live e2e tests (real provider calls, skipped without API keys)
ui/index.html         Self-contained stats dashboard
.env                  API keys — git-ignored
```

## Running tests

```bash
# Mock tests only (no network, fast)
uv run pytest tests/test_proxy_e2e.py tests/test_routers.py \
              tests/test_features.py tests/test_escalation.py \
              tests/test_session_pin.py tests/test_tool_result_compressor.py \
              tests/test_laya_tier.py tests/test_model_health.py \
              tests/test_best_quality_escalation.py -v

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

## What's not built yet

- `routing_policy.pkl` — produced by running WildClawBench; needed to enable the `decision_tree` router without falling back to `rule_based`
- Hermes plugin (thin wrapper calling `/optmod/*` endpoints)
- Feedback loop CLI (retrains TRouter / decision tree / PerfRouter from `routing.log.jsonl`)
- File-backed session-pin store (current implementation is in-memory only and lost on restart)
