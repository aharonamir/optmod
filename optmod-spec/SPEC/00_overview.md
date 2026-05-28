# 00 — Architecture Overview

## What optmod does

```
Hermes Agent
    │
    │  POST /v1/chat/completions
    ▼
optmod (localhost:8765)
    │
    ├── 1. Parse request (Pydantic)
    ├── 2. Extract features (<1ms, regex only)
    ├── 3. Route → pick ModelConfig
    ├── 4. Mutate context (optional)
    ├── 5. Forward to real model (httpx async)
    │       ├── success → return response
    │       └── error   → escalate to next tier (up to max_escalations)
    └── 6. Log to JSONL + return response to Hermes
```

## Same-turn escalation contract

Hermes sends **one** HTTP request. optmod may make up to `max_escalations + 1`
calls to real models internally. Hermes always sees exactly one response.
This is the core capability that a Hermes plugin cannot provide.

## Constraints (hard)

| Constraint | Value |
|---|---|
| Feature extraction | <1ms, no ML, no I/O |
| Routing decision | <5ms total |
| Max escalations per request | 2 (configurable) |
| Log format | JSONL, one line per request, append-only |
| API compatibility | OpenAI /v1/chat/completions wire format |

## Non-goals (do not build)

- No streaming support in v1 (accept stream=True in request, forward as non-streaming)
- No WebSocket or SSE
- No authentication on the proxy itself (it's local)
- No database (JSONL only)
- No Hermes plugin registration (separate project)
- No online model fine-tuning

## Model pool (PoC)

| Tier | int | Name | Provider | Notes |
|---|---|---|---|---|
| fast | 0 | qwen2.5:7b | ollama | Default for simple tasks |
| reasoning | 1 | qwen3:8b | ollama | Supports /think toggle |
| oracle | 2 | deepseek-v4 | deepseek | API, quality ceiling |

## Escalation path

`fast (0) → reasoning (1) → oracle (2) → fail with 502`

## Entry points

| Path | Description |
|---|---|
| POST /v1/chat/completions | Main proxy endpoint |
| GET /optmod/status | Current router, model list, primary |
| POST /optmod/router/{name} | Swap router at runtime |
| GET /api/stats?range=N | Aggregated stats from JSONL |
| GET /ui | Dashboard HTML |
