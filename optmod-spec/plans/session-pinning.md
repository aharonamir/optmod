# Session pinning to preserve provider-side prompt cache

## Context

`optmod` is a routing proxy. Today each request runs `_router.route(ctx)` independently, with no awareness of prior turns in the same conversation. When `perf_router` chooses a different model on turn N than on turn N-1, the provider-side prompt cache (DeepSeek auto-cache, Anthropic explicit cache via OpenRouter) is wasted: the new model pays full input price for the entire prefix. For multi-turn agent loops this can dwarf any savings from picking a "cheaper" model.

`session_id` already exists (SHA1 of first user message, derived in `main.py:68`) but is purely diagnostic — no router consults it, no state persists across requests. Cached-token usage in provider responses is silently dropped (`main.py:162-184` reads only `prompt_tokens` / `completion_tokens`).

Goal: keep multi-turn sessions on the same model while the provider cache is hot, then transition smoothly back to fresh routing once the cache is dead.

## Design

A three-phase stickiness model keyed on time-since-last-turn:

| Phase | Window | Behaviour |
|---|---|---|
| Hard pin | 0 – 300 s | Bypass `_router.route()` entirely. Force previous model. |
| Soft bonus | 300 – 1800 s | Let `perf_router` score normally, but discount the previous model's cost by its observed cache savings. |
| Fresh | > 1800 s | Drop pin, normal routing. |

Pin state is an in-memory dict keyed by `session_id`, lazily evicted on access. Wiped on restart — acceptable because the provider cache is also gone after the kind of pause that triggers a restart.

## Schema additions (`schemas.py`)

```python
@dataclass
class SessionPin:
    model_name:              str
    last_turn_at:            float    # epoch seconds
    last_cache_rate:         float    # 0..1, from previous turn's usage
    last_prompt_tokens:      int
    turn_count:              int
```

- `RoutingContext` gets `session_pin: SessionPin | None = None`
- `LogEntry` gets `cached_tokens: int = 0` and `pin_state: str = "fresh"` (one of: `fresh`, `hard`, `soft`, `evicted_vision`, `evicted_missing_model`, `evicted_post_failure`, `evicted_context`, `rehomed_context`)

## `ModelConfig` addition (`registry.py` + `config.yaml`)

Add `supports_vision: bool = False` to `ModelConfig`. Required for the vision-escape clause in the hard pin. Populate per-model in `config.yaml` (only the ones that actually accept images get `true`).

## Hard-pin gate (`main.py`)

A module-level `_session_pins: dict[str, SessionPin] = {}` plus two helpers:

- `_get_active_pin(session_id, now) -> SessionPin | None` — lookup with lazy eviction if `now - last_turn_at > SOFT_WINDOW_S`.
- `_update_pin(session_id, model_name, cached_tokens, prompt_tokens, now)` — write-through after a successful response.

In `chat_completions`, before the escalation loop:

```python
pin = _get_active_pin(session, time.time())
ctx.session_pin = pin

if pin and (now - pin.last_turn_at) < HARD_WINDOW_S:
    decision = _try_hard_pin(pin, ctx)   # returns RoutingDecision or None
    if decision is None:
        # escape — vision required and pinned model can't, or pinned model removed
        _session_pins.pop(session, None)
        ctx.session_pin = None
        pin_state = "evicted_vision" or "evicted_missing_model"
    else:
        pin_state = "hard"
```

Escape conditions checked inside `_try_hard_pin`:
1. `features.has_images` and not `pin_model.supports_vision` → evict, pin_state `evicted_vision`
2. `pin.model_name` not in `_registry` (config hot-reload removed it) → evict, pin_state `evicted_missing_model`
3. `features.token_count > pin_model.context_window` → **rehome** to the nearest fitting model (see below). The pin updates in place to the rehomed model; pin_state `rehomed_context`. This is *not* an eviction — stickiness intent is preserved against a model that's as close as possible to the original pin.

When the hard pin is honoured, `_router.route(ctx)` is skipped for the first attempt only; if that attempt errors, the escalation loop falls through to the normal router on attempt 1 (see below).

### Rehoming algorithm (context overflow)

Goal: pick a replacement that fits the current `token_count` and is closest to the pinned model in both cost and quality. Candidate set: all registered models with `context_window >= features.token_count`, `supports_tools >= pin_model.supports_tools`, `supports_vision >= pin_model.supports_vision`. Exclude the pinned model itself (it doesn't fit by definition).

If `perf_router` is the active router, use its quality vector (already loaded at startup) and score:

```
dist(m) = abs(quality(m) - quality(pin))
        + cost_weight × abs(cost_per_1k(m) - cost_per_1k(pin)) / max_cost
```

Pick `argmin(dist)`. Reuses the same `cost_weight` (α) already in config so there's no new knob.

If a non-perf router is active, fall back to a tier-based heuristic: among candidates, prefer same tier as the pin (else next higher tier), then tiebreak by `|cost_per_1k − pin.cost_per_1k|`. This is approximate but adequate since the soft bonus is perf_router-only anyway.

If no candidate fits at all (extreme prompt size), the rehome fails — drop the pin entirely, pin_state `evicted_context`, let `_router.route(ctx)` decide on attempt 0. perf_router's existing token-count eligibility filter will pick something sensible.

## Pinned-model failure recovery (`main.py`)

The existing escalation policy only walks `next_tier_up`. If the pinned model is top-tier (e.g. `deepseek-v4-pro`, tier 2) and fails, there's no tier up — request 502s. Today this is generally fine, but the pin makes top-tier retries more frequent.

Fix: on a pinned-model failure, drop the pin and let `_router.route(ctx)` choose the recovery model on the next attempt. Concretely, in the escalation loop:

```python
if error_type is not None and attempt == 0 and pin_state == "hard":
    _session_pins.pop(session, None)
    pin_state = "evicted_post_failure"
    ctx.session_pin = None   # next iteration will route via _router
```

## Soft bonus inside `perf_router` (`routing/perf_router_inference.py`)

Plumb `session_pin` from `PerfRouterRouter._route` (around line 185-190 in `perf_router_router.py`, where `degradation_threshold` is already passed) into `PerfRouterInference.route(...)`.

The bonus needs to land in **all three** selection branches at lines 514, 528, 540:

- **Standard branch (line 540)**: add bonus to `adjusted[pin_idx]` using a **normalised** magnitude so it interacts cleanly with `α × _costs_norm`:
  ```
  bonus = pin.last_cache_rate × _costs_norm[pin_idx] × soft_bonus_weight
  adjusted[pin_idx] += cost_weight × bonus
  ```
- **Degradation branch (line 528)**: the chosen_idx is `argmin(costs_for_selection)` over `_costs_raw`, ignoring `adjusted`. Apply the bonus in cost space instead:
  ```
  costs_for_selection[pin_idx] *= (1 - pin.last_cache_rate × soft_bonus_weight)
  ```
- **Fallback branch (line 514)**: same cost-space discount on `costs_for_fallback`.

New config key: `session_pin_soft_bonus_weight: 0.5` (tuneable; 1.0 = full cache savings credited, 0.0 = bonus disabled).

The bonus only applies when `HARD_WINDOW_S ≤ age < SOFT_WINDOW_S` and the pinned model is still in the candidate set.

Reason string includes `pin_soft_bonus=<value>` so it shows up in logs.

## Cache-token extraction (`forwarder.py`)

After receiving a 200 response, extract cached tokens defensively (providers differ):

```python
usage  = body.get("usage") or {}
cached = (
    usage.get("prompt_tokens_details", {}).get("cached_tokens")  # OpenAI/DeepSeek OpenAI-compat, OpenRouter normalised
    or usage.get("prompt_cache_hit_tokens")                       # DeepSeek legacy
    or usage.get("cache_read_input_tokens")                       # Anthropic native
    or 0
)
body.setdefault("usage", {})["_optmod_cached_tokens"] = cached
```

Read it back in `main.py` to update the pin and log it.

## Config (`config.yaml`)

```yaml
session_pin_hard_window_s: 300       # 5 min — DeepSeek/Anthropic cache TTL
session_pin_soft_window_s: 1800      # 30 min — bonus tail
session_pin_soft_bonus_weight: 0.5   # 0.0 disables soft bonus
```

Per-model `supports_vision: true` for any model in the pool that actually accepts image content (today: none of the configured models; future-proofed).

## Stats + dashboard (`stats.py`, `ui/index.html`)

- `stats.py` aggregation: `cache_hit_rate = sum(cached_tokens) / max(1, sum(prompt_tokens))`.
- Dashboard: one new stat tile next to the latency tile — "Cache hit rate" — and a small "Session pins active" counter (length of `_session_pins` at the moment `/optmod/status` is called).

Add the count to `/optmod/status` so the dashboard can poll it without a new endpoint.

## Concurrency

Single-worker uvicorn + asyncio. Two concurrent requests to the same session both read the same pin → both choose the same model (benign). Both write back → last writer wins on counters/cache-rate (also benign). No lock needed for the dict itself. Use a one-shot `asyncio.Lock` only around the lazy-eviction sweep to avoid two coroutines walking the dict simultaneously.

## Files to modify

| File | Change |
|---|---|
| `schemas.py` | Add `SessionPin` dataclass, `RoutingContext.session_pin`, `LogEntry.cached_tokens` + `pin_state` |
| `registry.py` | Add `supports_vision: bool = False` to `ModelConfig` |
| `config.yaml` | Three new top-level keys; populate `supports_vision` per model |
| `main.py` | `_session_pins` dict, `_get_active_pin` / `_update_pin` / `_try_hard_pin` helpers, hard-pin gate before escalation loop, post-failure pin drop, cached_tokens extraction into log, pin count in `/optmod/status` |
| `forwarder.py` | Defensive multi-key cached-token extraction into `body["usage"]["_optmod_cached_tokens"]` |
| `routing/perf_router_router.py` | Pass `ctx.session_pin` + new `session_pin_soft_bonus_weight` into inference |
| `routing/perf_router_inference.py` | Apply bonus in all three selection branches (standard / degradation / fallback) |
| `stats.py` | `cache_hit_rate` aggregation |
| `ui/index.html` | Cache-hit-rate tile, session-pins-active counter |
| `tests/test_session_pin.py` | New |

## Implementation order (each step independently observable)

1. **Cache visibility first.** `forwarder.py` extraction + `LogEntry.cached_tokens` + stats aggregation + dashboard tile. No behaviour change. Verify against live DeepSeek + OpenRouter traffic before building stickiness on top — confirms the response shape and gives a baseline cache-hit-rate to measure improvement against.
2. **Hard pin.** `SessionPin`, `_session_pins`, hard-pin gate, vision/missing escapes, post-failure pin drop, `pin_state` field. Ship and observe how often pins fire.
3. **Soft bonus.** Plumb pin into `perf_router_inference`, apply bonus in all three branches, new config key.
4. **Operator visibility.** Session-pins-active counter in `/optmod/status` + dashboard.
5. **Tests.** See below.

## Tests (`tests/test_session_pin.py`)

- First turn → no pin → router decides freely.
- Second turn within hard window → pinned model returned regardless of what router would have picked.
- Second turn between hard and soft windows → `perf_router` mock receives `session_pin` and applies bonus to the right candidate.
- Second turn after soft window → no pin in context, fresh routing.
- Vision escape: text-only pinned model + image content → pin evicted, fresh routing.
- Missing-model escape: pinned model removed from registry → pin evicted, fresh routing.
- Context-overflow rehome: prompt exceeds pinned model's context_window → rehomed to a larger-context model with closest (quality, cost) under perf_router; pin updates in place; `pin_state="rehomed_context"`.
- Context-overflow with no fitting candidate → `pin_state="evicted_context"`, fresh routing.
- Pinned-model failure: hard-pinned model returns 500 → pin dropped, next attempt routes via `_router.route(ctx)`.
- Cache-token extraction: mock responses for DeepSeek (`prompt_tokens_details.cached_tokens`), Anthropic-via-OpenRouter (same path), legacy DeepSeek (`prompt_cache_hit_tokens`) all surface a non-zero `cached_tokens` in `LogEntry`.
- Concurrent same-session requests don't crash and converge to a single pin entry.

## Verification

End-to-end:

1. Start server: `uv run --env-file .env uvicorn main:app --host 0.0.0.0 --port 8765 --reload`.
2. Run a multi-turn conversation via any OpenAI-compat client (or `curl`) hitting `/v1/chat/completions` — same first user message across turns so they hash to the same `session_id`.
3. Tail `routing.log.jsonl`: turn 1 shows `pin_state="fresh"`, turn 2 (within 5 min) shows `pin_state="hard"` and the same `final_model` as turn 1.
4. Wait > 5 min, send another turn: `pin_state="soft"` and the `decision_reason` includes `pin_soft_bonus=…`.
5. Dashboard shows non-zero cache-hit-rate climbing over the session.
6. Force a pin failure: edit `config.yaml` to put a bad `api_key_env` for the pinned model, hot-restart, send another turn → log shows `pin_state="evicted_post_failure"` and the next attempt uses a different model.

Mock tests: `uv run pytest tests/test_session_pin.py tests/test_routers.py -v`.

Live tests: `uv run pytest tests/test_live_e2e.py -v -s` (requires `OPENROUTER_API_KEY` and `DEEPSEEK_API_KEY`).
