# Escalation cost control and large-prompt handling

## Phases

| Phase | Status | Scope |
|---|---|---|
| 1 | **Done** (2026-10-26, `96d0681`, `78f24f8`) | Best-score escalation, provider-advertised TTLs, context-overflow recovery, adaptive prompt-size margin, `escalation.max_cost_multiplier: 10`, no pin on over-budget escalations. |
| 2 | Not started | Make the first pick cheaper and stop wasted calls. See below. |

Phase 1 is better but not optimal. Phase 2 collects what is still wrong, from the 2026-10-26 run analysis.

## Phase 2: remaining problems and proposed fixes

1. **The router keeps proposing dead models.** `perf_router` has no exclusion input, so
   `qwen/qwen3.8-27b:free`, `minimax-m2.5` and `gpt-5.1-codex-mini` are picked first, fail, and only
   then does `_avoid_failed()` swap. Every such request pays one wasted call plus latency.
   - Pre-filter: ask the router to rank (`rank_models`) and drop unhealthy / too-small-window models
     before choosing, so a known-bad model is never sent.
   - Fix the stale `:free` slugs in `../perfrouter/models.yaml` (outside this repo).
2. **Cost-blind best-quality target.** Scores differ by a few hundredths while prices differ 10-60x.
   Replace "highest score within N x primary" with a utility, e.g. `quality - a * cost_norm`, or
   "cheapest model within `epsilon` of the best score". Config: `escalation.quality_tolerance`.
3. **Cap relaxation is silent in cost terms.** When nothing affordable fits (e.g. ~500k-token prompts,
   only 525k-window models qualify), the expensive pick still goes out. Options: a hard ceiling
   (`escalation.hard_cost_multiplier`) that returns 502 instead, a log field `cost_cap_relaxed`, and a
   dashboard counter.
4. **Large prompts are billed in full every turn.** ~500k tokens per turn, partly cached.
   - Prefer models with cached-input pricing when the session prompt is large and growing.
   - Optionally compress old tool results above a size threshold before forwarding (the
     `tool_result_compressor` is off by default).
   - Warn / log when a session's prompt crosses a configurable token budget.
5. **Failed-call spend.** Requests that try 3 models and still 502 (`server` errors). Add a per-request
   failure budget and short-circuit once a provider is failing broadly (provider-level breaker, not just
   per-model TTL).
6. **Token estimate.** words x 1.3 undercounts big code/JSON prompts by 1.6-2.2x. Replace with a
   real tokenizer (or a code-aware ratio) so the margin hack is not needed.

## Validation

- Compare per-request cost before and after on the same replayed prompts (`routing.log.jsonl`).
- Track: escalation rate, wasted calls per request, cap-relaxed rate, cost per successful request.
