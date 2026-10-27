# PerfRouter taxonomy for agentic (SkillsBench-style) prompts

## Phases

| Phase | Status | Scope |
|---|---|---|
| 1 | Done (`bfa4094`) | Log/dashboard show `pr_task_type` instead of the keyword-regex `tool_call` label. |
| 2 | **Done** | 56 agentic examples in `../perfrouter/data/agentic_examples.json`, merged by `patch_taxonomy_agentic.py`; inference scores each type by its closest example (`example_pooling: max`). No new types, so no XGBoost retrain. |
| 3 | **Done** | `softmax_temperature` weights, `min_similarity` 0.20 -> 0.25, `low_similarity_fallback: general`. Metadata hint not built. Validation (live run, hand labels) is yours. |

Training repo: `../perfrouter` (taxonomy built by `perfrouter/pipeline/build_task_taxonomy.py`, one-shot; inference
in `perfrouter/inference/perf_router_inference.py`, copied into `routing/perfrouter/` here).

## Problem (2026-10-26 replay of 87 SkillsBench `task.md` prompts)

- Median top-1 similarity 0.25; 13 of 87 are below `min_similarity` 0.20, so they take `fallback_ambiguous` (cheapest model).
- Top-1 vs top-2 gaps are often a few hundredths, so the winning type is close to noise
  (`3d-scan-calc` -> `instruction.format_constraint`, `adaptive-cruise-control` -> `tool_use.multi_step_workflow` at 0.17).
- Each type is embedded from its **definition + the first 2 examples only** (`perf_router_inference.py:324`); the examples
  are short one-liners ("find the bug in this function"), far from a real agent task prompt (long, file paths, constraints).
- Agent traffic repeats the same task prompt every turn, so one classification decides the whole session.

## Phase 2: examples and embedding

1. **Write SkillsBench-style examples per type.** Realistic 2-4 sentence task prompts, grouped by the SkillsBench
   categories (software-engineering 16, office-white-collar 14, natural-science 14, industrial-physical-systems 14,
   finance-economics 9, mathematics/formal 8, cybersecurity 7, media-content 5) and their `task_type` tags
   (analysis, calculation, transformation, implementation, detection, optimization, extraction, planning, repair).
   Example domains to cover: PDF/spreadsheet/CSV form filling and extraction, binary/file-format parsing, scientific
   data analysis (hydrology, astronomy, seismology), simulation/control code, vulnerability and intrusion analysis,
   video/media processing, performance optimization, economic/finance calculation.
2. **Map them to existing types before adding new ones.** Most fit `tool_use.terminal_agent`, `code.scientific_coding`,
   `synthesis.data_analysis_report`, `retrieval.document_extraction`, `code.debugging`. Add a new type only where no
   type fits (candidates: `agentic.file_format_processing`, `agentic.domain_calculation`) and give it
   `benchmark_weights` from terminal-bench / tau2 / scicode so the quality model has an affinity column for it.
3. **Do not hand-write labels from the replay.** Hold out ~20 of the 87 `task.md` prompts as a test set; write examples
   from the rest (or from generic descriptions), so the accuracy figure is not measured on training text.
4. **Use all examples at inference.** Change the task-type embedding from `definition + examples[:2]` to the mean
   (normalised) of the per-example embeddings plus the definition, or max-sim over examples. Re-check cost: embeddings
   are computed once at startup.
5. **Strip harness boilerplate before classifying.** Agent first messages carry instructions around the task; extract the
   task body (the `_extract_content` hook in `router.py`) so wrapper text does not dominate the embedding.
6. **Regenerate** `task_taxonomy.json` via `patch`, not a full `build_task_taxonomy.py` rerun (CLAUDE.md in the
   training repo says never regenerate unless the taxonomy changes; adding types is that case), then retrain
   `perf_router.pkl` and copy the artifacts into `routing/perfrouter/`.

## Phase 3: scoring and thresholds

1. **Margin-aware weights.** `predict_quality` weights the top-3 types by raw similarity; use a softmax with a temperature
   so a 0.01 gap is a near-tie and the quality prediction is a blend, not a coin flip.
2. **`min_similarity` for agent traffic.** With better examples re-measure the similarity distribution and set the
   threshold from the data; consider a low-similarity rescue (see `laya-difficulty-tier.md` phase 3) instead of the
   cheapest-model fallback.
3. **Optional metadata hint.** If the harness passes SkillsBench category/subcategory, map it directly to a task type.

## Validation

- Replay the 87 `task.md` prompts (script: classify first 1,500 chars of the body after the front matter); report
  below-threshold rate, median top-1 similarity, and top-1/top-2 gap before and after.
- Held-out accuracy against a hand-labelled subset of ~20 tasks (primary type acceptable set, not one label).
- Offline: chosen model and predicted cost per task before/after; then a live SkillsBench run comparing pass rate and
  cost per solved task from `routing.log.jsonl` (`pr_task_type`, `pr_top_similarity`, `pr_routing_mode`).

## Implementation notes (2026-10-26)

- **Deviation: no new task types.** Types have per-type affinity columns in the XGBoost model, so a new type needs benchmark
  weights and a retrain. All SkillsBench-style prompts fit existing types (terminal_agent, scientific_coding,
  data_analysis_report, document_extraction, code_review, ...). Revisit if validation shows a real gap.
- **Examples** live in `../perfrouter/data/agentic_examples.json` (source of truth), merged into `task_taxonomy.json` as
  `agentic_examples` by `python -m perfrouter.pipeline.patch_taxonomy_agentic` (idempotent). Re-run it after any taxonomy rebuild.
  Written from generic domains, not benchmark text. Training does not read them; only inference does.
- **Inference** (`perf_router_inference.py` in both repos): a type's similarity = max over its definition text and its examples
  (`example_pooling: none` restores the legacy definition + 2 examples). `softmax_temperature` (0.05) replaces the linear
  similarity weights across the top-3 types in `predict_quality`; null = legacy.
- **Low similarity:** `low_similarity_fallback: general` routes below `min_similarity` using the task-agnostic quality prior
  (all types weighted equally, normal degradation/cost logic) with `pr_routing_mode = general_prior`; `cheapest` = legacy
  `fallback_ambiguous`. Laya stays observe-only on these rows.
- **Replay** (`python -m perfrouter.tools.replay_skillsbench --tasks ~/workspace/skillsbench/tasks --pooling none|max`),
  87 tasks, every 4th held out:

  | | median top-1 sim | below 0.20 | top1-top2 gap < 0.02 |
  |---|---|---|---|
  | legacy (`none`) | 0.251 | 15% | 30% |
  | `max` + examples | 0.373 | 0% | 32% |

  The similarity scale moved up (5th/10th percentile 0.24/0.26), hence `min_similarity` 0.25 (about 5-8% of tasks fall back).
  The top1-top2 gap barely moved: near-ties between neighbouring types remain, which is what the softmax blend is for.
  Held-out top-1 labels look sensible on a spot check; accuracy against hand labels is not measured.
- **Not built:** the SkillsBench category metadata hint (needs the harness to pass it), chunked embedding of long prompts
  (MiniLM truncates at 256 tokens, so only the start of a long prompt is embedded), and boilerplate stripping (no wrapper
  text seen in logged sessions, as the user text is the task prompt).

## Next steps: retrain the quality model on SkillsBench results (phase 4, after validation runs)

Phases 2-3 only improve which task type a prompt gets and how the top-3 types are weighted. `perf_router.pkl` (XGBoost) is
unchanged: it predicts quality per (model, task type) from benchmark-derived affinity scores (AA, Arena, terminal-bench,
scicode, ...) plus WildClawBench ground truth where a WCB CSV was supplied. No SkillsBench outcome is in it, so a correct
type does not guarantee the quality ranking for that type matches how models do on SkillsBench. After the validation runs:

1. **Collect outcomes.** Per model and task: pass/fail (or verifier score) and cost, from the run results joined with
   `routing.log.jsonl` (`pr_task_type`, `final_model`). Route runs through several models, not only the router's picks,
   or the data only covers what the router already chose.
2. **Convert to training rows.** Per (model, task type): mean score over tasks of that type, written in the format
   `train_perf_router.py` reads for `--wcb-csv` (`_load_wcb_scores`), so they get ground-truth weight 3.0 vs 1.0 for
   benchmark-derived rows. Use the same `pr_task_type` the classifier assigns, so train and inference types agree.
3. **Check coverage.** Models that never ran get benchmark-derived rows only; note which types have too few tasks to trust.
4. **Retrain** with `train_perf_router.py --wcb-csv <skillsbench.csv>`, keep a held-out set of tasks for evaluation,
   copy `perf_router.pkl` to `../perfrouter/models/`, restart optmod.
5. **Cheaper interim option:** give the agentic types (`tool_use.terminal_agent`, `code.scientific_coding`,
   `synthesis.data_analysis_report`) benchmark weights that reflect agent work, then rebuild features and retrain.
6. **Re-tune** `cost_weight`, `degradation_threshold` and `min_similarity` on the new model.

Also confirm how the current `.pkl` was trained (whether a WCB CSV was used; the file is dated 2026-10-05).

## Classifier comparison (2026-10-27)

Tool: `../perfrouter/perfrouter/tools/eval_task_classifiers.py`; labels: `../perfrouter/data/skillsbench_task_labels.json`
(acceptable type sets for the 87 tasks, written by an LLM from the first ~300 characters; **noisy**, a rough yardstick only).
Every method must output one of the 33 taxonomy types. Models get the first 2,000 characters; MiniLM sees the full body
(it truncates at 256 tokens). Run: `PYTHONPATH=~/workspace/perfrouter uv run [--with gliclass] python -m perfrouter.tools.eval_task_classifiers --methods ...`.

| method | top-1 = primary | in acceptable set | family ok | ms/task | model routed in acceptable set |
|---|---|---|---|---|---|
| MiniLM legacy (def + 2 examples) | 20% | 24% | 32% | 68 | 39% |
| MiniLM + agentic examples (current) | 26% | 45% | 61% | 43 | 60% |
| MiniLM chunk-pooled | 28% | 44% | 60% | 6 | 54% |
| Laya, 33 options | 13% | 17% | 57% | 27 | 37% |
| Laya, family then type | 23% | 37% | 57% | 37 | 47% |
| **Jev 1.13 hosted (OpenRouter), 33 options** | **53%** | **64%** | **72%** | 433 | 60% |
| Jev, family then type | 46% | 55% | 63% | 862 | 53% |
| deepseek-v4-flash as classifier (reference) | 40% | 53% | 68% | 1261 | 56% |
| ModernBERT-large zero-shot NLI | 41% | 51% | 55% | 611 | 61% |
| DeBERTa-v3-large zero-shot NLI | 30% | 38% | 39% | 711 | 53% |
| GLiClass modern-base v3 | 25% | 34% | 55% | 230 | 53% |
| GLiClass large v3 | 34% | 43% | 57% | 509 | 51% |

Reading it:
- Hosted Jev is the most accurate classifier here (+19 points over the current MiniLM in acceptable-set hits), but routed to a
  model the gain mostly disappears (60% vs 60%): many types map to the same model in the quality model (only 7 distinct models
  are picked across all labels), so label accuracy is not the bottleneck. The bottleneck is the quality model (phase 4).
- Laya is a poor 33-way classifier (17% flat); keep it for the 3-way difficulty tier only.
- Type overlap limits every method: the family-level ceiling is ~72%; many SkillsBench tasks fit several types.
- Not tested: the open Jev weights (`autotrust/JEV-9B/27B/35B`, Apache-2.0, Qwen3.5 base, vLLM, bf16 9B needs ~18 GB vs the
  12 GB GPU here) and NVIDIA's `prompt-task-and-complexity-classifier` (11 coarse categories, no mapping to the 33 types).
- Caveat: LLM-made labels favour LLM-style classifiers (Jev, the chat model). Hand-check the labels before deciding.
- Decision: keep MiniLM for now. Revisit Jev (hosted API: prompts leave the machine, ~0.4 s per session start, memoizable
  per session) once the quality model is retrained on SkillsBench; then a better type may matter.

## Next steps: optional Jev task classifier (phase 5, after phase 4)

Make the task-type classifier pluggable so hosted Jev can replace MiniLM by config, off by default (MiniLM stays the default,
needs no network). Jev only classifies; the quality model, softmax weights and escalation are unchanged. Worth doing after the
quality model is retrained (phase 4): today most types route to the same few models, so the better label barely changes the pick.

1. **Config:** `perf_router.task_classifier: minilm | jev` (default `minilm`), plus a `jev:` block (`model: typesafe/jev-1.13`,
   `api_key_env: OPENROUTER_API_KEY`, `timeout_ms`, `max_chars: 2000`, `breaker_failures`, `breaker_cooldown_s`).
   Toggle at runtime like Laya (`POST /optmod/task-classifier/{minilm|jev}`; status in `GET /optmod/status`).
2. **Backend:** one `choice` question over the 33 types, definitions as criteria, via `POST https://openrouter.ai/api/alpha/decisions`
   (flat beat family-then-type in the comparison). Output: the type plus its probability, used as the top-1 "similarity".
3. **Top-k:** the Jev probabilities give the top-3 types for the softmax weighting directly (they are already a distribution;
   use them as weights instead of the cosine scores, with `min_similarity` replaced by a probability floor).
4. **Memo per session:** the task prompt repeats every turn, so classify once per session id and reuse (also skips the 0.4 s call).
   Cache key = hash of the first user message, same idea as the Laya memo.
5. **Failure handling:** never raise; on timeout, error, 4xx/5xx or an open breaker, fall back to MiniLM for that request and
   log it. Same breaker pattern as `laya_tier.py`.
6. **Logging:** `pr_classifier` (`minilm` / `jev` / `jev_fallback_minilm`), `pr_task_type`, `pr_top_similarity` (probability
   for Jev), `jev_ms`. Prompt text is never logged.
7. **Privacy and cost:** task prompts leave the machine (first 2,000 characters). State this next to the config key; cost is
   about $0.04 per 1M input tokens, negligible next to the model calls.
8. **Validate before enabling:** rerun `eval_task_classifiers.py` on hand-checked labels and compare routed models, then a
   live SkillsBench run with `jev` vs `minilm`.
9. **Open weights alternative:** `autotrust/JEV-9B/27B` (Apache-2.0) would keep prompts local but need vLLM and more VRAM than
   the 12 GB GPU here; only if the hosted API is unacceptable.
