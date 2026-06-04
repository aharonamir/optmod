import logging
import os
import re
import json as _json
from pathlib import Path

from optmod.routing import BaseRouter
from optmod.routing.context import RoutingContext
from optmod.schemas import RoutingDecision

_DIR = Path(__file__).parent
_DEFAULT_COST_WEIGHT = 0.3
_DEFAULT_BASELINE    = "deepseek/deepseek-v4-pro"


def _normalize(s: str) -> str:
    """Lowercase and collapse all separators (/ : - _) to underscore."""
    return re.sub(r"[/_:\-]", "_", s.lower())


def _resolve_model(idx: int, model_ids: list[str], registry):
    """
    Map a model index to a ModelConfig by name matching.

    Copied verbatim from trouter_router.py — same matching strategy,
    same fallback to registry.primary.
    """
    if idx >= len(model_ids):
        return registry.primary

    norm_wid  = _normalize(model_ids[idx])
    all_models = registry.all()

    for m in all_models:
        if _normalize(m.name) == norm_wid:
            return m

    for m in all_models:
        local = _normalize(m.name.split("/")[-1])
        if local == norm_wid:
            return m

    for m in all_models:
        norm_name = _normalize(m.name)
        if norm_wid.startswith(norm_name) or norm_name.startswith(norm_wid):
            return m

    # Pass 4: match local component of trained name against registry full name.
    # Handles trained="deepseek/deepseek-v4-flash" → local="deepseek-v4-flash"
    # → registry entry "deepseek-v4-flash" (direct API, no prefix).
    if "/" in model_ids[idx]:
        local_wid = _normalize(model_ids[idx].split("/")[-1])
        for m in all_models:
            if _normalize(m.name) == local_wid:
                return m

    return registry.primary


def _extract_content(raw: str) -> str:
    """
    Extract actual user text from optmod's JSON-wrapped session message format.

    Messages arrive as:
      '{"source": "sess", ..., "content": "hi", "type": "user input"}'

    Falls back to raw string if not JSON or no content field.
    """
    if raw.strip().startswith("{"):
        try:
            return _json.loads(raw).get("content", raw)
        except (_json.JSONDecodeError, AttributeError):
            pass
    return raw


class PerfRouterRouter(BaseRouter):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._ready = False

        alpha_cfg = config.get("perf_router_cost_weight")
        alpha_env = os.environ.get("PERF_ROUTER_COST_WEIGHT")
        self._cost_weight = float(alpha_cfg or alpha_env or _DEFAULT_COST_WEIGHT)

        baseline_cfg = config.get("perf_router_baseline")
        baseline_env = os.environ.get("PERF_ROUTER_BASELINE")
        self._baseline = baseline_cfg or baseline_env or _DEFAULT_BASELINE

        threshold_cfg = config.get("perf_router_degradation_threshold")
        threshold_env = os.environ.get("PERF_ROUTER_DEGRADATION_THRESHOLD")
        self._degradation_threshold = float(
            threshold_cfg if threshold_cfg is not None else (threshold_env or 0.0)
        )

        min_sim_cfg = config.get("perf_router_min_similarity")
        min_sim_env = os.environ.get("PERF_ROUTER_MIN_SIMILARITY")
        self._min_similarity = float(
            min_sim_cfg if min_sim_cfg is not None else (min_sim_env or 0.20)
        )

        bonus_cfg = config.get("session_pin_soft_bonus_weight")
        bonus_env = os.environ.get("SESSION_PIN_SOFT_BONUS_WEIGHT")
        self._soft_bonus_weight = float(
            bonus_cfg if bonus_cfg is not None else (bonus_env or 0.5)
        )

        try:
            from optmod.routing.perf_router_inference import PerfRouterInference

            self._perf_router = PerfRouterInference(
                router_path              = _DIR / "perf_router.pkl",
                taxonomy_path            = _DIR / "task_taxonomy.json",
                registry_path            = _DIR / "model_registry.json",
                features_path            = _DIR / "model_features.csv",
                cost_weight              = self._cost_weight,
                baseline_model           = self._baseline,
                min_similarity_threshold = self._min_similarity,
            )
            self._ready = True
            logging.info(
                f"[optmod] PerfRouterRouter loaded "
                f"(α={self._cost_weight}, baseline={self._baseline}, "
                f"degradation={self._degradation_threshold}, "
                f"min_sim={self._min_similarity})"
            )
        except Exception as exc:
            logging.warning(f"[optmod] PerfRouterRouter failed to load: {exc}")

    def _find_inference_model_id(self, registry_name: str) -> str | None:
        """Inverse of _resolve_model: registry model name → inference _model_ids entry."""
        if not self._ready:
            return None
        norm_target = _normalize(registry_name)
        norm_local  = _normalize(registry_name.split("/")[-1])
        for mid in self._perf_router._model_ids:
            if _normalize(mid) == norm_target:
                return mid
        for mid in self._perf_router._model_ids:
            if _normalize(mid.split("/")[-1]) == norm_local:
                return mid
        return None

    def route(self, ctx: RoutingContext) -> RoutingDecision:
        if not self._ready:
            return self._passthrough(ctx, "perf_router: not initialised, using primary")
        try:
            return self._route(ctx)
        except Exception as exc:
            logging.warning(f"[optmod] PerfRouterRouter.route error: {exc}")
            return self._passthrough(ctx, f"perf_router error: {exc}")

    def _route(self, ctx: RoutingContext) -> RoutingDecision:
        token_count = ctx.features.token_count or None

        # ── Detect image attachments ──────────────────────────────────────────
        has_images = False
        for msg in ctx.request.messages:
            if isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        has_images = True
                        break
            if has_images:
                break

        # ── Build routing text from last 3 user messages ──────────────────────
        # Always use the last 3 user messages rather than just the last message.
        # This correctly handles follow-ups like "yes", "fix it", "do that"
        # which are meaningless without context from prior turns.
        #
        # Only user messages are included — system prompts describe model
        # behaviour (not the task), assistant messages are prior responses,
        # and tool messages are structured JSON blobs. All three add noise
        # to the sentence-BERT classification.
        #
        # The last message is repeated at the end to bias the embedding
        # toward the current intent without losing surrounding context.
        user_messages = []
        for msg in ctx.request.messages:
            if msg.role != "user":
                continue
            if isinstance(msg.content, str) and msg.content.strip():
                content = _extract_content(msg.content.strip())
            elif isinstance(msg.content, list):
                content = " ".join(
                    _extract_content(p.get("text", ""))
                    for p in msg.content
                    if isinstance(p, dict) and p.get("type") == "text"
                ).strip()
            else:
                content = ""
            if content:
                user_messages.append(content)

        last_3 = user_messages[-3:]

        # Repeat last message to weight current intent in the embedding
        if last_3:
            last_3 = last_3 + [last_3[-1]]

        routing_text = "\n".join(last_3) if last_3 else ""

        # ── Resolve session pin to inference model_id ─────────────────────────
        pin_info = None
        if ctx.session_pin is not None:
            pin_mid = self._find_inference_model_id(ctx.session_pin.model_name)
            if pin_mid is not None:
                pin_info = {
                    "model_id":     pin_mid,
                    "cache_rate":   ctx.session_pin.last_cache_rate,
                    "bonus_weight": self._soft_bonus_weight,
                }

        # ── Route ─────────────────────────────────────────────────────────────
        decision = self._perf_router.route(
            routing_text,
            token_count           = token_count,
            has_images            = has_images,
            degradation_threshold = self._degradation_threshold,
            pin_info              = pin_info,
        )

        chosen_id    = decision["decision_model"]
        model        = _resolve_model(0, [chosen_id], ctx.registry)
        mutator      = "thinking_mode" if model.thinking_mode else "noop"

        task_type      = decision.get("task_type", "unknown")
        cost_saved_pct = decision.get("cost_saved_pct", 0.0)
        alpha          = decision.get("alpha", self._cost_weight)
        quality        = decision.get("predicted_quality", 0.5)
        routing_mode   = decision.get("routing_mode", "normal")
        pin_bonus      = decision.get("pin_soft_bonus", 0.0)

        return RoutingDecision(
            model       = model,
            mutator     = mutator,
            reason      = (
                f"perf_router: {task_type} → {chosen_id} "
                f"quality={quality:.3f} "
                f"cost_saved={cost_saved_pct:+.1f}% "
                f"α={alpha:.2f} "
                f"degradation={self._degradation_threshold:.2f} "
                f"mode={routing_mode} "
                f"pin_soft_bonus={pin_bonus:.3f}"
            ),
            confidence  = float(quality),
            router_name = self.name,
        )