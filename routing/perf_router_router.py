import logging
import os
import re
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

        try:
            from optmod.routing.perf_router_inference import PerfRouterInference

            self._perf_router = PerfRouterInference(
                router_path   = _DIR / "perf_router.pkl",
                taxonomy_path = _DIR / "task_taxonomy.json",
                registry_path = _DIR / "model_registry.json",
                features_path = _DIR / "model_features.csv",
                cost_weight   = self._cost_weight,
                baseline_model = self._baseline,
            )
            self._ready = True
            logging.info(
                f"[optmod] PerfRouterRouter loaded "
                f"(α={self._cost_weight}, baseline={self._baseline})"
            )
        except Exception as exc:
            logging.warning(f"[optmod] PerfRouterRouter failed to load: {exc}")

    def route(self, ctx: RoutingContext) -> RoutingDecision:
        if not self._ready:
            return self._passthrough(ctx, "perf_router: not initialised, using primary")
        try:
            return self._route(ctx)
        except Exception as exc:
            logging.warning(f"[optmod] PerfRouterRouter.route error: {exc}")
            return self._passthrough(ctx, f"perf_router error: {exc}")

    def _route(self, ctx: RoutingContext) -> RoutingDecision:
        text        = ctx.features.last_user_message or ""
        token_count = ctx.features.token_count or None

        has_images = False
        for msg in ctx.request.messages:
            if isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        has_images = True
                        break
            if has_images:
                break

        decision = self._perf_router.route(
            text,
            token_count = token_count,
            has_images  = has_images,
        )

        chosen_id = decision["decision_model"]
        model     = _resolve_model(0, [chosen_id], ctx.registry)
        mutator   = "thinking_mode" if model.thinking_mode else "noop"

        task_type      = decision.get("task_type", "unknown")
        cost_saved_pct = decision.get("cost_saved_pct", 0.0)
        alpha          = decision.get("alpha", self._cost_weight)
        quality        = decision.get("predicted_quality", 0.5)

        return RoutingDecision(
            model       = model,
            mutator     = mutator,
            reason      = (
                f"perf_router: {task_type} → {chosen_id} "
                f"cost_saved={cost_saved_pct:+.1f}% α={alpha}"
            ),
            confidence  = float(quality),
            router_name = self.name,
        )
