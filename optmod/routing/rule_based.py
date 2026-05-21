from . import BaseRouter
from .context import RoutingContext
from ..schemas import RoutingDecision


class RuleBasedRouter(BaseRouter):
    def route(self, ctx: RoutingContext) -> RoutingDecision:
        try:
            return self._route(ctx)
        except Exception as e:
            return self._passthrough(ctx, f"rule_based error: {e}")

    def _route(self, ctx: RoutingContext) -> RoutingDecision:
        f     = ctx.features
        reg   = ctx.registry
        tried = ctx.models_tried

        oracle    = reg.by_tier(2)[0]
        reasoning = reg.by_tier(1)[0]
        fast      = reg.by_tier(0)[0]

        def pick(model, mutator="noop", reason="", conf=0.9) -> RoutingDecision:
            return RoutingDecision(
                model=model, mutator=mutator, reason=reason,
                confidence=conf, router_name=self.name,
            )

        # Rule 1
        if all(m.name in tried for m in reg.all()):
            return pick(oracle, reason="all tiers exhausted → oracle")
        # Rule 2
        if ctx.last_error_type == "rate_limit" and oracle.name in tried:
            return pick(reasoning, "thinking_mode", "oracle rate-limited, step down")
        # Rule 3
        if f.difficulty == "hard" or f.token_count > 8000:
            return pick(oracle, reason="hard/long context → oracle")
        # Rule 4
        if f.task_type == "reason" and f.difficulty != "easy":
            return pick(reasoning, "thinking_mode", "non-trivial reasoning → Qwen3 thinking")
        # Rule 5
        if f.task_type == "code" and f.difficulty == "hard":
            return pick(oracle, reason="hard code → oracle")
        # Rule 6
        if f.task_type in {"extract", "summarize"} and f.token_count < 3000:
            return pick(fast, reason="simple extract/summarize → fast")
        # Rule 7
        if f.has_tools and f.task_type == "tool_call":
            return pick(reasoning, reason="tool call → reasoning")
        # Rule 8
        if f.language == "he":
            return pick(reasoning, reason="hebrew content → reasoning")
        # Rule 9
        return pick(fast, reason="default → fast", conf=0.7)
