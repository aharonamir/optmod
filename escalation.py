from dataclasses import dataclass, field

from optmod.schemas import RoutingContext
from optmod.registry import ModelConfig


@dataclass
class EscalationPolicy:
    max_escalations: int = 2
    # True (legacy): only escalate while a higher tier exists. False: any other model may be tried,
    # which is what best-quality escalation needs (the best model for a task can be a mid-tier one).
    require_higher_tier: bool = True

    retryable: set[str] = field(default_factory=lambda: {
        "rate_limit", "server", "timeout", "not_found", "auth", "context_overflow"
    })

    non_retryable: set[str] = field(default_factory=lambda: {
        "bad_request"
    })

    def should_escalate(
        self,
        error_type: str,
        attempt: int,
        ctx: RoutingContext,
    ) -> bool:
        if attempt >= self.max_escalations:
            return False
        if error_type in self.non_retryable:
            return False
        if not ctx.models_tried:
            return False
        if not self.require_higher_tier:
            return len(ctx.models_tried) < len(ctx.registry.all())
        last_model = ctx.registry.get_or_default(ctx.models_tried[-1])
        return ctx.registry.next_tier_up(last_model) is not None

    def next_model(self, ctx: RoutingContext) -> ModelConfig | None:
        if not ctx.models_tried:
            return ctx.registry.primary
        last = ctx.registry.get_or_default(ctx.models_tried[-1])
        return ctx.registry.next_tier_up(last)
