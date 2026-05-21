from dataclasses import dataclass, field

from .schemas import RoutingContext
from .registry import ModelConfig


@dataclass
class EscalationPolicy:
    max_escalations: int = 2

    retryable: set[str] = field(default_factory=lambda: {
        "rate_limit", "server", "timeout"
    })

    non_retryable: set[str] = field(default_factory=lambda: {
        "auth", "bad_request"
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
        last_model = ctx.registry.get_or_default(ctx.models_tried[-1])
        return ctx.registry.next_tier_up(last_model) is not None

    def next_model(self, ctx: RoutingContext) -> ModelConfig | None:
        if not ctx.models_tried:
            return ctx.registry.primary
        last = ctx.registry.get_or_default(ctx.models_tried[-1])
        return ctx.registry.next_tier_up(last)
