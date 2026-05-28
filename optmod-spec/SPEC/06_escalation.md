# 06 — Escalation Policy (escalation.py)

## Purpose

Decides whether to escalate to the next model tier after a failure,
and which model to try next. Called inside the escalation loop in `main.py`.

## Implementation

```python
from dataclasses import dataclass, field
from .schemas import RoutingContext
from .registry import ModelConfig


@dataclass
class EscalationPolicy:
    max_escalations: int = 2

    # Errors that justify trying the next tier
    retryable: set[str] = field(default_factory=lambda: {
        "rate_limit", "server", "timeout"
    })

    # Errors that should NOT escalate (fail immediately)
    non_retryable: set[str] = field(default_factory=lambda: {
        "auth", "bad_request"
    })

    def should_escalate(
        self,
        error_type: str,
        attempt: int,
        ctx: RoutingContext,
    ) -> bool:
        """
        Return True if optmod should try a higher-tier model.

        Conditions for False (do not escalate):
          - attempt >= max_escalations
          - error_type is non_retryable
          - no higher-tier model exists (already at oracle)
        """
        if attempt >= self.max_escalations:
            return False
        if error_type in self.non_retryable:
            return False
        if not ctx.models_tried:
            return False
        last_model = ctx.registry.get_or_default(ctx.models_tried[-1])
        return ctx.registry.next_tier_up(last_model) is not None

    def next_model(self, ctx: RoutingContext) -> ModelConfig | None:
        """
        Return the model to try next.
        Always steps up exactly one tier from the last tried model.
        Returns None if already at top tier or no models tried.
        """
        if not ctx.models_tried:
            return ctx.registry.primary
        last = ctx.registry.get_or_default(ctx.models_tried[-1])
        return ctx.registry.next_tier_up(last)
```

## Error type classification (defined in forwarder.py)

| HTTP status | error_type |
|---|---|
| 429 | rate_limit |
| 401, 403 | auth |
| 400 | bad_request |
| 408, timeout | timeout |
| everything else | server |

## Escalation loop (pseudo-code, see main.py for real code)

```
for attempt in range(max_escalations + 1):
    ctx.attempt_number = attempt
    decision = router.route(ctx)          # may produce different model if ctx updated
    msgs     = mutator.mutate(messages, decision)
    response, error_type = await forwarder.forward(decision.model, request, msgs)
    ctx.models_tried.append(decision.model.name)

    if error_type is None:
        break   # success

    ctx.last_error_type = error_type
    if not policy.should_escalate(error_type, attempt, ctx):
        break   # non-retryable or exhausted

    # loop continues with next attempt
    # router.route() will see updated ctx.models_tried and ctx.last_error_type
```
