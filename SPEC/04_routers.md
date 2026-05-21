# 04 — Routers (routing/)

## BaseRouter (routing/__init__.py)

```python
from abc import ABC, abstractmethod
from .context import RoutingContext
from ..schemas import RoutingDecision


class BaseRouter(ABC):
    def __init__(self, config: dict) -> None:
        self.config = config

    @abstractmethod
    def route(self, ctx: RoutingContext) -> RoutingDecision:
        """
        Must never raise. Catch all exceptions internally.
        On any error, call self._passthrough(ctx, reason) and return it.
        """
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def _passthrough(self, ctx: RoutingContext, reason: str) -> RoutingDecision:
        from ..schemas import RoutingDecision
        return RoutingDecision(
            model=ctx.registry.primary,
            mutator="noop",
            reason=reason,
            confidence=1.0,
            router_name=self.name,
        )
```

## RoutingContext (routing/context.py)

Re-export from schemas for convenience:

```python
from ..schemas import RoutingContext
__all__ = ["RoutingContext"]
```

## PassthroughRouter (routing/passthrough.py)

```python
from . import BaseRouter
from .context import RoutingContext
from ..schemas import RoutingDecision


class PassthroughRouter(BaseRouter):
    def route(self, ctx: RoutingContext) -> RoutingDecision:
        return self._passthrough(ctx, "passthrough: primary model, no routing applied")
```

## RuleBasedRouter (routing/rule_based.py)

**9 rules, first match wins:**

```
Rule 1: All model names are in ctx.models_tried
         → oracle, noop, "all tiers exhausted"

Rule 2: ctx.last_error_type == "rate_limit" AND oracle.name in ctx.models_tried
         → reasoning tier, thinking_mode, "oracle rate-limited, step down"

Rule 3: features.difficulty == "hard" OR features.token_count > 8000
         → oracle, noop, "hard/long context → oracle"

Rule 4: features.task_type == "reason" AND features.difficulty != "easy"
         → reasoning tier, thinking_mode, "non-trivial reasoning → Qwen3 thinking"

Rule 5: features.task_type == "code" AND features.difficulty == "hard"
         → oracle, noop, "hard code → oracle"

Rule 6: features.task_type in {"extract", "summarize"} AND features.token_count < 3000
         → fast tier, noop, "simple extract/summarize → fast"

Rule 7: features.has_tools AND features.task_type == "tool_call"
         → reasoning tier, noop, "tool call → reasoning"

Rule 8: features.language == "he"
         → reasoning tier, noop, "hebrew content → reasoning"

Rule 9: default
         → fast tier, noop, "default → fast", confidence=0.7
```

Implementation pattern:

```python
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
```

## DecisionTreeRouter (routing/decision_tree.py)

```python
import logging
import joblib
from . import BaseRouter
from .rule_based import RuleBasedRouter
from .context import RoutingContext
from ..schemas import RoutingDecision


class DecisionTreeRouter(BaseRouter):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        pkl_path = config.get("policy_path", "routing_policy.pkl")
        self._fallback = RuleBasedRouter(config)
        try:
            data          = joblib.load(pkl_path)
            self.clf      = data["clf"]
            self.cat_enc  = data["cat_enc"]
            self.lvl_enc  = data["lvl_enc"]
            self._ready   = True
            logging.info(f"[optmod] DecisionTreeRouter loaded: {pkl_path}")
        except FileNotFoundError:
            self._ready = False
            logging.warning(f"[optmod] {pkl_path} not found, using RuleBasedRouter fallback")

    def route(self, ctx: RoutingContext) -> RoutingDecision:
        if not self._ready:
            return self._fallback.route(ctx)
        try:
            f = ctx.features
            X = [[
                self.cat_enc.transform([f.task_type])[0],
                self.lvl_enc.transform([f.difficulty])[0],
                f.token_count,
            ]]
            label  = self.clf.predict(X)[0]
            model  = ctx.registry.get_or_default(label)
            mutator = "thinking_mode" if model.thinking_mode else "noop"
            return RoutingDecision(
                model=model, mutator=mutator,
                reason=f"decision_tree: {label}",
                confidence=0.85, router_name=self.name,
            )
        except Exception as e:
            logging.warning(f"[optmod] DecisionTreeRouter error: {e}, falling back")
            return self._fallback.route(ctx)
```

## Router factory (used in main.py)

```python
def build_router(name: str, config: dict) -> BaseRouter:
    from .routing.passthrough    import PassthroughRouter
    from .routing.rule_based     import RuleBasedRouter
    from .routing.decision_tree  import DecisionTreeRouter

    mapping = {
        "passthrough":    PassthroughRouter,
        "rule_based":     RuleBasedRouter,
        "decision_tree":  DecisionTreeRouter,
    }
    cls = mapping.get(name, PassthroughRouter)
    return cls(config)
```
