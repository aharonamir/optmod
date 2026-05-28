# 05 — Context Mutators (mutators/)

## BaseContextMutator (mutators/__init__.py)

```python
from abc import ABC, abstractmethod
from ..schemas import ChatMessage, RoutingDecision


class BaseContextMutator(ABC):
    @abstractmethod
    def mutate(
        self,
        messages: list[ChatMessage],
        decision: RoutingDecision,
    ) -> list[ChatMessage]:
        """
        MUST return a new list. NEVER modify messages in-place.
        If no mutation needed, return the original list reference (not a copy).
        """
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__
```

## NoopMutator (mutators/noop.py)

```python
from . import BaseContextMutator
from ..schemas import ChatMessage, RoutingDecision


class NoopMutator(BaseContextMutator):
    def mutate(
        self,
        messages: list[ChatMessage],
        decision: RoutingDecision,
    ) -> list[ChatMessage]:
        return messages   # unchanged reference, not a copy
```

## ThinkingModeMutator (mutators/thinking_mode.py)

Prepends `/think` or `/no_think` to the last user message.
Only activates if the target model has `thinking_mode=True`.

```python
from . import BaseContextMutator
from ..schemas import ChatMessage, RoutingDecision


class ThinkingModeMutator(BaseContextMutator):
    def mutate(
        self,
        messages: list[ChatMessage],
        decision: RoutingDecision,
    ) -> list[ChatMessage]:
        # Only applies to thinking-capable models
        if not decision.model.thinking_mode:
            return messages

        # /think for reasoning, /no_think for everything else
        prefix = "/think" if decision.mutator == "thinking_mode" else "/no_think"

        # Find last user message and prepend
        result = list(messages)   # shallow copy of list
        for i in range(len(result) - 1, -1, -1):
            if result[i].role == "user" and isinstance(result[i].content, str):
                original = result[i].content
                result[i] = result[i].model_copy(
                    update={"content": f"{prefix}\n{original}"}
                )
                break
        return result
```

## Mutator registry (instantiated in main.py lifespan)

```python
mutators: dict[str, BaseContextMutator] = {
    "noop":          NoopMutator(),
    "thinking_mode": ThinkingModeMutator(),
}
```

The key in this dict matches the `mutator` field in `RoutingDecision`.
