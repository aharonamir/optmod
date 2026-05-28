# 02 — Model Registry (registry.py)

## ModelConfig

```python
from dataclasses import dataclass
import os

@dataclass
class ModelConfig:
    name:             str
    provider:         str        # 'ollama' | 'deepseek' | 'openai'
    base_url:         str        # actual inference endpoint (no trailing slash)
    api_key:          str        # resolved from env at startup (see below)
    tier:             int        # 0=fast  1=reasoning  2=oracle
    tier_name:        str        # 'fast' | 'reasoning' | 'oracle'
    context_window:   int
    cost_per_1k:      float      # 0.0 for local
    supports_tools:   bool
    thinking_mode:    bool       # True = model supports /think toggle
    thinking_default: bool       # True = thinking on by default
    timeout_s:        float = 120.0
```

**API key resolution**: In `config.py`, when loading a ModelConfig, if the yaml
field `api_key_env` is set, resolve the key via `os.environ.get(api_key_env, "")`.
If `api_key_env` is null/missing, use literal `api_key` field or empty string.

## ModelRegistry

```python
class ModelRegistry:
    def __init__(self, models: list[ModelConfig], primary_name: str) -> None:
        self._models   = {m.name: m for m in models}
        self._primary  = primary_name

    @property
    def primary(self) -> ModelConfig:
        return self._models[self._primary]

    def get(self, name: str) -> ModelConfig:
        """Raise KeyError if not found."""
        return self._models[name]

    def get_or_default(self, name: str) -> ModelConfig:
        """Return primary if name not found."""
        return self._models.get(name, self.primary)

    def all(self) -> list[ModelConfig]:
        """All models, sorted by tier asc."""
        return sorted(self._models.values(), key=lambda m: m.tier)

    def by_tier(self, tier: int) -> list[ModelConfig]:
        """All models of a given tier, sorted by cost_per_1k asc."""
        return sorted(
            [m for m in self._models.values() if m.tier == tier],
            key=lambda m: m.cost_per_1k
        )

    def next_tier_up(self, current: ModelConfig) -> ModelConfig | None:
        """
        Return the cheapest model one tier above current.
        Return None if current is already at the highest tier.
        """
        candidates = [m for m in self._models.values() if m.tier == current.tier + 1]
        if not candidates:
            return None
        return min(candidates, key=lambda m: m.cost_per_1k)

    def excluding(self, names: list[str]) -> list[ModelConfig]:
        """All models whose name is not in the exclusion list."""
        return [m for m in self._models.values() if m.name not in names]
```

## Notes

- `by_tier(0)` → fast models. `by_tier(1)` → reasoning. `by_tier(2)` → oracle.
- `next_tier_up` is used by `EscalationPolicy.next_model()`.
- Registry is immutable after startup. No thread locking needed.
