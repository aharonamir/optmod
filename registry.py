from dataclasses import dataclass


@dataclass
class ModelConfig:
    name:             str
    provider:         str
    base_url:         str
    api_key:          str
    tier:             int
    tier_name:        str
    context_window:   int
    cost_per_1k:      float
    supports_tools:   bool
    thinking_mode:    bool
    thinking_default: bool
    timeout_s:        float = 120.0
    supports_vision:  bool  = False


class ModelRegistry:
    def __init__(self, models: list[ModelConfig], primary_name: str) -> None:
        self._models  = {m.name: m for m in models}
        self._primary = primary_name

    @property
    def primary(self) -> ModelConfig:
        return self._models[self._primary]

    def get(self, name: str) -> ModelConfig:
        return self._models[name]

    def get_or_default(self, name: str) -> ModelConfig:
        return self._models.get(name, self.primary)

    def all(self) -> list[ModelConfig]:
        return sorted(self._models.values(), key=lambda m: m.tier)

    def by_tier(self, tier: int) -> list[ModelConfig]:
        return sorted(
            [m for m in self._models.values() if m.tier == tier],
            key=lambda m: m.cost_per_1k,
        )

    def next_tier_up(self, current: ModelConfig) -> ModelConfig | None:
        candidates = [m for m in self._models.values() if m.tier == current.tier + 1]
        if not candidates:
            return None
        return min(candidates, key=lambda m: m.cost_per_1k)

    def excluding(self, names: list[str]) -> list[ModelConfig]:
        return [m for m in self._models.values() if m.name not in names]
