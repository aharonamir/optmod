import logging
import os
from dataclasses import dataclass

_log = logging.getLogger(__name__)


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
    api_model_name:   str   = ""  # name sent to the provider API; falls back to name if empty


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


class ConfigError(ValueError):
    pass


def _assign_tier(model_dict: dict, override: dict) -> tuple[int, str]:
    if "tier" in override:
        tier = int(override["tier"])
        tier_name = override.get("tier_name") or _tier_name_for(tier)
        return tier, tier_name

    if model_dict.get("free", False):
        return 0, "fast"

    price_in  = model_dict.get("price_input_per_1M") or 0.0
    price_out = model_dict.get("price_output_per_1M") or 0.0
    blended   = (3 * price_in + price_out) / 4

    if blended < 0.50:
        return 1, "reasoning"
    return 2, "oracle"


def _tier_name_for(tier: int) -> str:
    return {0: "fast", 1: "reasoning", 2: "oracle"}.get(tier, "oracle")


def build_models_from_perfrouter(
    perfrouter_models: list[dict],
    providers: dict,
    curated_ids: list[str] | None,
    model_overrides: dict,
) -> list[ModelConfig]:
    """
    Build ModelConfig list by joining perfrouter's model universe with optmod's provider config.

    curated_ids=None means use all models in the perfrouter universe.
    Raises ConfigError if any curated_id is not found in the perfrouter universe.
    """
    universe = {m["id"]: m for m in perfrouter_models}

    if curated_ids is not None:
        missing = [mid for mid in curated_ids if mid not in universe]
        if missing:
            raise ConfigError(f"curated_models references unknown IDs: {missing}")
        work_ids = curated_ids
    else:
        work_ids = list(universe.keys())

    models = []
    for model_id in work_ids:
        m        = universe[model_id]
        override = model_overrides.get(model_id, {})

        # Resolve provider: use the model's native provider if its API key is set,
        # otherwise fall back to OpenRouter (which proxies all major providers).
        native_provider = m.get("provider", "openrouter")
        used_provider, prov, api_key = _pick_provider(
            model_id, native_provider, providers, override
        )
        base_url  = prov["base_url"].rstrip("/")
        timeout_s = float(override.get("timeout_s") or prov.get("timeout_s", 120.0))

        # Context window: prefer effective_context_k, fall back to context_window_k
        eff_ctx_k      = m.get("effective_context_k") or m.get("context_window_k")
        context_window = int(eff_ctx_k * 1000) if eff_ctx_k else 128000

        # Blended cost per 1k (3:1 input/output ratio)
        price_in       = m.get("price_input_per_1M") or 0.0
        price_out      = m.get("price_output_per_1M") or 0.0
        blended_per_1M = (3 * price_in + price_out) / 4
        cost_per_1k    = override.get("cost_per_1k", blended_per_1M / 1000.0)

        tier, tier_name = _assign_tier(m, override)

        thinking_mode    = bool(override.get("thinking_mode",    m.get("has_thinking_mode", False)))
        thinking_default = bool(override.get("thinking_default", False))
        supports_tools   = bool(override.get("supports_tools",   m.get("supports_tools", True)))
        supports_vision  = bool(override.get("supports_vision",  m.get("supports_vision", False)))

        # OpenRouter expects the full org/model id; direct providers want just the local name
        if used_provider == "openrouter":
            api_model_name = override.get("api_model_name", model_id)
        else:
            api_model_name = override.get("api_model_name") or model_id.split("/")[-1]

        models.append(ModelConfig(
            name=             model_id,
            provider=         used_provider,
            base_url=         base_url,
            api_key=          api_key,
            tier=             tier,
            tier_name=        tier_name,
            context_window=   context_window,
            cost_per_1k=      cost_per_1k,
            supports_tools=   supports_tools,
            thinking_mode=    thinking_mode,
            thinking_default= thinking_default,
            timeout_s=        timeout_s,
            supports_vision=  supports_vision,
            api_model_name=   api_model_name,
        ))

    return models


def _pick_provider(
    model_id: str,
    native_provider: str,
    providers: dict,
    override: dict,
) -> tuple[str, dict, str]:
    """
    Return (used_provider_name, provider_config_dict, api_key).

    Uses the model's native provider if its API key env var is set and non-empty.
    Falls back to openrouter otherwise (which proxies all major providers).
    Raises ConfigError only if neither native nor openrouter can supply a key.
    """
    api_key_env_override = override.get("api_key_env")

    # Try native provider
    if native_provider in providers:
        prov = providers[native_provider]
        key_env = api_key_env_override or prov.get("api_key_env", "")
        key     = os.environ.get(key_env, "") if key_env else ""
        if key:
            return native_provider, prov, key

    # Fall back to openrouter
    if native_provider != "openrouter" and "openrouter" in providers:
        or_prov = providers["openrouter"]
        or_env  = or_prov.get("api_key_env", "")
        or_key  = os.environ.get(or_env, "") if or_env else ""
        if or_key:
            _log.debug(
                "Model '%s': no key for provider '%s', routing via openrouter",
                model_id, native_provider,
            )
            return "openrouter", or_prov, or_key

    # Neither has a key — still configure the native provider so the model
    # appears in the registry; the forwarder will get a 401 at call time.
    if native_provider in providers:
        prov    = providers[native_provider]
        key_env = api_key_env_override or prov.get("api_key_env", "")
        key     = os.environ.get(key_env, "") if key_env else ""
        _log.warning(
            "Model '%s': provider '%s' has no API key configured "
            "(env var '%s' is unset) and openrouter is also unavailable.",
            model_id, native_provider, key_env,
        )
        return native_provider, prov, key

    raise ConfigError(
        f"Model '{model_id}': native provider '{native_provider}' is not configured "
        f"in providers and 'openrouter' is also absent. "
        f"Configured providers: {list(providers.keys())}"
    )
