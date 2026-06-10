import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from optmod.registry import ModelConfig, build_models_from_perfrouter, ConfigError

_log = logging.getLogger(__name__)


@dataclass
class EscalationConfig:
    max_escalations: int = 2


@dataclass
class Config:
    router:        str
    primary_model: str
    policy_path:   str
    log_path:      str
    escalation:    EscalationConfig
    models:        list[ModelConfig]
    _raw:          dict = field(default_factory=dict)

    def dict(self) -> dict:
        return {
            **self._raw,
            "router":        self.router,
            "primary_model": self.primary_model,
            "policy_path":   self.policy_path,
            "log_path":      self.log_path,
        }


def load_config(path: str = "config.yaml") -> Config:
    raw      = yaml.safe_load(Path(path).read_text())
    config_dir = Path(path).parent

    esc = EscalationConfig(
        max_escalations=raw.get("escalation", {}).get("max_escalations", 2)
    )

    if "models" in raw:
        models = _load_models_legacy(raw)
    else:
        models = _load_models_from_perfrouter(raw, config_dir)

    return Config(
        router=        raw.get("router", "passthrough"),
        primary_model= raw["primary_model"],
        policy_path=   raw.get("policy_path", "routing_policy.pkl"),
        log_path=      raw.get("log_path", "routing.log.jsonl"),
        escalation=    esc,
        models=        models,
        _raw=          raw,
    )


def _load_models_legacy(raw: dict) -> list[ModelConfig]:
    models = []
    for m in raw.get("models", []):
        api_key_env = m.get("api_key_env")
        api_key     = os.environ.get(api_key_env, "") if api_key_env else m.get("api_key", "")
        models.append(ModelConfig(
            name=             m["name"],
            provider=         m["provider"],
            base_url=         m["base_url"].rstrip("/"),
            api_key=          api_key,
            tier=             m["tier"],
            tier_name=        m["tier_name"],
            context_window=   m["context_window"],
            cost_per_1k=      m["cost_per_1k"],
            supports_tools=   m.get("supports_tools", True),
            thinking_mode=    m.get("thinking_mode", False),
            thinking_default= m.get("thinking_default", False),
            timeout_s=        m.get("timeout_s", 120.0),
            supports_vision=  m.get("supports_vision", False),
        ))
    return models


def _load_models_from_perfrouter(raw: dict, config_dir: Path) -> list[ModelConfig]:
    perf_cfg = raw.get("perf_router", {})

    # Log deprecation warning if only flat keys present
    flat_keys = [k for k in raw if k.startswith("perf_router_")]
    if flat_keys and not perf_cfg:
        _log.warning(
            "[optmod] Deprecated flat config keys in use: %s. "
            "Migrate to the nested perf_router: block.",
            flat_keys,
        )

    perfrouter_models_yaml = perf_cfg.get("perfrouter_models_yaml", "../perfrouter/models.yaml")
    models_yaml_path       = (config_dir / perfrouter_models_yaml).resolve()

    perfrouter_raw    = yaml.safe_load(models_yaml_path.read_text())
    perfrouter_models = perfrouter_raw.get("models", [])

    providers       = raw.get("providers", {})
    model_overrides = raw.get("model_overrides", {})
    model_set       = perf_cfg.get("model_set", "curated")
    curated_ids     = raw.get("curated_models") if model_set == "curated" else None

    return build_models_from_perfrouter(
        perfrouter_models=perfrouter_models,
        providers=providers,
        curated_ids=curated_ids,
        model_overrides=model_overrides,
    )
