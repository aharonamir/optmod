import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .registry import ModelConfig


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

    def dict(self) -> dict:
        return {
            "router":        self.router,
            "primary_model": self.primary_model,
            "policy_path":   self.policy_path,
            "log_path":      self.log_path,
        }


def load_config(path: str = "config.yaml") -> Config:
    raw = yaml.safe_load(Path(path).read_text())

    esc = EscalationConfig(
        max_escalations=raw.get("escalation", {}).get("max_escalations", 2)
    )

    models = []
    for m in raw.get("models", []):
        api_key_env = m.get("api_key_env")
        api_key = os.environ.get(api_key_env, "") if api_key_env else m.get("api_key", "")
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
        ))

    return Config(
        router=        raw.get("router", "passthrough"),
        primary_model= raw["primary_model"],
        policy_path=   raw.get("policy_path", "routing_policy.pkl"),
        log_path=      raw.get("log_path", "routing.log.jsonl"),
        escalation=    esc,
        models=        models,
    )
