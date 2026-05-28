# 11 — Configuration (config.py + config.yaml)

## Config dataclasses (config.py)

```python
from dataclasses import dataclass, field
from pathlib import Path
import yaml, os
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
```

## config.yaml (the file to create in project root)

```yaml
# Active router: passthrough | rule_based | decision_tree
router: rule_based

# Oracle model — quality ceiling, used as primary
primary_model: deepseek-v4

# Path to WildClawBench-trained decision tree (created in Phase 2)
policy_path: routing_policy.pkl

# JSONL log path (relative to working directory)
log_path: routing.log.jsonl

escalation:
  max_escalations: 2

models:
  - name: qwen2.5:7b
    provider: ollama
    base_url: http://localhost:11434/v1
    api_key_env: null        # Ollama doesn't need a key
    tier: 0
    tier_name: fast
    context_window: 32768
    cost_per_1k: 0.0
    supports_tools: true
    thinking_mode: false
    thinking_default: false
    timeout_s: 60.0

  - name: qwen3:8b
    provider: ollama
    base_url: http://localhost:11434/v1
    api_key_env: null
    tier: 1
    tier_name: reasoning
    context_window: 32768
    cost_per_1k: 0.0
    supports_tools: true
    thinking_mode: true       # /think prefix enables thinking mode
    thinking_default: false   # off by default, optmod enables per task
    timeout_s: 120.0

  - name: deepseek-v4
    provider: deepseek
    base_url: https://api.deepseek.com/v1
    api_key_env: DEEPSEEK_API_KEY    # set this env var before running
    tier: 2
    tier_name: oracle
    context_window: 128000
    cost_per_1k: 0.0014
    supports_tools: true
    thinking_mode: false
    thinking_default: false
    timeout_s: 180.0
```

## pyproject.toml

```toml
[project]
name = "optmod"
version = "0.1.0"
requires-python = ">= 3.11"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "httpx>=0.27",
    "pydantic>=2.7",
    "pyyaml>=6.0",
    "scikit-learn>=1.5",
    "joblib>=1.4",
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-asyncio>=0.23",
    "respx>=0.21",
    "httpx[test]",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"
```
