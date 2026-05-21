import pytest
from fastapi.testclient import TestClient
from optmod.registry import ModelConfig, ModelRegistry
from optmod.schemas import OpenAIChatRequest, ChatMessage


@pytest.fixture
def registry():
    models = [
        ModelConfig(name="qwen2.5:7b",  provider="ollama",   base_url="http://localhost:11434/v1", api_key="x", tier=0, tier_name="fast",      context_window=32768,  cost_per_1k=0.0,    supports_tools=True, thinking_mode=False, thinking_default=False),
        ModelConfig(name="qwen3:8b",    provider="ollama",   base_url="http://localhost:11434/v1", api_key="x", tier=1, tier_name="reasoning", context_window=32768,  cost_per_1k=0.0,    supports_tools=True, thinking_mode=True,  thinking_default=False),
        ModelConfig(name="deepseek-v4", provider="deepseek", base_url="http://deepseek/v1",        api_key="x", tier=2, tier_name="oracle",    context_window=128000, cost_per_1k=0.0014, supports_tools=True, thinking_mode=False, thinking_default=False),
    ]
    return ModelRegistry(models, "deepseek-v4")


@pytest.fixture
def simple_req():
    return OpenAIChatRequest(messages=[
        ChatMessage(role="user", content="summarize this document")
    ])
