# 10 — Test Plan (tests/)

## Setup (tests/conftest.py)

```python
import pytest
from fastapi.testclient import TestClient
from optmod.main import app
from optmod.registry import ModelConfig, ModelRegistry
from optmod.schemas import OpenAIChatRequest, ChatMessage

@pytest.fixture
def registry():
    models = [
        ModelConfig(name="qwen2.5:7b",  provider="ollama",   base_url="http://ollama/v1",    api_key="x", tier=0, tier_name="fast",      context_window=32768, cost_per_1k=0.0,    supports_tools=True,  thinking_mode=False, thinking_default=False),
        ModelConfig(name="qwen3:8b",    provider="ollama",   base_url="http://ollama/v1",    api_key="x", tier=1, tier_name="reasoning", context_window=32768, cost_per_1k=0.0,    supports_tools=True,  thinking_mode=True,  thinking_default=False),
        ModelConfig(name="deepseek-v4", provider="deepseek", base_url="http://deepseek/v1",  api_key="x", tier=2, tier_name="oracle",    context_window=128000,cost_per_1k=0.0014, supports_tools=True,  thinking_mode=False, thinking_default=False),
    ]
    return ModelRegistry(models, "deepseek-v4")

@pytest.fixture
def simple_req():
    return OpenAIChatRequest(messages=[
        ChatMessage(role="user", content="summarize this document")
    ])
```

## test_features.py — 10 required test cases

```python
from optmod.features import FeatureExtractor
from optmod.schemas import OpenAIChatRequest, ChatMessage

ext = FeatureExtractor()

def msg(text): return OpenAIChatRequest(messages=[ChatMessage(role="user", content=text)])

def test_reason():          assert ext.extract(msg("why does this algorithm fail?")).task_type == "reason"
def test_code():            assert ext.extract(msg("implement a binary search tree")).task_type == "code"
def test_extract():         assert ext.extract(msg("extract all emails from this document")).task_type == "extract"
def test_summarize():       assert ext.extract(msg("summarize this PDF in 3 bullet points")).task_type == "summarize"
def test_tool_call():       assert ext.extract(msg("call the get_weather function")).task_type == "tool_call"
def test_search():          assert ext.extract(msg("search for recent papers on LLM routing")).task_type == "search"
def test_hard_difficulty():
    r = ext.extract(msg("debug this undocumented multi-round agent pipeline"))
    assert r.difficulty == "hard"
def test_medium_difficulty():
    r = ext.extract(msg("build a multi-step pipeline for data extraction"))
    assert r.difficulty == "medium"
def test_hebrew():
    r = ext.extract(msg("שלום, תסכם את המסמך הזה"))
    assert r.language == "he"
def test_has_tools():
    req = OpenAIChatRequest(
        messages=[ChatMessage(role="user", content="hi")],
        tools=[{"type": "function", "function": {"name": "test"}}]
    )
    assert ext.extract(req).has_tools is True
def test_token_estimate():
    # 10 words → estimate ~13
    r = ext.extract(msg("one two three four five six seven eight nine ten"))
    assert 10 <= r.token_count <= 20
```

## test_routers.py — 9 required test cases (one per rule)

```python
from optmod.routing.rule_based import RuleBasedRouter
from optmod.schemas import RoutingContext, Features, OpenAIChatRequest, ChatMessage

def make_ctx(registry, task="general", difficulty="easy", tokens=500,
             has_tools=False, language="en", tried=None, last_error=None):
    req = OpenAIChatRequest(messages=[ChatMessage(role="user", content="test")])
    features = Features(task_type=task, difficulty=difficulty, token_count=tokens,
                        has_tools=has_tools, language=language, last_user_message="test")
    return RoutingContext(request=req, features=features, session_id="test",
                          registry=registry, models_tried=tried or [],
                          last_error_type=last_error)

router = RuleBasedRouter({})

def test_rule1_all_tried(registry):
    ctx = make_ctx(registry, tried=["qwen2.5:7b","qwen3:8b","deepseek-v4"])
    assert router.route(ctx).model.name == "deepseek-v4"

def test_rule2_rate_limit(registry):
    ctx = make_ctx(registry, tried=["deepseek-v4"], last_error="rate_limit")
    d = router.route(ctx)
    assert d.model.tier == 1
    assert d.mutator == "thinking_mode"

def test_rule3_hard(registry):
    ctx = make_ctx(registry, difficulty="hard")
    assert router.route(ctx).model.name == "deepseek-v4"

def test_rule3_long_context(registry):
    ctx = make_ctx(registry, tokens=9000)
    assert router.route(ctx).model.name == "deepseek-v4"

def test_rule4_nontrivial_reason(registry):
    ctx = make_ctx(registry, task="reason", difficulty="medium")
    d = router.route(ctx)
    assert d.model.tier == 1
    assert d.mutator == "thinking_mode"

def test_rule5_hard_code(registry):
    ctx = make_ctx(registry, task="code", difficulty="hard")
    assert router.route(ctx).model.name == "deepseek-v4"

def test_rule6_simple_extract(registry):
    ctx = make_ctx(registry, task="extract", tokens=1000)
    assert router.route(ctx).model.tier == 0

def test_rule7_tool_call(registry):
    ctx = make_ctx(registry, task="tool_call", has_tools=True)
    assert router.route(ctx).model.tier == 1

def test_rule8_hebrew(registry):
    ctx = make_ctx(registry, language="he")
    assert router.route(ctx).model.tier == 1

def test_rule9_default(registry):
    ctx = make_ctx(registry)   # no conditions match
    assert router.route(ctx).model.tier == 0
```

## test_escalation.py — 6 required test cases

```python
from optmod.escalation import EscalationPolicy
from optmod.schemas import RoutingContext, Features, OpenAIChatRequest, ChatMessage

policy = EscalationPolicy(max_escalations=2)

# Use conftest registry fixture

def test_escalate_on_rate_limit(registry):
    ctx = _ctx(registry, tried=["qwen2.5:7b"], error="rate_limit")
    assert policy.should_escalate("rate_limit", 0, ctx) is True

def test_no_escalate_auth(registry):
    ctx = _ctx(registry, tried=["qwen2.5:7b"], error="auth")
    assert policy.should_escalate("auth", 0, ctx) is False

def test_no_escalate_max_attempts(registry):
    ctx = _ctx(registry, tried=["qwen2.5:7b","qwen3:8b"])
    assert policy.should_escalate("server", 2, ctx) is False

def test_no_escalate_at_top(registry):
    ctx = _ctx(registry, tried=["deepseek-v4"], error="server")
    assert policy.should_escalate("server", 0, ctx) is False

def test_next_model_fast_to_reasoning(registry):
    ctx = _ctx(registry, tried=["qwen2.5:7b"])
    assert policy.next_model(ctx).tier == 1

def test_next_model_reasoning_to_oracle(registry):
    ctx = _ctx(registry, tried=["qwen2.5:7b","qwen3:8b"])
    assert policy.next_model(ctx).tier == 2

def _ctx(registry, tried, error=None):
    req = OpenAIChatRequest(messages=[ChatMessage(role="user", content="test")])
    f   = Features("general","easy",100,False,"en","test")
    return RoutingContext(req, f, "s", registry, models_tried=tried, last_error_type=error)
```

## test_proxy_e2e.py — 4 required test cases using respx

```python
import pytest
import respx
import httpx
from fastapi.testclient import TestClient
from optmod.main import app

client = TestClient(app)

OLLAMA_FAST    = "http://localhost:11434/v1"
OLLAMA_REASON  = "http://localhost:11434/v1"
DEEPSEEK       = "https://api.deepseek.com/v1"

OK_RESPONSE = {
    "id": "test", "object": "chat.completion", "created": 1,
    "model": "test", "choices": [{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
    "usage": {"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}
}

@respx.mock
def test_simple_route_succeeds():
    respx.post(f"{OLLAMA_FAST}/chat/completions").mock(return_value=httpx.Response(200, json=OK_RESPONSE))
    r = client.post("/v1/chat/completions", json={
        "model": "optmod", "messages": [{"role":"user","content":"summarize this briefly"}]
    })
    assert r.status_code == 200

@respx.mock
def test_escalation_on_429():
    """Fast model returns 429, proxy escalates to reasoning, succeeds."""
    respx.post(f"{OLLAMA_FAST}/chat/completions").mock(return_value=httpx.Response(429))
    respx.post(f"{OLLAMA_REASON}/chat/completions").mock(return_value=httpx.Response(200, json=OK_RESPONSE))
    r = client.post("/v1/chat/completions", json={
        "model": "optmod", "messages": [{"role":"user","content":"test"}]
    })
    assert r.status_code == 200

@respx.mock
def test_all_models_fail_returns_502():
    """All models fail → 502."""
    respx.post(f"{OLLAMA_FAST}/chat/completions").mock(return_value=httpx.Response(500))
    respx.post(f"{OLLAMA_REASON}/chat/completions").mock(return_value=httpx.Response(500))
    respx.post(f"{DEEPSEEK}/chat/completions").mock(return_value=httpx.Response(500))
    r = client.post("/v1/chat/completions", json={
        "model": "optmod", "messages": [{"role":"user","content":"test"}]
    })
    assert r.status_code == 502

@respx.mock
def test_auth_error_no_escalation():
    """401 is non-retryable → 502 immediately, no escalation."""
    respx.post(f"{OLLAMA_FAST}/chat/completions").mock(return_value=httpx.Response(401))
    r = client.post("/v1/chat/completions", json={
        "model": "optmod", "messages": [{"role":"user","content":"test"}]
    })
    assert r.status_code == 502
```
