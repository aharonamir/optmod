import os

import pytest
import respx
import httpx
from fastapi.testclient import TestClient
from optmod.main import app

# All provider base URLs that optmod might route to
_PROVIDERS = [
    "https://openrouter.ai/api/v1",
    "https://api.deepseek.com",
    "https://generativelanguage.googleapis.com/v1beta/openai",
    "https://api.anthropic.com/v1",
    "https://api.openai.com/v1",
    "https://api.moonshot.cn/v1",
]

OK_RESPONSE = {
    "id": "test", "object": "chat.completion", "created": 1,
    "model": "test",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


_KEY_ENVS = ["OPENROUTER_API_KEY", "DEEPSEEK_API_KEY", "GOOGLE_API_KEY",
             "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "MOONSHOT_API_KEY"]


@pytest.fixture(scope="module")
def client():
    # Models whose native provider is not in config.yaml fall back to OpenRouter,
    # which needs a key at startup. Dummy keys keep the suite independent of .env;
    # real keys, if present, are left alone. Nothing here touches the network.
    mp = pytest.MonkeyPatch()
    for env in _KEY_ENVS:
        if not os.environ.get(env):
            mp.setenv(env, "test-key")
    try:
        with TestClient(app) as c:
            yield c
    finally:
        mp.undo()


@respx.mock
def test_simple_route_succeeds(client):
    for p in _PROVIDERS:
        respx.post(f"{p}/chat/completions").mock(
            return_value=httpx.Response(200, json=OK_RESPONSE)
        )
    r = client.post("/v1/chat/completions", json={
        "model": "optmod",
        "messages": [{"role": "user", "content": "summarize this briefly"}],
    })
    assert r.status_code == 200


@respx.mock
def test_escalation_on_429(client):
    """First chosen model returns 429; proxy escalates and the next attempt succeeds."""
    counter = {"n": 0}

    def respond(request):
        counter["n"] += 1
        if counter["n"] == 1:
            return httpx.Response(429)
        return httpx.Response(200, json=OK_RESPONSE)

    for p in _PROVIDERS:
        respx.post(f"{p}/chat/completions").mock(side_effect=respond)

    r = client.post("/v1/chat/completions", json={
        "model": "optmod",
        "messages": [{"role": "user", "content": "test"}],
    })
    assert r.status_code == 200


@respx.mock
def test_all_models_fail_returns_502(client):
    """All models fail → 502."""
    for p in _PROVIDERS:
        respx.post(f"{p}/chat/completions").mock(
            return_value=httpx.Response(500)
        )
    r = client.post("/v1/chat/completions", json={
        "model": "optmod",
        "messages": [{"role": "user", "content": "test"}],
    })
    assert r.status_code == 502


@respx.mock
def test_auth_error_all_models_fail_returns_502(client):
    """401 on every model → escalates, then 502."""
    for p in _PROVIDERS:
        respx.post(f"{p}/chat/completions").mock(
            return_value=httpx.Response(401)
        )
    r = client.post("/v1/chat/completions", json={
        "model": "optmod",
        "messages": [{"role": "user", "content": "test"}],
    })
    assert r.status_code == 502


@respx.mock
def test_retry_never_resends_to_failed_model(client):
    """A retryable failure must move to a different model, even if the router
    would pick the same one again (e.g. the cheapest-model fallback)."""
    import json

    sent: list[str] = []

    def respond(request):
        sent.append(json.loads(request.content)["model"])
        if len(sent) == 1:
            return httpx.Response(404, json={"error": {"message": "gone"}})
        return httpx.Response(200, json=OK_RESPONSE)

    for p in _PROVIDERS:
        respx.post(f"{p}/chat/completions").mock(side_effect=respond)

    r = client.post("/v1/chat/completions", json={
        "model": "optmod",
        "messages": [{"role": "user", "content": "retry-distinct-model probe, unique text 7f3a"}],
    })
    assert r.status_code == 200
    assert len(sent) == 2
    assert sent[0] != sent[1]


@respx.mock
def test_failed_model_is_reported_unhealthy_and_status_lists_it(client):
    """A 404 marks the model in /optmod/status so later requests skip it."""
    import json

    sent: list[str] = []

    def respond(request):
        sent.append(json.loads(request.content)["model"])
        if len(sent) == 1:
            return httpx.Response(404, json={"error": {"message": "gone"}})
        return httpx.Response(200, json=OK_RESPONSE)

    for p in _PROVIDERS:
        respx.post(f"{p}/chat/completions").mock(side_effect=respond)

    r = client.post("/v1/chat/completions", json={
        "model": "optmod",
        "messages": [{"role": "user", "content": "health-mark probe, unique text 91bc"}],
    })
    assert r.status_code == 200
    health = client.get("/optmod/status").json()["model_health"]
    assert any(v["error_type"] == "not_found" and v["remaining_s"] > 3000
               for v in health.values())

    # A similar request routes to the same model; it must be swapped out, not retried.
    failed_model = sent[0]
    sent.clear()
    r = client.post("/v1/chat/completions", json={
        "model": "optmod",
        "messages": [{"role": "user", "content": "health-mark probe, unique text 91bd"}],
    })
    assert r.status_code == 200
    assert failed_model not in sent


@respx.mock
def test_auth_error_escalates_to_another_model(client):
    """A 401 from one model is per-model: the next attempt goes elsewhere and succeeds."""
    import json

    sent: list[str] = []

    def respond(request):
        sent.append(json.loads(request.content)["model"])
        if len(sent) == 1:
            return httpx.Response(401, json={"error": {"message": "bad key"}})
        return httpx.Response(200, json=OK_RESPONSE)

    for p in _PROVIDERS:
        respx.post(f"{p}/chat/completions").mock(side_effect=respond)

    r = client.post("/v1/chat/completions", json={
        "model": "optmod",
        "messages": [{"role": "user", "content": "auth-escalation probe, unique text c40e"}],
    })
    assert r.status_code == 200
    assert len(sent) == 2 and sent[0] != sent[1]


def test_status_reports_dashboard_baseline_model(client):
    """Dashboard 'saved vs always-<model>' baseline defaults to the primary model."""
    st = client.get("/optmod/status").json()
    assert st["baseline_model"] == st["primary"]
