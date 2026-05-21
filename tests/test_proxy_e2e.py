import pytest
import respx
import httpx
from fastapi.testclient import TestClient
from optmod.main import app

OLLAMA_FAST   = "http://localhost:11434/v1"
OLLAMA_REASON = "http://localhost:11434/v1"
DEEPSEEK      = "https://api.deepseek.com/v1"

OK_RESPONSE = {
    "id": "test", "object": "chat.completion", "created": 1,
    "model": "test",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@respx.mock
def test_simple_route_succeeds(client):
    respx.post(f"{OLLAMA_FAST}/chat/completions").mock(
        return_value=httpx.Response(200, json=OK_RESPONSE)
    )
    r = client.post("/v1/chat/completions", json={
        "model": "optmod",
        "messages": [{"role": "user", "content": "summarize this briefly"}],
    })
    assert r.status_code == 200


@respx.mock
def test_escalation_on_429(client):
    """Fast model returns 429; proxy escalates and succeeds on second call."""
    respx.post(f"{OLLAMA_FAST}/chat/completions").mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(200, json=OK_RESPONSE),
        ]
    )
    r = client.post("/v1/chat/completions", json={
        "model": "optmod",
        "messages": [{"role": "user", "content": "test"}],
    })
    assert r.status_code == 200


@respx.mock
def test_all_models_fail_returns_502(client):
    """All models fail → 502."""
    respx.post(f"{OLLAMA_FAST}/chat/completions").mock(
        return_value=httpx.Response(500)
    )
    respx.post(f"{DEEPSEEK}/chat/completions").mock(
        return_value=httpx.Response(500)
    )
    r = client.post("/v1/chat/completions", json={
        "model": "optmod",
        "messages": [{"role": "user", "content": "test"}],
    })
    assert r.status_code == 502


@respx.mock
def test_auth_error_no_escalation(client):
    """401 is non-retryable → 502 immediately, no escalation."""
    respx.post(f"{OLLAMA_FAST}/chat/completions").mock(
        return_value=httpx.Response(401)
    )
    r = client.post("/v1/chat/completions", json={
        "model": "optmod",
        "messages": [{"role": "user", "content": "test"}],
    })
    assert r.status_code == 502
