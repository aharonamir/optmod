from optmod.escalation import EscalationPolicy
from optmod.schemas import RoutingContext, Features, OpenAIChatRequest, ChatMessage

policy = EscalationPolicy(max_escalations=2)


def _ctx(registry, tried, error=None):
    req = OpenAIChatRequest(messages=[ChatMessage(role="user", content="test")])
    f   = Features("general", "easy", 100, False, "en", "test")
    return RoutingContext(req, f, "s", registry, models_tried=tried, last_error_type=error)


def test_escalate_on_rate_limit(registry):
    ctx = _ctx(registry, tried=["qwen2.5:7b"], error="rate_limit")
    assert policy.should_escalate("rate_limit", 0, ctx) is True


def test_no_escalate_auth(registry):
    ctx = _ctx(registry, tried=["qwen2.5:7b"], error="auth")
    assert policy.should_escalate("auth", 0, ctx) is False


def test_no_escalate_max_attempts(registry):
    ctx = _ctx(registry, tried=["qwen2.5:7b", "qwen3:8b"])
    assert policy.should_escalate("server", 2, ctx) is False


def test_no_escalate_at_top(registry):
    ctx = _ctx(registry, tried=["deepseek-v4"], error="server")
    assert policy.should_escalate("server", 0, ctx) is False


def test_next_model_fast_to_reasoning(registry):
    ctx = _ctx(registry, tried=["qwen2.5:7b"])
    assert policy.next_model(ctx).tier == 1


def test_next_model_reasoning_to_oracle(registry):
    ctx = _ctx(registry, tried=["qwen2.5:7b", "qwen3:8b"])
    assert policy.next_model(ctx).tier == 2
