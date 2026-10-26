"""Best-score escalation, provider-advertised TTLs and the token-estimate margin."""
import httpx
import pytest
import respx

from optmod.forwarder import ModelForwarder, retry_after_s
from optmod.model_health import ModelHealth
from optmod.registry import ModelConfig, ModelRegistry
from optmod.routing.perfrouter.router import PerfRouterRouter
from optmod.schemas import (
    ChatMessage, Features, OpenAIChatRequest, RoutingContext, RoutingDecision,
)


# ── retry_after_s ─────────────────────────────────────────────────────────────

NOW = 1_792_800_000.0     # epoch seconds


def test_reset_epoch_ms_header():
    assert retry_after_s({"X-RateLimit-Reset": str(int((NOW + 3600) * 1000))}, now=NOW) == pytest.approx(3600)


def test_reset_epoch_seconds_and_delta_headers():
    assert retry_after_s({"x-ratelimit-reset": str(int(NOW + 90))}, now=NOW) == pytest.approx(90)
    assert retry_after_s({"x-ratelimit-reset": "45"}, now=NOW) == 45


def test_retry_after_header_wins():
    h = {"Retry-After": "7", "X-RateLimit-Reset": str(int((NOW + 3600) * 1000))}
    assert retry_after_s(h, now=NOW) == 7


def test_openrouter_body_metadata_headers():
    import json
    body = json.dumps({"error": {"message": "Rate limit exceeded", "code": 429, "metadata": {
        "headers": {"X-RateLimit-Limit": "1000", "X-RateLimit-Reset": str(int((NOW + 600) * 1000))}}}})
    assert retry_after_s({}, body, now=NOW) == pytest.approx(600)


def test_nothing_usable_is_none_and_past_reset_is_zero():
    assert retry_after_s({}, "not json", now=NOW) is None
    assert retry_after_s({"Retry-After": "soon"}, now=NOW) is None
    assert retry_after_s({"X-RateLimit-Reset": str(int((NOW - 10) * 1000))}, now=NOW) == 0


@respx.mock
async def test_forward_reports_retry_after_through_info():
    respx.post("https://prov.test/v1/chat/completions").mock(
        return_value=httpx.Response(429, headers={"Retry-After": "120"}, json={"error": {}}))
    model = ModelConfig(name="m", provider="p", base_url="https://prov.test/v1", api_key="k",
                        tier=0, tier_name="fast", context_window=1000, cost_per_1k=0.0,
                        supports_tools=True, thinking_mode=False, thinking_default=False)
    info: dict = {}
    fwd = ModelForwarder()
    resp, err = await fwd.forward(model, OpenAIChatRequest(messages=[ChatMessage(role="user", content="x")]),
                                  [ChatMessage(role="user", content="x")], info)
    await fwd.close()
    assert (resp, err) == ({}, "rate_limit")      # error body stays empty so main returns 502
    assert info == {"retry_after_s": 120.0}


# ── ModelHealth uses the advertised reset ─────────────────────────────────────

class _Clock:
    t = 1000.0

    def __call__(self):
        return self.t


def test_advertised_reset_replaces_default_ttl_and_is_capped():
    c = _Clock()
    h = ModelHealth({"max_ttl_s": 5000}, clock=c)
    h.mark_failure("a", "rate_limit", retry_after_s=4000)     # default would be 30 s
    h.mark_failure("b", "rate_limit", retry_after_s=99999)    # capped
    h.mark_failure("c", "rate_limit", retry_after_s=0)        # already passed -> default
    c.t += 31
    assert h.is_bad("a") and h.is_bad("b") and not h.is_bad("c")
    c.t += 4970
    assert not h.is_bad("a") and not h.is_bad("b")


# ── EscalationPolicy without the tier requirement ─────────────────────────────

def test_escalate_at_top_tier_when_tier_not_required(registry):
    from optmod.escalation import EscalationPolicy
    f = Features("general", "easy", 100, False, "en", "t")
    ctx = RoutingContext(OpenAIChatRequest(messages=[ChatMessage(role="user", content="t")]),
                         f, "s", registry, models_tried=["deepseek-v4"])
    assert EscalationPolicy().should_escalate("server", 0, ctx) is False
    assert EscalationPolicy(require_higher_tier=False).should_escalate("server", 0, ctx) is True


# ── rank_models ───────────────────────────────────────────────────────────────

class _Inf:
    def classify_task(self, q):
        return [("code.debugging", 0.5)]

    def predict_quality(self, top):
        return {"qwen3:8b": 0.9, "deepseek-v4": 0.5, "unknown/model": 0.99}


def _ctx(registry, tokens=100, text="fix this bug"):
    return RoutingContext(
        request=OpenAIChatRequest(messages=[ChatMessage(role="user", content=text)]),
        features=Features("code", "hard", tokens, False, "en", text),
        session_id="s", registry=registry)


def _router():
    r = object.__new__(PerfRouterRouter)
    r._ready = True
    r._perf_router = _Inf()
    return r


def test_rank_models_maps_inference_ids_to_registry_names(registry):
    assert _router().rank_models(_ctx(registry)) == {"qwen3:8b": 0.9, "deepseek-v4": 0.5}


def test_rank_models_never_raises(registry):
    r = _router()
    r._perf_router = object()
    assert r.rank_models(_ctx(registry)) is None
    r._ready = False
    assert r.rank_models(_ctx(registry)) is None


# ── main._avoid_failed: best score, fits the prompt ───────────────────────────

@pytest.fixture
def wired(monkeypatch):
    from optmod import main
    big = lambda n, tier, ctx, cost: ModelConfig(   # noqa: E731
        name=n, provider="p", base_url="http://x/v1", api_key="k", tier=tier,
        tier_name=["fast", "reasoning", "oracle"][tier], context_window=ctx, cost_per_1k=cost,
        supports_tools=True, thinking_mode=False, thinking_default=False)
    models = [
        big("dead", 0, 500_000, 0.0),
        big("cheap-mid", 1, 500_000, 0.001),
        big("best-small", 1, 64_000, 0.002),         # best score but window too small
        big("best-fit", 2, 500_000, 0.01),
        big("worse-fit", 2, 500_000, 0.005),
    ]
    reg = ModelRegistry(models, "cheap-mid")

    class R:
        name = "R"

        def rank_models(self, ctx):
            return {"dead": 0.2, "cheap-mid": 0.6, "best-small": 0.95, "best-fit": 0.9, "worse-fit": 0.7}

    monkeypatch.setattr(main, "_registry", reg)
    monkeypatch.setattr(main, "_router", R())
    monkeypatch.setattr(main, "_model_health", ModelHealth())
    monkeypatch.setattr(main, "_token_margin", 1.8)
    monkeypatch.setattr(main, "_escalation_strategy", "best_quality")
    monkeypatch.setattr(main, "_escalation_cost_mult", None)
    return main, reg


def _decision(reg, name):
    return RoutingDecision(model=reg.get(name), mutator="noop", reason="r",
                           confidence=1.0, router_name="R")


def test_swap_picks_best_score_that_fits_the_prompt(wired):
    main, reg = wired
    main._model_health.mark_failure("dead", "rate_limit")
    ctx = _ctx(reg, tokens=200_000)                 # x1.8 = 360k: excludes the 64k model
    out = main._avoid_failed(_decision(reg, "dead"), ctx)
    assert out.model.name == "best-fit"
    assert "best predicted quality=0.900" in out.reason


def test_swap_best_quality_can_go_to_a_lower_tier_than_the_failed_model(wired):
    main, reg = wired
    ctx = _ctx(reg, tokens=100)
    ctx.models_tried.append("best-fit")             # top tier already failed
    out = main._avoid_failed(_decision(reg, "best-fit"), ctx)
    assert out.model.name == "best-small"           # 0.95, fits a short prompt, lower tier


def test_next_tier_strategy_keeps_legacy_behaviour(wired):
    main, reg = wired
    main._escalation_strategy = "next_tier"
    main._model_health.mark_failure("dead", "server")
    out = main._avoid_failed(_decision(reg, "dead"), _ctx(reg, tokens=200_000))
    assert out.model.name == "cheap-mid"            # cheapest of the next tier that fits
    assert "next tier up" in out.reason


def test_healthy_pick_is_left_alone(wired):
    main, reg = wired
    d = _decision(reg, "cheap-mid")
    assert main._avoid_failed(d, _ctx(reg)) is d


# ── context overflow + prompt-size knowledge ──────────────────────────────────

OVERFLOW_BODY = {"error": {"message": (
    "This endpoint's maximum context length is 131072 tokens. However, you requested about "
    "489693 tokens (484443 of text input, 5250 of tool input).")}}


def test_needed_tokens_uses_margin_ratio_and_floor(registry):
    ctx = _ctx(registry, tokens=1000)
    assert ctx.needed_tokens(1.7) == 1700
    ctx.token_margin = 1.2
    assert ctx.needed_tokens(1.7) == 1200
    ctx.token_floor = 5000
    assert ctx.needed_tokens(1.7) == 5000


@respx.mock
async def test_context_overflow_is_its_own_retryable_error_with_size():
    from optmod.escalation import EscalationPolicy
    respx.post("https://prov.test/v1/chat/completions").mock(
        return_value=httpx.Response(400, json=OVERFLOW_BODY))
    respx.post("https://prov2.test/v1/chat/completions").mock(
        return_value=httpx.Response(400, json={"error": {"message": "invalid field"}}))
    mk = lambda url: ModelConfig(name="m", provider="p", base_url=url, api_key="k",   # noqa: E731
                                 tier=0, tier_name="fast", context_window=1000, cost_per_1k=0.0,
                                 supports_tools=True, thinking_mode=False, thinking_default=False)
    req = OpenAIChatRequest(messages=[ChatMessage(role="user", content="x")])
    fwd = ModelForwarder()
    info: dict = {}
    _, err = await fwd.forward(mk("https://prov.test/v1"), req, req.messages, info)
    assert err == "context_overflow" and info["context_tokens"] == 489693
    _, err2 = await fwd.forward(mk("https://prov2.test/v1"), req, req.messages, {})
    assert err2 == "bad_request"                        # an ordinary 400 still stops
    await fwd.close()
    assert "context_overflow" in EscalationPolicy().retryable


def test_swap_after_overflow_requires_a_window_for_the_reported_size(wired):
    main, reg = wired
    ctx = _ctx(reg, tokens=100)                         # estimate alone would allow the 64k model
    ctx.models_tried.append("cheap-mid")
    ctx.token_floor = 80_000
    out = main._avoid_failed(_decision(reg, "cheap-mid"), ctx)
    assert out.model.name == "best-fit"                 # best score among models with window >= 80k


def test_swap_relaxes_margin_when_no_window_holds_the_scaled_estimate(wired):
    main, reg = wired
    # 300k x 1.8 = 540k exceeds every window (500k), but the raw estimate fits: must not resend.
    ctx = _ctx(reg, tokens=300_000)
    ctx.models_tried.append("cheap-mid")
    out = main._avoid_failed(_decision(reg, "cheap-mid"), ctx)
    assert out.model.name == "best-fit"


def test_swap_picks_largest_window_when_even_the_raw_estimate_does_not_fit(wired):
    main, reg = wired
    ctx = _ctx(reg, tokens=600_000)
    ctx.models_tried.append("cheap-mid")
    out = main._avoid_failed(_decision(reg, "cheap-mid"), ctx)
    assert out.model.name != "cheap-mid"
    assert out.model.context_window == 500_000


def test_escalation_pin_only_when_within_cost_multiplier(wired, monkeypatch):
    main, reg = wired
    monkeypatch.setattr(main, "_escalation_cost_mult", 10)      # primary cheap-mid 0.001 -> 0.01 ceiling
    ok = _decision(reg, "worse-fit")                            # 0.005
    ok.reason = "escalated from=dead to=worse-fit (failed this request)"
    assert main._escalated_over_budget(ok) is False
    pricey = _decision(reg, "best-fit")                         # 0.01 -> at ceiling, still ok
    pricey.reason = ok.reason
    assert main._escalated_over_budget(pricey) is False
    over = RoutingDecision(model=reg.get("best-fit"), mutator="noop", reason=ok.reason,
                           confidence=1.0, router_name="R")
    monkeypatch.setattr(main, "_escalation_cost_mult", 5)       # ceiling 0.005 -> 0.01 is over
    assert main._escalated_over_budget(over) is True
    plain = _decision(reg, "best-fit")
    plain.reason = "perf_router: normal pick"
    assert main._escalated_over_budget(plain) is False          # only escalations are affected


def test_e2e_overflow_recovers_on_a_bigger_model_and_learns_the_ratio(client_main):
    import json
    client, main = client_main
    sent: list[str] = []

    def respond(request):
        sent.append(json.loads(request.content)["model"])
        if len(sent) == 1:
            return httpx.Response(400, json=OVERFLOW_BODY)
        ok = {"id": "t", "object": "chat.completion", "created": 1, "model": "t",
              "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                           "finish_reason": "stop"}],
              "usage": {"prompt_tokens": 5000, "completion_tokens": 5, "total_tokens": 5005}}
        return httpx.Response(200, json=ok)

    text = "overflow-recovery probe " + "word " * 1200      # ~1560 estimated tokens
    with respx.mock:
        for base in ("https://openrouter.ai/api/v1", "https://api.deepseek.com"):
            respx.post(f"{base}/chat/completions").mock(side_effect=respond)
        r = client.post("/v1/chat/completions", json={
            "model": "optmod", "messages": [{"role": "user", "content": text}]})
    assert r.status_code == 200
    assert len(sent) == 2 and sent[0] != sent[1]
    assert any(v > 2.0 for v in main._token_ratio.values())        # 5000 billed vs ~1560 estimated


@pytest.fixture(scope="module")
def client_main():
    import os
    from fastapi.testclient import TestClient
    from optmod import main
    mp = pytest.MonkeyPatch()
    for env in ("OPENROUTER_API_KEY", "DEEPSEEK_API_KEY"):
        if not os.environ.get(env):
            mp.setenv(env, "test-key")
    try:
        with TestClient(main.app) as c:
            yield c, main
    finally:
        mp.undo()


def test_cost_multiplier_picks_best_ranked_within_budget(wired):
    main, reg = wired                                   # primary cheap-mid costs 0.001
    main._model_health.mark_failure("dead", "rate_limit")
    ctx = _ctx(reg, tokens=200_000)
    main._escalation_cost_mult = 6                      # ceiling 0.006: best-fit (0.01) excluded
    out = main._avoid_failed(_decision(reg, "dead"), ctx)
    assert out.model.name == "worse-fit" and "cost<=6x primary" in out.reason
    main._escalation_cost_mult = None                   # off: best overall
    assert main._avoid_failed(_decision(reg, "dead"), ctx).model.name == "best-fit"


def test_cost_multiplier_relaxed_when_nothing_affordable(wired):
    main, reg = wired
    main._model_health.mark_failure("dead", "rate_limit")
    main._escalation_cost_mult = 0.5                    # ceiling 0.0005: only free models qualify, none fit
    out = main._avoid_failed(_decision(reg, "dead"), _ctx(reg, tokens=200_000))
    assert out.model.name == "best-fit" and "cost cap relaxed" in out.reason
