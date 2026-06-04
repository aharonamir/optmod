import time
from dataclasses import asdict

import httpx
import pytest
import respx

from optmod.registry import ModelConfig, ModelRegistry
from optmod.schemas import (
    ChatMessage,
    Features,
    OpenAIChatRequest,
    RoutingContext,
    SessionPin,
)


@pytest.fixture
def registry():
    """Mirror the conftest registry but include supports_vision + a small-context model."""
    models = [
        ModelConfig(
            name="tiny-fast", provider="ollama",
            base_url="http://localhost/v1", api_key="x",
            tier=0, tier_name="fast",
            context_window=8000, cost_per_1k=0.0,
            supports_tools=True, thinking_mode=False, thinking_default=False,
            supports_vision=False,
        ),
        ModelConfig(
            name="mid-reasoner", provider="ollama",
            base_url="http://localhost/v1", api_key="x",
            tier=1, tier_name="reasoning",
            context_window=32_000, cost_per_1k=0.0001,
            supports_tools=True, thinking_mode=False, thinking_default=False,
            supports_vision=False,
        ),
        ModelConfig(
            name="big-oracle", provider="deepseek",
            base_url="http://deepseek/v1", api_key="x",
            tier=2, tier_name="oracle",
            context_window=200_000, cost_per_1k=0.001,
            supports_tools=True, thinking_mode=False, thinking_default=False,
            supports_vision=False,
        ),
        ModelConfig(
            name="vision-oracle", provider="openrouter",
            base_url="http://openrouter/v1", api_key="x",
            tier=2, tier_name="oracle",
            context_window=200_000, cost_per_1k=0.003,
            supports_tools=True, thinking_mode=False, thinking_default=False,
            supports_vision=True,
        ),
    ]
    return ModelRegistry(models, "big-oracle")


@pytest.fixture
def text_features():
    return Features(
        task_type="general", difficulty="easy", token_count=500,
        has_tools=False, language="en", last_user_message="hi",
    )


@pytest.fixture
def ctx(registry, text_features):
    return RoutingContext(
        request=OpenAIChatRequest(messages=[ChatMessage(role="user", content="hi")]),
        features=text_features,
        session_id="s1",
        registry=registry,
    )


# ── Helper-level tests ────────────────────────────────────────────────────────

class TestPinHelpers:

    def setup_method(self):
        from optmod import main
        main._session_pins.clear()
        main._hard_window_s = 300.0
        main._soft_window_s = 1800.0

    def test_get_active_pin_returns_none_when_empty(self):
        from optmod.main import _get_active_pin
        assert _get_active_pin("nope", time.time()) is None

    def test_get_active_pin_returns_pin_within_soft_window(self):
        from optmod import main
        from optmod.main import _get_active_pin
        now = 1_000_000.0
        main._session_pins["s1"] = SessionPin(
            model_name="mid-reasoner", last_turn_at=now - 1000,
        )
        assert _get_active_pin("s1", now) is not None

    def test_get_active_pin_evicts_after_soft_window(self):
        from optmod import main
        from optmod.main import _get_active_pin
        now = 1_000_000.0
        main._session_pins["s1"] = SessionPin(
            model_name="mid-reasoner", last_turn_at=now - 5000,
        )
        assert _get_active_pin("s1", now) is None
        assert "s1" not in main._session_pins

    def test_update_pin_creates_entry_and_increments_turn_count(self):
        from optmod import main
        from optmod.main import _update_pin
        now = time.time()
        _update_pin("s1", "mid-reasoner", cached_tokens=800, prompt_tokens=1000, now=now)
        pin = main._session_pins["s1"]
        assert pin.model_name == "mid-reasoner"
        assert pin.turn_count == 1
        assert pin.last_cache_rate == 0.8
        _update_pin("s1", "mid-reasoner", cached_tokens=900, prompt_tokens=1000, now=now)
        assert main._session_pins["s1"].turn_count == 2

    def test_update_pin_handles_zero_prompt_tokens(self):
        from optmod import main
        from optmod.main import _update_pin
        _update_pin("s1", "mid-reasoner", cached_tokens=0, prompt_tokens=0, now=time.time())
        assert main._session_pins["s1"].last_cache_rate == 0.0


class TestRequestHasImages:

    def test_text_only(self):
        from optmod.main import _request_has_images
        req = OpenAIChatRequest(messages=[ChatMessage(role="user", content="hi")])
        assert _request_has_images(req) is False

    def test_image_url_part(self):
        from optmod.main import _request_has_images
        req = OpenAIChatRequest(messages=[ChatMessage(role="user", content=[
            {"type": "text", "text": "what's this?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,xyz"}},
        ])])
        assert _request_has_images(req) is True

    def test_plain_image_part(self):
        from optmod.main import _request_has_images
        req = OpenAIChatRequest(messages=[ChatMessage(role="user", content=[
            {"type": "image", "data": "..."},
        ])])
        assert _request_has_images(req) is True


# ── Hard-pin gate ─────────────────────────────────────────────────────────────

class TestTryHardPin:

    def test_honoured_for_text_request_under_context(self, ctx):
        from optmod.main import _try_hard_pin
        pin = SessionPin(model_name="mid-reasoner", last_turn_at=time.time())
        decision, state = _try_hard_pin(pin, ctx, time.time())
        assert state == "hard"
        assert decision is not None
        assert decision.model.name == "mid-reasoner"
        assert decision.router_name == "session_pin"
        assert decision.mutator == "noop"

    def test_evicts_when_pinned_model_missing(self, ctx):
        from optmod.main import _try_hard_pin
        pin = SessionPin(model_name="ghost-model", last_turn_at=time.time())
        decision, state = _try_hard_pin(pin, ctx, time.time())
        assert decision is None
        assert state == "evicted_missing_model"

    def test_evicts_on_vision_request_against_text_only_pin(self, registry, text_features):
        from optmod.main import _try_hard_pin
        req = OpenAIChatRequest(messages=[ChatMessage(role="user", content=[
            {"type": "image_url", "image_url": {"url": "..."}},
        ])])
        ctx = RoutingContext(
            request=req, features=text_features,
            session_id="s1", registry=registry,
        )
        pin = SessionPin(model_name="mid-reasoner", last_turn_at=time.time())
        decision, state = _try_hard_pin(pin, ctx, time.time())
        assert decision is None
        assert state == "evicted_vision"

    def test_honoured_on_vision_request_against_vision_pin(self, registry, text_features):
        from optmod.main import _try_hard_pin
        req = OpenAIChatRequest(messages=[ChatMessage(role="user", content=[
            {"type": "image_url", "image_url": {"url": "..."}},
        ])])
        ctx = RoutingContext(
            request=req, features=text_features,
            session_id="s1", registry=registry,
        )
        pin = SessionPin(model_name="vision-oracle", last_turn_at=time.time())
        decision, state = _try_hard_pin(pin, ctx, time.time())
        assert state == "hard"
        assert decision.model.name == "vision-oracle"

    def test_rehomes_on_context_overflow(self, registry):
        """Pinned 8k model + 20k tokens → rehome to 32k mid-reasoner (same provider tier-adjacent)."""
        from optmod.main import _try_hard_pin
        features = Features(
            task_type="general", difficulty="easy", token_count=20_000,
            has_tools=False, language="en", last_user_message="hi",
        )
        ctx = RoutingContext(
            request=OpenAIChatRequest(messages=[ChatMessage(role="user", content="hi")]),
            features=features, session_id="s1", registry=registry,
        )
        pin = SessionPin(model_name="tiny-fast", last_turn_at=time.time())
        decision, state = _try_hard_pin(pin, ctx, time.time())
        assert state == "rehomed_context"
        assert decision is not None
        # Should pick the cheapest larger-context model — mid-reasoner (0.0001) over big-oracle (0.001)
        assert decision.model.name == "mid-reasoner"
        # Pin entry mutated in place to the rehomed model
        assert pin.model_name == "mid-reasoner"

    def test_evicts_context_when_no_candidate_fits(self, registry):
        from optmod.main import _try_hard_pin
        features = Features(
            task_type="general", difficulty="hard", token_count=10_000_000,
            has_tools=False, language="en", last_user_message="hi",
        )
        ctx = RoutingContext(
            request=OpenAIChatRequest(messages=[ChatMessage(role="user", content="hi")]),
            features=features, session_id="s1", registry=registry,
        )
        pin = SessionPin(model_name="tiny-fast", last_turn_at=time.time())
        decision, state = _try_hard_pin(pin, ctx, time.time())
        assert decision is None
        assert state == "evicted_context"


# ── Rehome direct ─────────────────────────────────────────────────────────────

class TestRehomePin:

    def test_prefers_same_tier(self, registry):
        from optmod.main import _rehome_pin
        features = Features(
            task_type="general", difficulty="easy", token_count=20_000,
            has_tools=False, language="en", last_user_message="hi",
        )
        ctx = RoutingContext(
            request=OpenAIChatRequest(messages=[ChatMessage(role="user", content="hi")]),
            features=features, session_id="s1", registry=registry,
        )
        # Pin small "tiny-fast"; same-tier (fast) has nothing else that fits → falls to other tiers.
        # mid-reasoner has the closest cost (0.0001 vs 0.0); but big-oracle (0.001) also fits.
        replacement = _rehome_pin(registry.get("tiny-fast"), ctx)
        assert replacement.name == "mid-reasoner"


# ── Cache-token extraction (forwarder) ───────────────────────────────────────

class TestCacheTokenExtraction:

    @respx.mock
    @pytest.mark.asyncio
    async def test_deepseek_openai_compat_shape(self):
        from optmod.forwarder import ModelForwarder
        respx.post("http://x/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "id": "1", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 5,
                          "prompt_tokens_details": {"cached_tokens": 80}},
            }),
        )
        fwd = ModelForwarder()
        model = ModelConfig(
            name="m", provider="p", base_url="http://x", api_key="k",
            tier=0, tier_name="fast", context_window=1000, cost_per_1k=0.0,
            supports_tools=True, thinking_mode=False, thinking_default=False,
        )
        req = OpenAIChatRequest(messages=[ChatMessage(role="user", content="hi")])
        body, err = await fwd.forward(model, req, req.messages)
        await fwd.close()
        assert err is None
        assert body["usage"]["_optmod_cached_tokens"] == 80

    @respx.mock
    @pytest.mark.asyncio
    async def test_deepseek_legacy_field(self):
        from optmod.forwarder import ModelForwarder
        respx.post("http://x/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "id": "1", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 5,
                          "prompt_cache_hit_tokens": 75},
            }),
        )
        fwd = ModelForwarder()
        model = ModelConfig(
            name="m", provider="p", base_url="http://x", api_key="k",
            tier=0, tier_name="fast", context_window=1000, cost_per_1k=0.0,
            supports_tools=True, thinking_mode=False, thinking_default=False,
        )
        req = OpenAIChatRequest(messages=[ChatMessage(role="user", content="hi")])
        body, err = await fwd.forward(model, req, req.messages)
        await fwd.close()
        assert err is None
        assert body["usage"]["_optmod_cached_tokens"] == 75

    @respx.mock
    @pytest.mark.asyncio
    async def test_anthropic_native_field(self):
        from optmod.forwarder import ModelForwarder
        respx.post("http://x/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "id": "1", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 5,
                          "cache_read_input_tokens": 60},
            }),
        )
        fwd = ModelForwarder()
        model = ModelConfig(
            name="m", provider="p", base_url="http://x", api_key="k",
            tier=0, tier_name="fast", context_window=1000, cost_per_1k=0.0,
            supports_tools=True, thinking_mode=False, thinking_default=False,
        )
        req = OpenAIChatRequest(messages=[ChatMessage(role="user", content="hi")])
        body, err = await fwd.forward(model, req, req.messages)
        await fwd.close()
        assert err is None
        assert body["usage"]["_optmod_cached_tokens"] == 60

    @respx.mock
    @pytest.mark.asyncio
    async def test_no_cache_field_defaults_to_zero(self):
        from optmod.forwarder import ModelForwarder
        respx.post("http://x/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "id": "1", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 5},
            }),
        )
        fwd = ModelForwarder()
        model = ModelConfig(
            name="m", provider="p", base_url="http://x", api_key="k",
            tier=0, tier_name="fast", context_window=1000, cost_per_1k=0.0,
            supports_tools=True, thinking_mode=False, thinking_default=False,
        )
        req = OpenAIChatRequest(messages=[ChatMessage(role="user", content="hi")])
        body, err = await fwd.forward(model, req, req.messages)
        await fwd.close()
        assert err is None
        assert body["usage"]["_optmod_cached_tokens"] == 0
