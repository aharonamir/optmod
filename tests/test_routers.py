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
    ctx = make_ctx(registry, tried=["qwen2.5:7b", "qwen3:8b", "deepseek-v4"])
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
    ctx = make_ctx(registry)
    assert router.route(ctx).model.tier == 0
