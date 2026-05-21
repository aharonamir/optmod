from optmod.features import FeatureExtractor
from optmod.schemas import OpenAIChatRequest, ChatMessage

ext = FeatureExtractor()


def msg(text):
    return OpenAIChatRequest(messages=[ChatMessage(role="user", content=text)])


def test_reason():
    assert ext.extract(msg("why does this algorithm fail?")).task_type == "reason"

def test_code():
    assert ext.extract(msg("implement a binary search tree")).task_type == "code"

def test_extract():
    assert ext.extract(msg("extract all emails from this document")).task_type == "extract"

def test_summarize():
    assert ext.extract(msg("summarize this PDF in 3 bullet points")).task_type == "summarize"

def test_tool_call():
    assert ext.extract(msg("call the get_weather function")).task_type == "tool_call"

def test_search():
    assert ext.extract(msg("search for recent papers on LLM routing")).task_type == "search"

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
    r = ext.extract(msg("one two three four five six seven eight nine ten"))
    assert 10 <= r.token_count <= 20
