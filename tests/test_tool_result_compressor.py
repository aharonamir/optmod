import pytest
from optmod.registry import ModelConfig
from optmod.schemas import ChatMessage, RoutingDecision
from optmod.mutators.tool_result_compressor import (
    ToolResultCompressorMutator,
    _compress,
    _filter_git_diff,
    _filter_git_status,
    _filter_build_output,
    _filter_grep,
    _filter_ls_tree,
    _filter_test_output,
    _filter_log_dedup,
    _smart_truncate,
    _looks_like_grep,
    _looks_like_ls,
    _MIN_COMPRESS_SIZE,
    _GREP_PER_FILE_MAX,
    _LOG_MIN_STREAK,
    _LOG_MAX_LINES,
    _MAX_TREE_LINES,
    _MAX_LS_LINES,
    _SMART_TRUNC_HEAD,
    _SMART_TRUNC_TAIL,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _tool_msg(content: str) -> ChatMessage:
    return ChatMessage(role="tool", content=content, tool_call_id="tc1")

def _user_msg(content: str) -> ChatMessage:
    return ChatMessage(role="user", content=content)

def _make_decision() -> RoutingDecision:
    model = ModelConfig(
        name="test", provider="test", base_url="http://x", api_key="x",
        tier=0, tier_name="fast", context_window=4096, cost_per_1k=0.0,
        supports_tools=True, thinking_mode=False, thinking_default=False,
    )
    return RoutingDecision(model=model, mutator="noop", reason="", confidence=1.0, router_name="test")

MUTATOR  = ToolResultCompressorMutator()
DECISION = _make_decision()


# ── mutate() interface ─────────────────────────────────────────────────────

def test_mutate_non_tool_messages_pass_through():
    msgs = [_user_msg("hello"), ChatMessage(role="assistant", content="hi")]
    assert MUTATOR.mutate(msgs, DECISION) is msgs

def test_mutate_short_tool_content_unchanged():
    short = "x" * (_MIN_COMPRESS_SIZE - 1)
    msgs  = [_tool_msg(short)]
    assert MUTATOR.mutate(msgs, DECISION) is msgs

def test_mutate_returns_original_list_when_nothing_changes():
    msgs = [_tool_msg("short")]
    assert MUTATOR.mutate(msgs, DECISION) is msgs

def test_mutate_creates_new_list_lazily():
    long_ls = "\n".join(f"file_{i}.py" for i in range(200))
    msgs    = [_user_msg("hi"), _tool_msg(long_ls)]
    result  = MUTATOR.mutate(msgs, DECISION)
    assert result is not msgs
    assert result[0] is msgs[0]    # unchanged ref
    assert result[1] is not msgs[1]  # new ChatMessage

def test_mutate_does_not_modify_original_message():
    long_ls          = "\n".join(f"file_{i}.py" for i in range(200))
    original_content = long_ls
    msg  = _tool_msg(long_ls)
    msgs = [msg]
    MUTATOR.mutate(msgs, DECISION)
    assert msg.content == original_content

def test_mutate_skips_list_typed_content():
    msgs = [ChatMessage(role="tool", content=[{"type": "text", "text": "x"}])]
    assert MUTATOR.mutate(msgs, DECISION) is msgs


# ── MIN_COMPRESS_SIZE guard ────────────────────────────────────────────────

def test_compress_skips_short_content():
    # Even a diff won't fire if content is below the byte floor
    short = "diff --git a/f b/f\n@@ -1 +1 @@\n-x\n+y\n"
    assert len(short) < _MIN_COMPRESS_SIZE
    assert _compress(short) == short


# ── git diff ───────────────────────────────────────────────────────────────

_SIMPLE_DIFF = """\
diff --git a/foo.py b/foo.py
index abc..def 100644
--- a/foo.py
+++ b/foo.py
@@ -1,5 +1,5 @@
 context 1
 context 2
-old line
+new line
 context 4
 context 5
"""

def test_git_diff_short_passes_through():
    result = _filter_git_diff(_SIMPLE_DIFF)
    assert "-old line" in result
    assert "+new line" in result

def test_git_diff_collapses_long_context_run():
    diff = (
        "diff --git a/f b/f\nindex 0..0 100644\n--- a/f\n+++ b/f\n"
        "@@ -1,20 +1,20 @@\n"
        "-first change\n+first change new\n"
        + " context\n" * 10
        + "-second change\n+second change new\n"
    )
    result = _filter_git_diff(diff)
    assert "unchanged lines" in result
    assert "-first change" in result
    assert "-second change" in result

def test_git_diff_truncates_large_file_diff():
    body = "@@ -1,200 +1,200 @@\n" + "+added line\n" * 200
    diff = (
        "diff --git a/big.py b/big.py\nindex 0..0 100644\n"
        "--- a/big.py\n+++ b/big.py\n" + body
    )
    result = _filter_git_diff(diff)
    assert "not shown" in result
    assert result.count("+added line") < 200

def test_git_diff_multi_file_each_gets_own_budget():
    def make_file_diff(name: str) -> str:
        body = "@@ -1,110 +1,110 @@\n" + "+line\n" * 110
        return (
            f"diff --git a/{name} b/{name}\nindex 0..0 100644\n"
            f"--- a/{name}\n+++ b/{name}\n{body}"
        )
    combined = make_file_diff("a.py") + make_file_diff("b.py")
    result   = _filter_git_diff(combined)
    assert result.count("not shown") == 2

def test_git_diff_binary_left_untouched():
    binary = "diff --git a/img.png b/img.png\nindex abc..def 100644\nBinary files differ\n"
    assert _filter_git_diff(binary) == binary

def test_git_diff_preamble_before_first_file_preserved():
    text = "some preamble\n" + _SIMPLE_DIFF
    result = _filter_git_diff(text)
    assert "some preamble" in result


# ── git status ─────────────────────────────────────────────────────────────

_GIT_STATUS = """\
On branch main
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
\tmodified:   foo.py

Untracked files:
  (use "git add <file>..." to include in what will be committed)
\tbar.py
"""

def test_git_status_strips_hint_lines():
    assert '(use "git' not in _filter_git_status(_GIT_STATUS)

def test_git_status_keeps_file_names():
    result = _filter_git_status(_GIT_STATUS)
    assert "foo.py" in result
    assert "bar.py" in result

def test_git_status_keeps_branch_line():
    assert "On branch main" in _filter_git_status(_GIT_STATUS)

def test_git_status_nothing_to_commit():
    text = "On branch main\nnothing to commit, working tree clean\n"
    result = _filter_git_status(text)
    assert "nothing to commit" in result


# ── build output ───────────────────────────────────────────────────────────

def test_build_deduplicates_repeated_npm_warnings():
    text   = "npm warn deprecated foo@1.0.0: use bar\n" * 5
    result = _filter_build_output(text)
    assert result.count("npm warn") == 1

def test_build_smart_truncates_very_long_output():
    # Distinct lines — dedup won't collapse them, so smart-truncate must fire
    text   = "\n".join(f"npm warn deprecated pkg_{i}: use something_else" for i in range(300)) + "\n"
    result = _filter_build_output(text)
    assert "truncated" in result
    assert result.count("npm warn") < 300

def test_build_short_output_unchanged():
    text = "npm warn foo\nnpm error bar\n"
    assert _filter_build_output(text) == text

def test_build_different_warnings_not_collapsed():
    text = "npm warn foo\nnpm warn bar\nnpm warn baz\n"
    result = _filter_build_output(text)
    assert "foo" in result and "bar" in result and "baz" in result


# ── grep ───────────────────────────────────────────────────────────────────

def test_looks_like_grep_true():
    head = "src/foo.py:10: match\nsrc/foo.py:20: another\nsrc/bar.py:5: third\n"
    assert _looks_like_grep(head) is True

def test_looks_like_grep_false_for_few_matches():
    head = "src/foo.py:10: one match\nsome prose line\n"
    assert _looks_like_grep(head) is False

def test_looks_like_grep_false_for_diff():
    head = "diff --git a/f b/f\n--- a/f\n+++ b/f\n@@ -1 +1 @@\n"
    assert _looks_like_grep(head) is False

def test_grep_limits_per_file():
    lines  = [f"src/big.py:{i}: match {i}" for i in range(1, 25)]
    text   = "\n".join(lines) + "\n"
    result = _filter_grep(text)
    assert result.count("src/big.py:") == _GREP_PER_FILE_MAX
    assert f"{24 - _GREP_PER_FILE_MAX} more matches" in result

def test_grep_multiple_files_each_capped():
    lines = (
        [f"a.py:{i}: x" for i in range(1, 15)]
        + [f"b.py:{i}: y" for i in range(1, 15)]
    )
    result = _filter_grep("\n".join(lines) + "\n")
    assert result.count("a.py:") == _GREP_PER_FILE_MAX
    assert result.count("b.py:") == _GREP_PER_FILE_MAX

def test_grep_under_limit_unchanged():
    lines  = [f"src/f.py:{i}: match" for i in range(1, 8)]
    text   = "\n".join(lines) + "\n"
    result = _filter_grep(text)
    assert "more matches" not in result
    assert result.count("src/f.py:") == 7

def test_grep_omitted_total_note():
    lines  = [f"f.py:{i}: x" for i in range(1, 25)]
    result = _filter_grep("\n".join(lines) + "\n")
    assert "omitted total" in result

def test_grep_no_matches_returns_original():
    text = "just some prose\nwith no grep lines\n"
    assert _filter_grep(text) == text


# ── ls / tree ──────────────────────────────────────────────────────────────

def test_looks_like_ls_true():
    text = "\n".join(f"file_{i}.py" for i in range(15))
    assert _looks_like_ls(text) is True

def test_looks_like_ls_false_for_short_list():
    text = "\n".join(f"file_{i}.py" for i in range(5))
    assert _looks_like_ls(text) is False

def test_looks_like_ls_false_when_lines_have_spaces():
    text = "\n".join(f"-rw-r--r-- 1 user grp 1234 file_{i}.py" for i in range(15))
    assert _looks_like_ls(text) is False

def test_filter_ls_truncates_at_max_ls_lines():
    lines  = [f"file_{i}.py" for i in range(_MAX_LS_LINES + 50)]
    text   = "\n".join(lines) + "\n"
    result = _filter_ls_tree(text, _MAX_LS_LINES)
    assert "not shown" in result
    assert result.count("file_") == _MAX_LS_LINES

def test_filter_tree_uses_larger_cap():
    # 150 lines — under tree cap (200), should pass unchanged
    lines  = [f"├── file_{i}.py" for i in range(150)]
    text   = "\n".join(lines) + "\n"
    result = _filter_ls_tree(text, _MAX_TREE_LINES)
    assert "not shown" not in result

def test_filter_tree_truncates_above_cap():
    lines  = [f"├── file_{i}.py" for i in range(_MAX_TREE_LINES + 50)]
    text   = "\n".join(lines) + "\n"
    result = _filter_ls_tree(text, _MAX_TREE_LINES)
    assert "not shown" in result

def test_filter_ls_tree_short_unchanged():
    text = "file_a.py\nfile_b.py\n"
    assert _filter_ls_tree(text, _MAX_LS_LINES) == text


# ── test runner output ─────────────────────────────────────────────────────

def test_pytest_collapses_passing():
    lines  = [f"tests/test_foo.py::test_{i} PASSED" for i in range(10)]
    text   = "\n".join(lines) + "\n=== 10 passed ===\n"
    result = _filter_test_output(text)
    assert "PASSED" not in result
    assert "10 pytest passing tests collapsed" in result

def test_pytest_keeps_failure_lines():
    text = (
        "tests/test_foo.py::test_ok PASSED\n" * 6
        + "tests/test_foo.py::test_bad FAILED\n"
        + "=== 1 failed, 6 passed ===\n"
    )
    result = _filter_test_output(text)
    assert "FAILED" in result
    assert "PASSED" not in result

def test_cargo_collapses_passing():
    lines  = [f"test module::fn_{i} ... ok" for i in range(8)]
    text   = "\n".join(lines) + "\ntest result: ok. 8 passed\n"
    result = _filter_test_output(text)
    assert "... ok" not in result
    assert "8 cargo passing tests collapsed" in result

def test_jest_collapses_passing():
    lines  = [f"    ✓ should do thing {i}" for i in range(6)]
    text   = "\n".join(lines) + "\n"
    result = _filter_test_output(text)
    assert "✓" not in result
    assert "6 jest passing tests collapsed" in result

def test_test_output_no_collapse_below_five():
    lines  = [f"tests/test_foo.py::test_{i} PASSED" for i in range(3)]
    text   = "\n".join(lines) + "\n"
    assert _filter_test_output(text) == text


# ── log deduplication ──────────────────────────────────────────────────────

def test_log_dedup_collapses_identical_streak():
    line   = "2024-01-01 10:00:00 INFO  Processing item\n"
    text   = line * 5
    result = _filter_log_dedup(text)
    assert result.count("Processing item") == 1
    assert "×4 more identical" in result

def test_log_dedup_ignores_timestamps():
    lines  = [f"2024-01-01 10:00:0{i} INFO  Same message\n" for i in range(5)]
    result = _filter_log_dedup("".join(lines))
    assert result.count("Same message") == 1

def test_log_dedup_preserves_non_consecutive_repeats():
    text = (
        "2024-01-01 10:00:01 INFO  Message A\n"
        "2024-01-01 10:00:02 INFO  Message B\n"
        "2024-01-01 10:00:03 INFO  Message A\n"
    )
    result = _filter_log_dedup(text)
    assert result.count("Message A") == 2
    assert result.count("Message B") == 1

def test_log_dedup_below_min_streak_unchanged():
    line = "INFO  thing\n"
    text = line * (_LOG_MIN_STREAK - 1)
    assert _filter_log_dedup(text) == text

def test_log_dedup_does_not_collapse_blank_lines():
    text   = "\n\n\n\nsome content\n"
    result = _filter_log_dedup(text)
    assert "identical" not in result

def test_log_dedup_hard_cap():
    text   = "".join(f"2024-01-01 10:00:00 INFO line {i}\n" for i in range(_LOG_MAX_LINES + 500))
    result = _filter_log_dedup(text)
    assert "capped" in result
    assert result.count("\n") <= _LOG_MAX_LINES + 5


# ── smart-truncate ─────────────────────────────────────────────────────────

def test_smart_truncate_keeps_head_and_tail():
    total  = _SMART_TRUNC_HEAD + _SMART_TRUNC_TAIL + 50
    lines  = [f"line {i}\n" for i in range(total)]
    result = _smart_truncate("".join(lines))
    assert "truncated" in result
    assert "line 0\n" in result
    assert f"line {total - 1}\n" in result

def test_smart_truncate_shows_correct_dropped_count():
    total  = _SMART_TRUNC_HEAD + _SMART_TRUNC_TAIL + 40
    lines  = [f"line {i}\n" for i in range(total)]
    result = _smart_truncate("".join(lines))
    assert "40 lines truncated" in result

def test_smart_truncate_short_text_unchanged():
    text = "line 1\nline 2\nline 3\n"
    assert _smart_truncate(text) == text

def test_smart_truncate_exactly_at_limit_unchanged():
    lines = [f"line {i}\n" for i in range(_SMART_TRUNC_HEAD + _SMART_TRUNC_TAIL)]
    assert _smart_truncate("".join(lines)) == "".join(lines)

def test_smart_truncate_custom_head_tail():
    lines  = [f"line {i}\n" for i in range(50)]
    result = _smart_truncate("".join(lines), head=10, tail=5)
    assert "35 lines truncated" in result
    assert "line 0\n" in result
    assert "line 49\n" in result


# ── dispatch integration ───────────────────────────────────────────────────

def test_compress_routes_to_git_diff():
    diff = (
        "diff --git a/f.py b/f.py\nindex 0..0 100644\n--- a/f.py\n+++ b/f.py\n"
        "@@ -1,2 +1,2 @@\n-old\n+new\n" + " context\n" * 200
    )
    result = _compress(diff)
    assert "unchanged lines" in result or "not shown" in result

def test_compress_routes_to_git_status():
    # 8 repetitions ≈ 560 B, above the 500B MIN_COMPRESS_SIZE guard
    status = 'On branch main\n  (use "git add <file>..." to update)\n\tmodified: f.py\n' * 8
    result = _compress(status)
    assert '(use "git' not in result

def test_compress_routes_to_grep():
    # Needs > 500B (MIN_COMPRESS_SIZE) and > GREP_PER_FILE_MAX matches
    lines  = [f"src/f.py:{i}: some match content found here" for i in range(40)]
    result = _compress("\n".join(lines) + "\n")
    assert "more matches" in result

def test_compress_routes_to_ls():
    text   = "\n".join(f"file_{i}.py" for i in range(200)) + "\n"
    result = _compress(text)
    assert "not shown" in result

def test_compress_smart_truncate_fallback_for_unknown():
    # Plain prose: no filter matches, but long → smart-truncate fires
    text   = "\n".join(f"random output line number {i}" for i in range(250)) + "\n"
    result = _compress(text)
    assert "truncated" in result

def test_compress_passthrough_below_min_size():
    assert _compress("short") == "short"
