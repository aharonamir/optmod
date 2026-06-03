import re
from collections import defaultdict

from optmod.mutators import BaseContextMutator
from optmod.schemas import ChatMessage, RoutingDecision

# ── Thresholds ─────────────────────────────────────────────────────────────
_MIN_COMPRESS_SIZE   = 500    # skip compression on short content (< 500 B)
_DETECT_WINDOW       = 1024   # chars to inspect for format detection
_MAX_DIFF_FILE_LINES = 100    # lines per file-diff before truncating
_MAX_CONTEXT_RUN     = 5      # consecutive unchanged diff lines before collapsing
_MAX_TREE_LINES      = 200    # tree output lines before truncating
_MAX_LS_LINES        = 80     # plain file-list lines before truncating
_GREP_PER_FILE_MAX   = 10     # grep matches shown per file
_LOG_MIN_STREAK      = 3      # identical consecutive lines before deduplicating
_LOG_MAX_LINES       = 2000   # hard cap on total log lines after dedup
_SMART_TRUNC_HEAD    = 120    # lines kept from top in smart-truncate
_SMART_TRUNC_TAIL    = 60     # lines kept from bottom in smart-truncate

# ── Compiled patterns ──────────────────────────────────────────────────────
_RE_DIFF_FENCE   = re.compile(r'(?m)^(?=diff --git )')
_RE_GIT_STATUS   = re.compile(
    r'(?m)^(On branch |Changes not staged|Changes to be committed|'
    r'Untracked files:|nothing to commit|HEAD detached)',
)
_RE_GIT_HINT     = re.compile(r'(?m)^[ \t]+\(use "git [^)]+\)[ \t]*\n')
_RE_BUILD_DETECT = re.compile(
    r'(?im)^(npm (warn|error)|warning[\[\s]|error\[E\d|Compiling \w|Building \[)',
)
_RE_BUILD_DUP    = re.compile(r'(?m)^(npm (?:warn|error) .+)(\n\1)+')
_RE_GREP_LINE    = re.compile(r'(?m)^([^:\n\s][^:\n]*):(\d+):(.*)$')
_RE_TREE_CHAR    = re.compile(r'(?m)^[│├└─ ]{2,}\S')
_RE_LOG_TS       = re.compile(
    r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?'
)
_TEST_PASS_PATTERNS = [
    (re.compile(r'(?m)^.+? PASSED\s*$'),         'pytest'),
    (re.compile(r'(?m)^test .+ \.\.\. ok\s*$'),  'cargo'),
    (re.compile(r'(?m)^\s+✓ .+$'),               'jest'),
]


class ToolResultCompressorMutator(BaseContextMutator):
    def mutate(
        self,
        messages: list[ChatMessage],
        decision: RoutingDecision,
    ) -> list[ChatMessage]:
        result = None
        for i, msg in enumerate(messages):
            if msg.role != "tool" or not isinstance(msg.content, str):
                continue
            compressed = _compress(msg.content)
            if compressed == msg.content:
                continue
            if result is None:
                result = list(messages)
            result[i] = msg.model_copy(update={"content": compressed})
        return result if result is not None else messages


# ── Dispatch ───────────────────────────────────────────────────────────────

def _compress(text: str) -> str:
    if len(text) < _MIN_COMPRESS_SIZE:
        return text

    head       = text[:_DETECT_WINDOW]
    did_filter = False

    if _RE_DIFF_FENCE.search(head):
        text, did_filter = _filter_git_diff(text), True
    elif _RE_GIT_STATUS.search(head):
        text, did_filter = _filter_git_status(text), True
    elif _RE_BUILD_DETECT.search(head):
        text, did_filter = _filter_build_output(text), True
    elif _looks_like_grep(head):
        text, did_filter = _filter_grep(text), True
    elif _RE_TREE_CHAR.search(head):
        text, did_filter = _filter_ls_tree(text, _MAX_TREE_LINES), True
    elif _looks_like_ls(text):
        text, did_filter = _filter_ls_tree(text, _MAX_LS_LINES), True

    new = _filter_test_output(text)
    if new != text:
        text, did_filter = new, True

    new = _filter_log_dedup(text)
    if new != text:
        text, did_filter = new, True

    # Smart-truncate fallback: catch-all for long unrecognised output
    if not did_filter and len(text.splitlines()) > _SMART_TRUNC_HEAD + _SMART_TRUNC_TAIL:
        text = _smart_truncate(text)

    return text


# ── git diff ───────────────────────────────────────────────────────────────

def _filter_git_diff(text: str) -> str:
    segments = _RE_DIFF_FENCE.split(text)
    out = []
    for seg in segments:
        if not seg.startswith('diff --git '):
            out.append(seg)
            continue
        lines = seg.splitlines(keepends=True)
        try:
            hunk_idx = next(i for i, l in enumerate(lines) if l.startswith('@@'))
            header   = lines[:hunk_idx]
            hunks    = lines[hunk_idx:]
        except StopIteration:
            out.append(seg)   # binary diff / empty — leave untouched
            continue

        hunks = _collapse_context(hunks)
        total = len(header) + len(hunks)
        if total <= _MAX_DIFF_FILE_LINES:
            out.append(''.join(header + hunks))
        else:
            budget  = max(8, _MAX_DIFF_FILE_LINES - len(header))
            dropped = len(hunks) - budget
            out.append(
                ''.join(header + hunks[:budget])
                + f'\n[… {dropped} hunk lines not shown]\n'
            )
    return ''.join(out)


def _collapse_context(lines: list[str]) -> list[str]:
    """Replace long runs of unchanged diff lines with a one-line note."""
    out = []
    run: list[str] = []

    def flush() -> None:
        if len(run) > _MAX_CONTEXT_RUN:
            keep_head = 3
            keep_tail = 2
            collapsed = len(run) - keep_head - keep_tail
            out.extend(run[:keep_head])
            if collapsed > 0:
                out.append(f'[… {collapsed} unchanged lines …]\n')
            out.extend(run[-keep_tail:])
        else:
            out.extend(run)
        run.clear()

    for line in lines:
        if line[:1] == ' ':   # space prefix = context line in unified diff
            run.append(line)
        else:
            flush()
            out.append(line)
    flush()
    return out


# ── git status ─────────────────────────────────────────────────────────────

def _filter_git_status(text: str) -> str:
    """Strip the '(use "git ...")' instruction lines git status adds."""
    return _RE_GIT_HINT.sub('', text)


# ── build output (npm / cargo / webpack) ───────────────────────────────────

def _filter_build_output(text: str) -> str:
    text = _RE_BUILD_DUP.sub(r'\1\n', text)   # collapse consecutive dup warnings
    if len(text.splitlines()) > 200:
        text = _smart_truncate(text)
    return text


# ── grep ───────────────────────────────────────────────────────────────────

def _looks_like_grep(head: str) -> bool:
    return len(_RE_GREP_LINE.findall(head)) >= 3


def _filter_grep(text: str) -> str:
    by_file: dict[str, list[str]] = defaultdict(list)
    order: list[str] = []
    unmatched: list[str] = []

    for line in text.splitlines(keepends=True):
        m = _RE_GREP_LINE.match(line.rstrip('\n'))
        if m:
            fname = m.group(1)
            if fname not in by_file:
                order.append(fname)
            by_file[fname].append(line)
        else:
            unmatched.append(line)

    if not by_file:
        return text

    out: list[str] = []
    total_dropped = 0
    for fname in order:
        matches = by_file[fname]
        dropped = max(0, len(matches) - _GREP_PER_FILE_MAX)
        out.extend(matches[:_GREP_PER_FILE_MAX])
        if dropped:
            total_dropped += dropped
            out.append(f'[… {dropped} more matches in {fname}]\n')

    out.extend(unmatched)
    if total_dropped:
        out.append(f'[grep: {total_dropped} matches omitted total]\n')
    return ''.join(out)


# ── directory listings / tree ──────────────────────────────────────────────

def _looks_like_ls(text: str) -> bool:
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 10:
        return False
    path_like = sum(1 for l in lines if ' ' not in l and len(l) < 80)
    return path_like / len(lines) > 0.80


def _filter_ls_tree(text: str, max_lines: int) -> str:
    lines = text.splitlines(keepends=True)
    if len(lines) <= max_lines:
        return text
    dropped = len(lines) - max_lines
    return ''.join(lines[:max_lines]) + f'[… {dropped} more entries not shown]\n'


# ── test runner output ─────────────────────────────────────────────────────

def _filter_test_output(text: str) -> str:
    """Collapse passing-test lines; keep failing/error lines and the summary."""
    for pat, runner in _TEST_PASS_PATTERNS:
        matches = pat.findall(text)
        if len(matches) >= 5:
            text = pat.sub('', text)
            text = text.rstrip('\n')
            text += f'\n[{len(matches)} {runner} passing tests collapsed]\n'
    return text


# ── log deduplication ──────────────────────────────────────────────────────

def _filter_log_dedup(text: str) -> str:
    """Collapse consecutive identical lines (timestamps stripped for comparison)."""
    lines = text.splitlines(keepends=True)
    out   = []
    i     = 0
    while i < len(lines):
        key    = _RE_LOG_TS.sub('', lines[i]).strip()
        streak = 1
        while i + streak < len(lines):
            if _RE_LOG_TS.sub('', lines[i + streak]).strip() == key:
                streak += 1
            else:
                break
        if streak >= _LOG_MIN_STREAK and key:   # don't dedup blank lines
            out.append(lines[i])
            out.append(f'[… ×{streak - 1} more identical lines]\n')
        else:
            out.extend(lines[i : i + streak])
        i += streak
    # Hard cap: prevent unbounded log dumps
    if len(out) > _LOG_MAX_LINES:
        dropped = len(out) - _LOG_MAX_LINES
        out = out[:_LOG_MAX_LINES]
        out.append(f'[… {dropped} more log lines capped]\n')
    return ''.join(out)


# ── smart-truncate fallback ────────────────────────────────────────────────

def _smart_truncate(
    text: str,
    head: int = _SMART_TRUNC_HEAD,
    tail: int = _SMART_TRUNC_TAIL,
) -> str:
    """Keep first `head` + last `tail` lines; collapse the middle."""
    lines = text.splitlines(keepends=True)
    total = len(lines)
    if total <= head + tail:
        return text
    dropped = total - head - tail
    return (
        ''.join(lines[:head])
        + f'[… {dropped} lines truncated …]\n'
        + ''.join(lines[-tail:])
    )
