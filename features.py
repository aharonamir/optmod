import re

from optmod.schemas import Features, OpenAIChatRequest

# ── Compiled at import time ──────────────────────────────────────
_TASK: dict[str, re.Pattern] = {
    "tool_call": re.compile(r"\{|\[|call|invoke|function|tool", re.I),
    "code":      re.compile(r"implement|debug|refactor|codebase|script|test", re.I),
    "reason":    re.compile(r"why|analyze|reason|compare|plan|evaluate|explain", re.I),
    "extract":   re.compile(r"extract|parse|convert|format|structure|table", re.I),
    "summarize": re.compile(r"summarize|summary|digest|brief|tldr|overview", re.I),
    "search":    re.compile(r"search|find|lookup|retrieve|research|crawl", re.I),
}

_DIFF: dict[str, re.Pattern] = {
    "hard":   re.compile(r"undocumented|multi.round|complex|autonomous|adversarial", re.I),
    "medium": re.compile(r"multi.step|pipeline|structured|long.context|cross", re.I),
}

_HE: re.Pattern = re.compile(r"[א-ת]")


def _message_text(content: str | list | None) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
            if not isinstance(part, dict) or part.get("type") == "text"
        )
    return ""


class FeatureExtractor:
    def extract(self, req: OpenAIChatRequest) -> Features:
        last_user: str = next(
            (
                _message_text(m.content)
                for m in reversed(req.messages)
                if m.role == "user" and m.content is not None
            ),
            "",
        )
        text = last_user.lower()

        task_type = next(
            (t for t, pat in _TASK.items() if pat.search(text)),
            "general",
        )

        difficulty = next(
            (d for d, pat in _DIFF.items() if pat.search(text)),
            "easy",
        )

        token_count = int(
            sum(len(str(m.content or "").split()) for m in req.messages) * 1.3
        )

        return Features(
            task_type=task_type,
            difficulty=difficulty,
            token_count=token_count,
            has_tools=bool(req.tools),
            language="he" if _HE.search(last_user) else "en",
            last_user_message=last_user,
        )
