# 03 — Feature Extractor (features.py)

## Rules

- All regex patterns MUST be compiled at **module level** (not inside functions).
- `extract()` must complete in **<1ms**. No file I/O, no network, no ML.
- Hebrew detection uses Unicode range `\u05d0-\u05ea` (alef to tav).
- Token count is estimated: whitespace-split word count × 1.3, cast to int.
- Task type: **first match wins** in the order defined in `_TASK`.
- Difficulty: **first match wins** in the order defined in `_DIFF`.

## Implementation

```python
import re
from .schemas import Features, OpenAIChatRequest

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

_HE: re.Pattern = re.compile(r"[\u05d0-\u05ea]")


class FeatureExtractor:
    def extract(self, req: OpenAIChatRequest) -> Features:
        # Last user message text (string content only)
        last_user: str = next(
            (
                m.content
                for m in reversed(req.messages)
                if m.role == "user" and isinstance(m.content, str)
            ),
            "",
        )
        text = last_user.lower()

        # Task type: first matching key wins
        task_type = next(
            (t for t, pat in _TASK.items() if pat.search(text)),
            "general",
        )

        # Difficulty: first matching key wins
        difficulty = next(
            (d for d, pat in _DIFF.items() if pat.search(text)),
            "easy",
        )

        # Token estimate
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
```

## Test vectors (used in test_features.py)

| Prompt | task_type | difficulty |
|---|---|---|
| "why does this algorithm fail?" | reason | easy |
| "implement a binary search tree" | code | easy |
| "extract all emails from this document" | extract | easy |
| "summarize this PDF in 3 bullet points" | summarize | easy |
| "debug this undocumented multi-round agent" | code | hard |
| "analyze and compare these three approaches" | reason | easy |
| "search for recent papers on routing" | search | easy |
| "call the get_weather function with city=Paris" | tool_call | easy |
| "שלום, תסכם את המסמך הזה" (Hebrew summarize) | summarize | easy |
| prompt with 5000+ tokens | any | medium or hard based on content |
