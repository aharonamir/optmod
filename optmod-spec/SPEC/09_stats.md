# 09 — Stats API + Routing Log (stats.py, log.py)

## RoutingLog (log.py)

```python
import threading
from pathlib import Path
from .schemas import LogEntry


class RoutingLog:
    def __init__(self, path: str = "routing.log.jsonl") -> None:
        self._path = Path(path)
        self._lock = threading.Lock()   # JSONL append must be atomic

    def append(self, entry: LogEntry) -> None:
        line = entry.to_jsonl() + "\n"
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line)

    def flush(self) -> None:
        pass   # append mode, no buffer to flush
```

## Stats API (stats.py)

Mount as an APIRouter in main.py:
```python
from .stats import stats_router
app.include_router(stats_router)
```

### GET /api/stats

Query param: `range` (str, default "200")

Range values:
- `"50"`, `"200"` → last N records
- `"1h"`, `"6h"`, `"24h"` → time-based window
- `"all"` → all records

Response shape:

```json
{
  "total_requests":    1842,
  "rpm":               3.7,
  "success_rate":      0.934,
  "success_count":     1720,
  "escalation_rate":   0.087,
  "escalation_count":  160,
  "error_rate":        0.066,
  "error_count":       122,
  "router":            "RuleBasedRouter",
  "model_tiers": {
    "qwen2.5:7b":  "fast",
    "qwen3:8b":    "reasoning",
    "deepseek-v4": "oracle"
  },
  "model_distribution": {
    "qwen2.5:7b":  980,
    "qwen3:8b":    654,
    "deepseek-v4": 208
  },
  "task_distribution": {
    "extract": 420,
    "tool_call": 380,
    "reason": 310
  },
  "latency_buckets": {
    "lt100": 120, "lt500": 680, "lt1000": 540,
    "lt3000": 380, "lt10000": 100, "gt10000": 22
  },
  "latency_stats": {
    "p50": 480.0, "p90": 2100.0, "p99": 7800.0, "avg": 920.0
  },
  "escalation_flows": [
    {"model": "qwen2.5:7b", "count": 160},
    {"model": "qwen3:8b",   "count": 98},
    {"model": "deepseek-v4","count": 42}
  ],
  "escalation_reasons": {
    "rate_limit": 88,
    "timeout":    48,
    "server":     24
  },
  "recent_requests": [
    {
      "ts":               "2026-05-20T14:32:01.123Z",
      "task_type":        "reason",
      "difficulty":       "medium",
      "final_model":      "qwen3:8b",
      "latency_ms":       1842.3,
      "prompt_tokens":    980,
      "completion_tokens":260,
      "escalation_count": 0,
      "ok":               true,
      "error_type":       null,
      "confidence":       0.9,
      "router":           "RuleBasedRouter"
    }
  ]
}
```

### GET /api/stats/live

Lightweight endpoint for header status bar:

```json
{
  "total": 1842,
  "recent_ok": 47,
  "recent_total": 50,
  "router": "RuleBasedRouter"
}
```

## Tier inference (in stats.py)

When building `model_tiers` from JSONL records, infer tier from model name:

```python
def _infer_tier(model_name: str) -> str:
    name = model_name.lower()
    if "qwen2.5" in name or "2.5" in name:
        return "fast"
    if "qwen3" in name or "glm" in name:
        return "reasoning"
    return "oracle"
```

This is a fallback. In Phase 2 you can enrich from the registry directly.

## Percentile calculation

```python
def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = (len(s) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)
```
