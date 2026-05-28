# optmod/stats.py
# ─────────────────────────────────────────────────────────────────
# Reads routing.log.jsonl and returns aggregated stats for the UI.
# Mounted into main.py:
#   app.include_router(stats_router)
#   app.mount("/ui", StaticFiles(...), name="ui")
# ─────────────────────────────────────────────────────────────────

import json
import time
import statistics
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

stats_router = APIRouter()

LOG_PATH = Path("routing.log.jsonl")


# ── LOG READER ────────────────────────────────────────────────────

def _read_log() -> list[dict]:
    """Read all JSONL records. Returns [] if file missing."""
    if not LOG_PATH.exists():
        return []
    records = []
    with LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _filter_by_range(records: list[dict], range_param: str) -> list[dict]:
    """
    range_param:
      integer str  → last N records (e.g. "50", "200")
      "1h"         → last 1 hour
      "6h"         → last 6 hours
      "24h"        → last 24 hours
      "all"        → all records
    """
    if range_param == "all":
        return records

    # Time-based ranges
    time_map = {"1h": 3600, "6h": 21600, "24h": 86400}
    if range_param in time_map:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=time_map[range_param])
        filtered = []
        for r in records:
            try:
                ts = datetime.fromisoformat(r["ts"].replace("Z", "+00:00"))
                if ts >= cutoff:
                    filtered.append(r)
            except (KeyError, ValueError):
                continue
        return filtered

    # Last-N records
    try:
        n = int(range_param)
        return records[-n:]
    except (ValueError, TypeError):
        return records[-200:]


# ── PERCENTILE ────────────────────────────────────────────────────

def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (len(sorted_data) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (idx - lo)


# ── STATS AGGREGATION ─────────────────────────────────────────────

def _aggregate(records: list[dict], all_records: list[dict]) -> dict[str, Any]:
    total = len(records)

    if total == 0:
        return {
            "total_requests": 0,
            "rpm": 0.0,
            "success_rate": 0.0,
            "success_count": 0,
            "escalation_rate": 0.0,
            "escalation_count": 0,
            "error_rate": 0.0,
            "error_count": 0,
            "router": "—",
            "model_tiers": {},
            "model_distribution": {},
            "task_distribution": {},
            "latency_buckets": {},
            "latency_stats": {"p50": 0, "p90": 0, "p99": 0, "avg": 0},
            "escalation_flows": [],
            "escalation_reasons": {},
            "recent_requests": [],
        }

    # Basic counts
    success_count    = sum(1 for r in records if r.get("ok", True))
    escalation_count = sum(1 for r in records if r.get("escalation_count", 0) > 0)
    error_count      = total - success_count

    # RPM: based on time span of records
    rpm = 0.0
    try:
        timestamps = []
        for r in records:
            if "ts" in r:
                ts = datetime.fromisoformat(r["ts"].replace("Z", "+00:00"))
                timestamps.append(ts.timestamp())
        if len(timestamps) > 1:
            span_s = max(timestamps) - min(timestamps)
            if span_s > 0:
                rpm = len(records) / (span_s / 60)
    except Exception:
        pass

    # Most recent router
    router = "—"
    for r in reversed(records):
        if r.get("router"):
            router = r["router"]
            break

    # Model distribution
    model_dist: dict[str, int] = {}
    model_tiers: dict[str, str] = {}
    for r in records:
        model = r.get("final_model") or r.get("decision_model", "unknown")
        model_dist[model] = model_dist.get(model, 0) + 1
        # Infer tier from model name (can be enriched from registry later)
        if model not in model_tiers:
            if "qwen2.5" in model or "2.5" in model:
                model_tiers[model] = "fast"
            elif "qwen3" in model or "glm" in model:
                model_tiers[model] = "reasoning"
            else:
                model_tiers[model] = "oracle"

    # Task distribution
    task_dist: dict[str, int] = {}
    for r in records:
        t = r.get("task_type", "general")
        task_dist[t] = task_dist.get(t, 0) + 1

    # Latency
    latencies = [r["latency_ms"] for r in records if "latency_ms" in r and r["latency_ms"] > 0]
    buckets = {"lt100": 0, "lt500": 0, "lt1000": 0, "lt3000": 0, "lt10000": 0, "gt10000": 0}
    for ms in latencies:
        if ms < 100:       buckets["lt100"]   += 1
        elif ms < 500:     buckets["lt500"]   += 1
        elif ms < 1000:    buckets["lt1000"]  += 1
        elif ms < 3000:    buckets["lt3000"]  += 1
        elif ms < 10000:   buckets["lt10000"] += 1
        else:              buckets["gt10000"] += 1

    lat_stats = {"p50": 0.0, "p90": 0.0, "p99": 0.0, "avg": 0.0}
    if latencies:
        lat_stats = {
            "p50": round(_percentile(latencies, 50), 1),
            "p90": round(_percentile(latencies, 90), 1),
            "p99": round(_percentile(latencies, 99), 1),
            "avg": round(statistics.mean(latencies), 1),
        }

    # Escalation flows: count per model how many times it was the first attempted model
    # and how many times it was the final model after escalation
    esc_records = [r for r in records if r.get("escalation_count", 0) > 0]
    esc_flow_counts: dict[str, int] = {}
    esc_reasons: dict[str, int] = {}

    for r in esc_records:
        tried = r.get("models_tried", [])
        for m in tried:
            esc_flow_counts[m] = esc_flow_counts.get(m, 0) + 1
        err = r.get("error_type")
        if err:
            esc_reasons[err] = esc_reasons.get(err, 0) + 1

    # Build ordered escalation flow (fast → reasoning → oracle)
    tier_order = {"fast": 0, "reasoning": 1, "oracle": 2}
    esc_flow = sorted(
        [{"model": m, "count": c} for m, c in esc_flow_counts.items()],
        key=lambda x: tier_order.get(model_tiers.get(x["model"], "oracle"), 99)
    )

    # Recent requests (last 20, newest first)
    recent = list(reversed(records[-20:]))

    return {
        "total_requests":    total,
        "rpm":               round(rpm, 2),
        "success_rate":      round(success_count / total, 4) if total else 0,
        "success_count":     success_count,
        "escalation_rate":   round(escalation_count / total, 4) if total else 0,
        "escalation_count":  escalation_count,
        "error_rate":        round(error_count / total, 4) if total else 0,
        "error_count":       error_count,
        "router":            router,
        "model_tiers":       model_tiers,
        "model_distribution":model_dist,
        "task_distribution": task_dist,
        "latency_buckets":   buckets,
        "latency_stats":     lat_stats,
        "escalation_flows":  esc_flow,
        "escalation_reasons":esc_reasons,
        "recent_requests":   recent,
    }


# ── API ENDPOINT ──────────────────────────────────────────────────

@stats_router.get("/api/stats")
async def get_stats(range: str = Query(default="200")):
    """
    Query params:
      range: "50" | "200" | "1h" | "6h" | "24h" | "all"
    """
    all_records = _read_log()
    filtered    = _filter_by_range(all_records, range)
    stats       = _aggregate(filtered, all_records)
    return JSONResponse(content=stats)


@stats_router.get("/api/stats/live")
async def get_live():
    """Lightweight endpoint: just the current counts. For polling status bar."""
    all_records = _read_log()
    recent = all_records[-50:]
    total = len(all_records)
    ok = sum(1 for r in recent if r.get("ok", True))
    router = next((r["router"] for r in reversed(recent) if r.get("router")), "—")
    return JSONResponse(content={
        "total": total,
        "recent_ok": ok,
        "recent_total": len(recent),
        "router": router,
    })
