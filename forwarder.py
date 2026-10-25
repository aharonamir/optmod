import json
import logging
import re
import time

import httpx

from optmod.registry import ModelConfig
from optmod.schemas import OpenAIChatRequest, ChatMessage


def retry_after_s(headers, body_text: str = "", now: float | None = None) -> float | None:
    """Seconds until the provider says the model is usable again, if it says.

    Reads `Retry-After` (seconds) and `X-RateLimit-Reset` (epoch ms or s, or a delta in
    seconds) from the response headers, then from OpenRouter's error body
    (`error.metadata.headers`). Returns None when nothing usable is present.
    """
    now = time.time() if now is None else now

    def parse(h) -> float | None:
        try:
            ra = h.get("retry-after") or h.get("Retry-After")
            if ra is not None:
                return max(0.0, float(ra))
            reset = h.get("x-ratelimit-reset") or h.get("X-RateLimit-Reset")
            if reset is None:
                return None
            v = float(reset)
            if v > 1e11:        # epoch milliseconds
                return max(0.0, v / 1000.0 - now)
            if v > 1e9:         # epoch seconds
                return max(0.0, v - now)
            return max(0.0, v)  # already a delta
        except (TypeError, ValueError):
            return None

    got = parse(headers or {})
    if got is not None:
        return got
    try:
        meta = (json.loads(body_text).get("error") or {}).get("metadata") or {}
        return parse(meta.get("headers") or {})
    except Exception:
        return None


_OVERFLOW = re.compile(r"context.length|context window|too many tokens|prompt is too long", re.I)
_REQUESTED = re.compile(r"(?:requested|resulted in|contains|have)\D{0,25}?(\d[\d,]{3,})\s*(?:total\s+)?tokens", re.I)


def _context_overflow_tokens(body_text: str) -> int | None:
    """Tokens the request needed if a 400 body is a context-overflow error (0 = overflow, size unknown)."""
    if not _OVERFLOW.search(body_text or ""):
        return None
    m = _REQUESTED.search(body_text)
    return int(m.group(1).replace(",", "")) if m else 0


class ModelForwarder:
    def __init__(self) -> None:
        self._clients: dict[str, httpx.AsyncClient] = {}

    def _get_client(self, base_url: str, api_key: str) -> httpx.AsyncClient:
        if base_url not in self._clients:
            self._clients[base_url] = httpx.AsyncClient(
                base_url=base_url,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=None,
            )
        return self._clients[base_url]

    async def forward(
        self,
        model: ModelConfig,
        req: OpenAIChatRequest,
        messages: list[ChatMessage],
        info: dict | None = None,
    ) -> tuple[dict, str | None]:
        """info: optional out-param; on a provider error gets `retry_after_s` when advertised."""
        client = self._get_client(model.base_url, model.api_key)
        payload = {
            "model":    model.api_model_name or model.name,
            "messages": [m.model_dump(exclude_none=True) for m in messages],
            **req.model_dump(
                exclude={"model", "messages", "stream", "stream_options"},
                exclude_none=True,
            ),
            "stream": False,
        }

        try:
            r = await client.post(
                "/chat/completions",
                json=payload,
                timeout=model.timeout_s,
            )
            if r.status_code == 200:
                body  = r.json()
                usage = body.get("usage") or {}
                cached = (
                    (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                    or usage.get("prompt_cache_hit_tokens")
                    or usage.get("cache_read_input_tokens")
                    or 0
                )
                body.setdefault("usage", {})["_optmod_cached_tokens"] = int(cached)
                return body, None
            error_type = self._classify(r.status_code)
            if r.status_code == 400:
                overflow = _context_overflow_tokens(r.text)
                if overflow is not None:
                    error_type = "context_overflow"
                    if info is not None and overflow > 0:
                        info["context_tokens"] = overflow
            if info is not None:
                ra = retry_after_s(r.headers, r.text)
                if ra is not None:
                    info["retry_after_s"] = ra
            logging.warning(
                f"[optmod] {model.name} HTTP {r.status_code} ({error_type}): {r.text[:400]}"
            )
            return {}, error_type

        except httpx.TimeoutException:
            return {}, "timeout"
        except httpx.RequestError:
            return {}, "server"
        except Exception:
            return {}, "server"

    def _classify(self, status: int) -> str:
        mapping = {429: "rate_limit", 401: "auth", 403: "auth", 400: "bad_request",
                   404: "not_found"}
        return mapping.get(status, "server")

    async def close(self) -> None:
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()
