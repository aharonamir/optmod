# 07 — Model Forwarder (forwarder.py)

## Rules

- One `httpx.AsyncClient` per `base_url`, stored in `self._clients` dict.
- Clients are created lazily on first use.
- All clients closed in `close()` called from FastAPI lifespan shutdown.
- Never create a new client per request.
- Timeout per model is `model.timeout_s`.
- Non-streaming only in v1. If `req.stream=True`, forward as non-streaming anyway.

## Implementation

```python
import httpx
from .registry import ModelConfig
from .schemas import OpenAIChatRequest, ChatMessage


class ModelForwarder:
    def __init__(self) -> None:
        self._clients: dict[str, httpx.AsyncClient] = {}

    def _get_client(self, base_url: str, api_key: str) -> httpx.AsyncClient:
        if base_url not in self._clients:
            self._clients[base_url] = httpx.AsyncClient(
                base_url=base_url,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=None,   # per-request timeout set in post()
            )
        return self._clients[base_url]

    async def forward(
        self,
        model: ModelConfig,
        req: OpenAIChatRequest,
        messages: list[ChatMessage],
    ) -> tuple[dict, str | None]:
        """
        Returns: (response_dict, error_type | None)

        error_type values:
          'rate_limit' | 'auth' | 'bad_request' | 'timeout' | 'server'
          None = success
        """
        client  = self._get_client(model.base_url, model.api_key)
        payload = {
            "model":    model.name,
            "messages": [m.model_dump(exclude_none=True) for m in messages],
            # Forward all original fields except model and messages
            **req.model_dump(
                exclude={"model", "messages", "stream"},
                exclude_none=True,
            ),
            "stream": False,   # always non-streaming in v1
        }

        try:
            r = await client.post(
                "/chat/completions",
                json=payload,
                timeout=model.timeout_s,
            )
            if r.status_code == 200:
                return r.json(), None
            return {}, self._classify(r.status_code)

        except httpx.TimeoutException:
            return {}, "timeout"
        except httpx.RequestError:
            return {}, "server"
        except Exception:
            return {}, "server"

    def _classify(self, status: int) -> str:
        mapping = {429: "rate_limit", 401: "auth", 403: "auth", 400: "bad_request"}
        return mapping.get(status, "server")

    async def close(self) -> None:
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()
```
