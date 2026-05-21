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
                timeout=None,
            )
        return self._clients[base_url]

    async def forward(
        self,
        model: ModelConfig,
        req: OpenAIChatRequest,
        messages: list[ChatMessage],
    ) -> tuple[dict, str | None]:
        client = self._get_client(model.base_url, model.api_key)
        payload = {
            "model":    model.name,
            "messages": [m.model_dump(exclude_none=True) for m in messages],
            **req.model_dump(
                exclude={"model", "messages", "stream"},
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
