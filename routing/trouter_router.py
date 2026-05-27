import logging
import os
from pathlib import Path

from optmod.routing import BaseRouter
from optmod.routing.context import RoutingContext
from optmod.schemas import RoutingDecision

_WEIGHTS_PATH = Path(__file__).parent / "trouter_weights.pt"
_DEFAULT_COST_WEIGHT = 0.3

# TRouter model index → optmod tier
# Trained on: idx0=arcee (reasoning/free), idx1=deepseek-v4-flash, idx2=openai-120b (oracle)
# Maps positionally to tier 0 (fast), tier 1 (reasoning), tier 2 (oracle)
_IDX_TO_TIER = {0: 0, 1: 1, 2: 2}


class TRouterRouter(BaseRouter):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._ready = False

        alpha_cfg = config.get("trouter_cost_weight")
        alpha_env = os.environ.get("TROUTER_COST_WEIGHT")
        self._cost_weight = float(alpha_cfg or alpha_env or _DEFAULT_COST_WEIGHT)

        try:
            import torch
            from sentence_transformers import SentenceTransformer
            from optmod.routing.train_trouter import build_model
            from optmod.routing.train_trouter import route as _trouter_route

            ckpt = torch.load(_WEIGHTS_PATH, weights_only=False, map_location="cpu")
            cfg = ckpt["config"]

            self._nn = build_model(
                embed_dim=cfg["embed_dim"],
                num_task_types=cfg["num_task_types"],
                num_models=cfg["num_models"],
                hidden_dim=cfg["hidden_dim"],
            )
            self._nn.load_state_dict(ckpt["model_state_dict"])
            self._nn.eval()

            self._prior = ckpt["prior"]
            encoder_name = ckpt.get("encoder_name", "all-MiniLM-L6-v2")
            self._encoder = SentenceTransformer(encoder_name)
            self._route_fn = _trouter_route
            self._torch = torch
            self._ready = True
            logging.info(
                f"[optmod] TRouterRouter loaded (α={self._cost_weight}, "
                f"encoder={encoder_name})"
            )
        except Exception as exc:
            logging.warning(f"[optmod] TRouterRouter failed to load: {exc}")

    def route(self, ctx: RoutingContext) -> RoutingDecision:
        if not self._ready:
            return self._passthrough(ctx, "trouter: not initialised, using primary")
        try:
            return self._route(ctx)
        except Exception as exc:
            logging.warning(f"[optmod] TRouterRouter.route error: {exc}")
            return self._passthrough(ctx, f"trouter error: {exc}")

    def _route(self, ctx: RoutingContext) -> RoutingDecision:
        text = ctx.features.last_user_message or ""

        emb = self._encoder.encode(text, convert_to_tensor=True).cpu()  # [384]

        chosen_idx, scores_adj, _ = self._route_fn(
            self._nn, emb, self._prior, self._cost_weight
        )

        tier = _IDX_TO_TIER.get(int(chosen_idx), 2)
        candidates = ctx.registry.by_tier(tier)
        model = candidates[0] if candidates else ctx.registry.primary

        mutator = "thinking_mode" if model.thinking_mode else "noop"
        adj_score = float(scores_adj[chosen_idx].item())
        confidence = float(
            self._torch.softmax(scores_adj, dim=0)[chosen_idx].item()
        )

        return RoutingDecision(
            model=model,
            mutator=mutator,
            reason=(
                f"trouter: idx={chosen_idx} → tier={tier} ({model.name}) "
                f"score={adj_score:.3f} α={self._cost_weight}"
            ),
            confidence=confidence,
            router_name=self.name,
        )
