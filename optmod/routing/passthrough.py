from . import BaseRouter
from .context import RoutingContext
from ..schemas import RoutingDecision


class PassthroughRouter(BaseRouter):
    def route(self, ctx: RoutingContext) -> RoutingDecision:
        return self._passthrough(ctx, "passthrough: primary model, no routing applied")
