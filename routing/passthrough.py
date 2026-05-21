from optmod.routing import BaseRouter
from optmod.routing.context import RoutingContext
from optmod.schemas import RoutingDecision


class PassthroughRouter(BaseRouter):
    def route(self, ctx: RoutingContext) -> RoutingDecision:
        return self._passthrough(ctx, "passthrough: primary model, no routing applied")
