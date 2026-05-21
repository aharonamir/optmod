import logging

from . import BaseRouter
from .rule_based import RuleBasedRouter
from .context import RoutingContext
from ..schemas import RoutingDecision


class DecisionTreeRouter(BaseRouter):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        pkl_path = config.get("policy_path", "routing_policy.pkl")
        self._fallback = RuleBasedRouter(config)
        try:
            import joblib
            data         = joblib.load(pkl_path)
            self.clf     = data["clf"]
            self.cat_enc = data["cat_enc"]
            self.lvl_enc = data["lvl_enc"]
            self._ready  = True
            logging.info(f"[optmod] DecisionTreeRouter loaded: {pkl_path}")
        except FileNotFoundError:
            self._ready = False
            logging.warning(f"[optmod] {pkl_path} not found, using RuleBasedRouter fallback")

    def route(self, ctx: RoutingContext) -> RoutingDecision:
        if not self._ready:
            return self._fallback.route(ctx)
        try:
            f = ctx.features
            X = [[
                self.cat_enc.transform([f.task_type])[0],
                self.lvl_enc.transform([f.difficulty])[0],
                f.token_count,
            ]]
            label   = self.clf.predict(X)[0]
            model   = ctx.registry.get_or_default(label)
            mutator = "thinking_mode" if model.thinking_mode else "noop"
            return RoutingDecision(
                model=model, mutator=mutator,
                reason=f"decision_tree: {label}",
                confidence=0.85, router_name=self.name,
            )
        except Exception as e:
            logging.warning(f"[optmod] DecisionTreeRouter error: {e}, falling back")
            return self._fallback.route(ctx)
