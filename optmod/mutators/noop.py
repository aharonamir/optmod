from . import BaseContextMutator
from ..schemas import ChatMessage, RoutingDecision


class NoopMutator(BaseContextMutator):
    def mutate(
        self,
        messages: list[ChatMessage],
        decision: RoutingDecision,
    ) -> list[ChatMessage]:
        return messages
