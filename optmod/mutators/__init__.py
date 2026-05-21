from abc import ABC, abstractmethod

from ..schemas import ChatMessage, RoutingDecision


class BaseContextMutator(ABC):
    @abstractmethod
    def mutate(
        self,
        messages: list[ChatMessage],
        decision: RoutingDecision,
    ) -> list[ChatMessage]:
        """Must return a new list. Never modify messages in-place."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__
