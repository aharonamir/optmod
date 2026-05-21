from optmod.mutators import BaseContextMutator
from optmod.schemas import ChatMessage, RoutingDecision


class ThinkingModeMutator(BaseContextMutator):
    def mutate(
        self,
        messages: list[ChatMessage],
        decision: RoutingDecision,
    ) -> list[ChatMessage]:
        if not decision.model.thinking_mode:
            return messages

        prefix = "/think" if decision.mutator == "thinking_mode" else "/no_think"

        result = list(messages)
        for i in range(len(result) - 1, -1, -1):
            if result[i].role == "user" and isinstance(result[i].content, str):
                original = result[i].content
                result[i] = result[i].model_copy(
                    update={"content": f"{prefix}\n{original}"}
                )
                break
        return result
