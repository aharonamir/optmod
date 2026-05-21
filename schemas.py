from __future__ import annotations

import json
import dataclasses
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict


class ChatMessage(BaseModel):
    role:         str
    content:      str | list | None = None
    tool_calls:   list[dict] | None = None
    tool_call_id: str | None = None
    name:         str | None = None
    model_config  = ConfigDict(extra="allow")


class OpenAIChatRequest(BaseModel):
    model:       str = "optmod-router"
    messages:    list[ChatMessage]
    tools:       list[dict] | None = None
    tool_choice: str | dict | None = None
    stream:      bool = False
    temperature: float | None = None
    max_tokens:  int | None = None
    model_config = ConfigDict(extra="allow")


@dataclass
class Features:
    task_type:         str
    difficulty:        str
    token_count:       int
    has_tools:         bool
    language:          str
    last_user_message: str


@dataclass
class RoutingContext:
    request:         OpenAIChatRequest
    features:        Features
    session_id:      str
    registry:        "ModelRegistry"
    attempt_number:  int = 0
    last_error_type: str | None = None
    models_tried:    list[str] = field(default_factory=list)


@dataclass
class RoutingDecision:
    model:       "ModelConfig"
    mutator:     str
    reason:      str
    confidence:  float
    router_name: str


@dataclass
class LogEntry:
    ts:                str
    session_id:        str
    task_type:         str
    difficulty:        str
    token_count:       int
    has_tools:         bool
    language:          str
    router:            str
    decision_model:    str
    decision_reason:   str
    confidence:        float
    mutator:           str
    models_tried:      list[str]
    escalation_count:  int
    final_model:       str
    ok:                bool
    error_type:        str | None
    latency_ms:        float
    prompt_tokens:     int
    completion_tokens: int

    def to_jsonl(self) -> str:
        return json.dumps(dataclasses.asdict(self))
