# 01 — Schemas (schemas.py)

All types live in `schemas.py`. Use Pydantic v2 for HTTP types, dataclasses for internal types.

## ChatMessage

```python
from pydantic import BaseModel, ConfigDict

class ChatMessage(BaseModel):
    role:         str
    content:      str | list | None = None
    tool_calls:   list[dict] | None = None
    tool_call_id: str | None = None
    name:         str | None = None
    model_config  = ConfigDict(extra='allow')
```

## OpenAIChatRequest

```python
class OpenAIChatRequest(BaseModel):
    model:       str = "optmod-router"   # ignored internally
    messages:    list[ChatMessage]
    tools:       list[dict] | None = None
    tool_choice: str | dict | None = None
    stream:      bool = False
    temperature: float | None = None
    max_tokens:  int | None = None
    model_config = ConfigDict(extra='allow')  # forward unknown fields unchanged
```

## Features (dataclass)

```python
from dataclasses import dataclass

@dataclass
class Features:
    task_type:         str    # reason|code|extract|summarize|tool_call|search|general
    difficulty:        str    # easy|medium|hard
    token_count:       int    # estimated BPE tokens across all messages
    has_tools:         bool   # True if tools defined in request
    language:          str    # 'he' | 'en' | 'other'
    last_user_message: str    # raw text of last user turn
```

## RoutingContext (dataclass)

```python
from dataclasses import dataclass, field

@dataclass
class RoutingContext:
    request:          OpenAIChatRequest
    features:         Features
    session_id:       str
    registry:         "ModelRegistry"     # forward ref, imported at runtime
    attempt_number:   int = 0
    last_error_type:  str | None = None   # rate_limit|auth|server|timeout|bad_request
    models_tried:     list[str] = field(default_factory=list)
```

## RoutingDecision (dataclass)

```python
@dataclass
class RoutingDecision:
    model:       "ModelConfig"   # forward ref
    mutator:     str             # 'noop' | 'thinking_mode'
    reason:      str             # log-friendly string
    confidence:  float           # 0.0–1.0
    router_name: str
```

## LogEntry (dataclass)

```python
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
        import json, dataclasses
        return json.dumps(dataclasses.asdict(self))
```
