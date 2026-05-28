# -*- coding: utf-8 -*-
"""Message adapters shared by model clients."""

from __future__ import annotations

import json
from typing import Any, ClassVar

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


class MessageAdapter:
    """Convert LangChain messages to OpenAI-compatible chat message dicts."""

    TYPE_TO_ROLE: ClassVar[dict[type[BaseMessage], str]] = {
        HumanMessage: "user",
        AIMessage: "assistant",
        SystemMessage: "system",
        FunctionMessage: "function",
        ToolMessage: "tool",
    }

    @classmethod
    def to_openai_messages(
        cls,
        messages: list[dict[str, Any] | BaseMessage],
    ) -> list[dict[str, Any]]:
        if not messages:
            return []
        if isinstance(messages[0], dict):
            return [dict(msg) for msg in messages if isinstance(msg, dict)]
        return [cls._convert_message(msg) for msg in messages if isinstance(msg, BaseMessage)]

    @classmethod
    def _convert_message(cls, msg: BaseMessage) -> dict[str, Any]:
        role = cls.TYPE_TO_ROLE.get(type(msg))
        if role is None:
            raise ValueError(f"Unknown message type: {type(msg)}")

        message: dict[str, Any] = {"role": role, "content": msg.content or ""}
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            message["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(
                            tc.get("args", {}),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    },
                }
                for tc in msg.tool_calls
            ]
            message["content"] = msg.content or None
        if isinstance(msg, ToolMessage):
            message["tool_call_id"] = msg.tool_call_id
        return message

