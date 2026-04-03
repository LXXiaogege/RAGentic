# -*- coding: utf-8 -*-
"""
@Time：2026/4/2
@Auth：Plan
@File：llm_litellm.py
@IDE：PyCharm

LiteLLM 客户端封装 - 兼容现有 LLMWrapper 接口
"""

from typing import Any, Dict, List, Optional, Union, Type, cast
import json
import litellm
from litellm import acompletion, completion
from langfuse.decorators import observe
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    FunctionMessage,
    ToolMessage,
)
from src.configs.logger_config import setup_logger

logger = setup_logger(__name__)


class LiteLLMClient:
    """LiteLLM 封装类 - 兼容 OpenAI SDK 返回格式"""

    def __init__(self, api_key: str, base_url: str, provider: str = "openai"):
        self.logger = setup_logger(f"{__name__}.LiteLLMClient")
        self.api_key = api_key
        self.base_url = base_url
        self.provider = provider

        litellm.api_key = api_key
        if base_url:
            litellm.api_base = base_url

        if provider == "minimax":
            litellm.drop_params = True

    def _prepare_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """准备 litellm 调用参数"""
        extra_body = kwargs.pop("extra_body", {}) or {}

        chat_template_kwargs = extra_body.pop("chat_template_kwargs", {}) or {}

        if "enable_thinking" in chat_template_kwargs:
            kwargs["thinking"] = chat_template_kwargs["enable_thinking"]

        kwargs.update(extra_body)
        return kwargs

    @observe(name="LiteLLM.chat", as_type="generation")
    def chat(
        self,
        model: str,
        messages: List[Dict],
        stream: bool = False,
        **kwargs,
    ):
        self.logger.info(f"[LiteLLM] chat request: model={model}, stream={stream}")

        if self.provider == "minimax":
            # MiniMax 需要用 OpenAI 兼容模式，传递 custom_llm_provider
            if model.startswith("minimax/"):
                model = model[len("minimax/"):]
            kwargs["custom_llm_provider"] = "openai"
        elif self.provider != "openai" and not model.startswith(self.provider + "/"):
            model = f"{self.provider}/{model}"

        kwargs = self._prepare_kwargs(kwargs)

        try:
            response = completion(
                model=model,
                messages=messages,
                stream=stream,
                **kwargs,
            )
            self.logger.info("[LiteLLM] chat response received")
            return response
        except Exception as e:
            self.logger.exception(f"[LiteLLM] chat error: {str(e)}")
            raise

    @observe(name="LiteLLM.achat", as_type="generation")
    async def achat(
        self,
        model: str,
        messages: List[Dict],
        stream: bool = False,
        **kwargs,
    ):
        self.logger.info(f"[LiteLLM] achat request: model={model}, stream={stream}")

        if self.provider == "minimax":
            # MiniMax 需要用 OpenAI 兼容模式，传递 custom_llm_provider
            if model.startswith("minimax/"):
                model = model[len("minimax/"):]
            kwargs["custom_llm_provider"] = "openai"
        elif self.provider != "openai" and not model.startswith(self.provider + "/"):
            model = f"{self.provider}/{model}"

        kwargs = self._prepare_kwargs(kwargs)

        try:
            response = await acompletion(
                model=model,
                messages=messages,
                stream=stream,
                **kwargs,
            )
            self.logger.info("[LiteLLM] achat response received")
            return response
        except Exception as e:
            self.logger.exception(f"[LiteLLM] achat error: {str(e)}")
            raise


class LiteLLMWrapper:
    """LiteLLM 版本的 LLMWrapper - 完全兼容原有接口"""

    TYPE_TO_ROLE: Dict[Type[BaseMessage], str] = {
        HumanMessage: "user",
        AIMessage: "assistant",
        SystemMessage: "system",
        FunctionMessage: "function",
        ToolMessage: "tool",
    }

    def __init__(self, api_key: str, base_url: str, provider: str = "openai"):
        self.logger = setup_logger(f"{__name__}.LiteLLMWrapper")
        self.client = LiteLLMClient(api_key, base_url, provider)
        self.provider = provider

    def convert_messages_to_dicts(self, messages: List[BaseMessage]) -> List[Dict]:
        formatted = []
        for msg in messages:
            role = self.TYPE_TO_ROLE.get(type(msg))
            if role is None:
                raise ValueError(f"Unknown message type: {type(msg)}")
            d = {"role": role, "content": msg.content or ""}
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                d["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"]),
                        },
                    }
                    for tc in msg.tool_calls
                ]
                d["content"] = msg.content or None
            if isinstance(msg, ToolMessage):
                d["tool_call_id"] = msg.tool_call_id
            formatted.append(d)
        return formatted

    def chat(
        self,
        messages: List[Union[Dict, BaseMessage]],
        stream: bool = False,
        return_raw: bool = False,
        **kwargs,
    ):
        if messages and isinstance(messages[0], BaseMessage):
            messages = self.convert_messages_to_dicts(messages)

        model = kwargs.pop("model", None)

        response = self.client.chat(
            model=model,
            messages=messages,
            stream=stream,
            **kwargs,
        )

        if return_raw or stream:
            return response
        if not response.choices or response.choices[0].message.content is None:
            raise ValueError("LLM 返回了空的 choices 或 content，请检查模型服务")
        return response.choices[0].message.content

    async def achat(
        self,
        messages: List[Union[Dict, BaseMessage]],
        stream: bool = False,
        return_raw: bool = False,
        **kwargs,
    ):
        if messages and isinstance(messages[0], BaseMessage):
            messages = self.convert_messages_to_dicts(messages)

        model = kwargs.pop("model", None)

        response = await self.client.achat(
            model=model,
            messages=messages,
            stream=stream,
            **kwargs,
        )

        if return_raw or stream:
            return response
        if not response.choices or response.choices[0].message.content is None:
            raise ValueError("LLM 返回了空的 choices 或 content，请检查模型服务")
        return response.choices[0].message.content
