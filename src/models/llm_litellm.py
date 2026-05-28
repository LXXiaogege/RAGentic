# -*- coding: utf-8 -*-
"""
@Time：2026/4/2
@Auth：Plan
@File：llm_litellm.py
@IDE：PyCharm

LiteLLM 客户端封装 - 兼容现有 LLMWrapper 接口
"""

from typing import Any, Dict, List, Union

from litellm import acompletion, completion
from langfuse.decorators import observe
from langchain_core.messages import BaseMessage

from src.configs.logger_config import setup_logger
from src.models.message_adapter import MessageAdapter
from src.models.response_utils import extract_text_response

logger = setup_logger(__name__)


class LiteLLMClient:
    """LiteLLM 封装类 - 兼容 OpenAI SDK 返回格式"""

    def __init__(self, api_key: str, base_url: str, provider: str = "openai"):
        self.logger = setup_logger(f"{__name__}.LiteLLMClient")
        self.api_key = api_key
        self.base_url = base_url
        self.provider = provider

    def _prepare_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """准备 litellm 调用参数"""
        kwargs = dict(kwargs)
        extra_body = dict(kwargs.pop("extra_body", {}) or {})

        chat_template_kwargs = dict(extra_body.pop("chat_template_kwargs", {}) or {})

        if "enable_thinking" in chat_template_kwargs:
            kwargs["thinking"] = chat_template_kwargs["enable_thinking"]

        kwargs.update(extra_body)
        return kwargs

    def _prepare_request(
        self, model: str | None, kwargs: Dict[str, Any]
    ) -> tuple[str, Dict[str, Any]]:
        model = model or ""
        prepared_kwargs = dict(kwargs)

        if self.provider == "minimax":
            if model.startswith("minimax/"):
                model = model[len("minimax/") :]
            prepared_kwargs["custom_llm_provider"] = "openai"
            prepared_kwargs["drop_params"] = True
        elif self.provider != "openai" and not model.startswith(self.provider + "/"):
            model = f"{self.provider}/{model}"

        prepared_kwargs = self._prepare_kwargs(prepared_kwargs)
        prepared_kwargs["api_key"] = self.api_key
        if self.base_url:
            prepared_kwargs["api_base"] = self.base_url
        return model, prepared_kwargs

    @observe(name="LiteLLM.chat", as_type="generation")
    def chat(
        self,
        model: str,
        messages: List[Dict],
        stream: bool = False,
        **kwargs,
    ):
        self.logger.info(f"[LiteLLM] chat request: model={model}, stream={stream}")
        model, prepared_kwargs = self._prepare_request(model, kwargs)

        try:
            response = completion(
                model=model,
                messages=messages,
                stream=stream,
                **prepared_kwargs,
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
        model, prepared_kwargs = self._prepare_request(model, kwargs)

        try:
            response = await acompletion(
                model=model,
                messages=messages,
                stream=stream,
                **prepared_kwargs,
            )
            self.logger.info("[LiteLLM] achat response received")
            return response
        except Exception as e:
            self.logger.exception(f"[LiteLLM] achat error: {str(e)}")
            raise


class LiteLLMWrapper:
    """LiteLLM 版本的 LLMWrapper - 完全兼容原有接口"""

    def __init__(self, api_key: str, base_url: str, provider: str = "openai"):
        self.logger = setup_logger(f"{__name__}.LiteLLMWrapper")
        self.client = LiteLLMClient(api_key, base_url, provider)
        self.provider = provider

    def convert_messages_to_dicts(self, messages: List[BaseMessage]) -> List[Dict]:
        return MessageAdapter.to_openai_messages(messages)

    def _prepare_call(
        self, messages: List[Union[Dict, BaseMessage]], kwargs: dict
    ) -> tuple[str | None, list[dict]]:
        if messages and isinstance(messages[0], BaseMessage):
            messages = self.convert_messages_to_dicts(messages)
        model = kwargs.pop("model", None)
        return model, messages

    def chat(
        self,
        messages: List[Union[Dict, BaseMessage]],
        stream: bool = False,
        return_raw: bool = False,
        **kwargs,
    ):
        model, messages_to_send = self._prepare_call(messages, kwargs)

        response = self.client.chat(
            model=model,
            messages=messages_to_send,
            stream=stream,
            **kwargs,
        )

        if return_raw or stream:
            return response
        return extract_text_response(response)

    async def achat(
        self,
        messages: List[Union[Dict, BaseMessage]],
        stream: bool = False,
        return_raw: bool = False,
        **kwargs,
    ):
        model, messages_to_send = self._prepare_call(messages, kwargs)

        response = await self.client.achat(
            model=model,
            messages=messages_to_send,
            stream=stream,
            **kwargs,
        )

        if return_raw or stream:
            return response
        return extract_text_response(response)
