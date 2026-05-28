# -*- coding: utf-8 -*-
"""
@Time：2025/4/12 14:17
@Auth：吕鑫
@File：llm.py
@IDE：PyCharm
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union, cast

from langchain_core.messages import BaseMessage
from langfuse.decorators import langfuse_context, observe
from langfuse.openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletionMessageParam

from src.configs.logger_config import setup_logger
from src.configs.model_config import LLMConfig
from src.models.message_adapter import MessageAdapter
from src.models.response_utils import extract_text_response

logger = setup_logger(__name__)


class BaseLLM:
    def chat(
        self,
        model: str,
        messages: List[ChatCompletionMessageParam],
        stream: bool = False,
        **kwargs,
    ):
        raise NotImplementedError

    async def achat(
        self,
        model: str,
        messages: List[ChatCompletionMessageParam],
        stream: bool = False,
        **kwargs,
    ):
        raise NotImplementedError


class OpenAICompatibleLLM(BaseLLM):
    """OpenAI-compatible client for OpenAI and MiniMax compatible endpoints."""

    def __init__(self, api_key: str, base_url: str, provider: str = "openai"):
        self.logger = setup_logger(f"{__name__}.OpenAICompatibleLLM")
        self.provider = provider
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.aclient = AsyncOpenAI(api_key=api_key, base_url=base_url)

    def _normalize_model(self, model: str) -> str:
        if self.provider == "minimax" and model.startswith("minimax/"):
            return model[len("minimax/") :]
        return model

    def _prepare_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        kwargs = dict(kwargs)
        kwargs.pop("return_raw", None)
        if self.provider != "minimax":
            return kwargs

        extra_body = kwargs.get("extra_body")
        if not isinstance(extra_body, dict):
            return kwargs

        chat_template_kwargs = extra_body.get("chat_template_kwargs") or {}
        if "enable_thinking" not in chat_template_kwargs:
            return kwargs

        next_extra_body = dict(extra_body)
        next_extra_body["thinking"] = chat_template_kwargs["enable_thinking"]
        next_extra_body.pop("chat_template_kwargs", None)
        return {**kwargs, "extra_body": next_extra_body}

    def _prepare_request(
        self, model: str, kwargs: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        return self._normalize_model(model), self._prepare_kwargs(kwargs)

    @observe(name="LLMWrapper.chat", as_type="generation")
    def chat(self, model: str, messages: List[dict], stream: bool = False, **kwargs):
        """与 OpenAI-compatible LLM 进行对话。"""
        typed_messages = cast(List[ChatCompletionMessageParam], messages)
        normalized_model, prepared_kwargs = self._prepare_request(model, kwargs)
        try:
            response = self.client.chat.completions.create(
                model=normalized_model,
                messages=typed_messages,
                stream=stream,
                **prepared_kwargs,
            )
            self.logger.info("成功获取模型响应")
            return response
        except Exception as e:
            self.logger.exception(f"调用模型时发生错误: {str(e)}")
            raise

    @observe(name="LLMWrapper.achat", as_type="generation")
    async def achat(
        self, model: str, messages: List[dict], stream: bool = False, **kwargs
    ):
        """与 OpenAI-compatible LLM 进行异步对话。"""
        typed_messages = cast(List[ChatCompletionMessageParam], messages)
        normalized_model, prepared_kwargs = self._prepare_request(model, kwargs)
        try:
            response = await self.aclient.chat.completions.create(
                model=normalized_model,
                messages=typed_messages,
                stream=stream,
                **prepared_kwargs,
            )
            self.logger.info("成功获取模型异步响应")
            return response
        except Exception as e:
            self.logger.exception(f"调用模型时发生错误: {str(e)}")
            raise


class LLMWrapper:
    """统一 LLM 入口，支持 OpenAI-compatible、MiniMax 和 LiteLLM。"""

    def __init__(self, config: LLMConfig):
        self.logger = setup_logger(f"{__name__}.LLMWrapper")
        self.config = config
        self.client = self._build_client()

        self._cache = None
        self._last_cache_hit = False
        if self.config.use_response_cache:
            self._init_cache()

    def _build_client(self):
        provider = self.config.provider.lower()
        if provider == "litellm":
            from src.models.llm_litellm import LiteLLMWrapper

            lite_provider = (
                self.config.model.split("/")[0] if "/" in self.config.model else "openai"
            )
            return LiteLLMWrapper(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                provider=lite_provider,
            )
        if provider in {"openai", "minimax"}:
            return OpenAICompatibleLLM(
                self.config.api_key,
                self.config.base_url,
                provider=provider,
            )
        raise ValueError(f"暂不支持的模型提供商：{self.config.provider}")

    def _init_cache(self):
        """初始化 LLM 响应缓存。"""
        try:
            from src.configs.config import AppConfig
            from src.models.llm_response_cache import LLMResponseCache

            app_config = AppConfig()
            self._cache = LLMResponseCache(
                redis_url=app_config.redis.redis_url,
                embedding_model=app_config.embedding.model,
                embedding_api_key=app_config.embedding.api_key,
                embedding_base_url=app_config.embedding.base_url,
                similarity_threshold=self.config.cache_similarity_threshold,
                ttl_seconds=self.config.cache_ttl_seconds,
            )
            self.logger.info("LLM 语义缓存已启用")
        except Exception as e:
            self.logger.warning(f"LLM 语义缓存初始化失败: {e}")
            self._cache = None

    def convert_messages_to_dicts(self, messages: List[BaseMessage]) -> List[Dict]:
        return MessageAdapter.to_openai_messages(messages)

    def _normalize_messages(
        self, messages: List[Union[Dict, BaseMessage]]
    ) -> list[dict[str, Any]]:
        return MessageAdapter.to_openai_messages(messages)

    def _should_use_cache(self, stream: bool, return_raw: bool, kwargs: dict) -> bool:
        """判断是否应该使用缓存。"""
        if not self._cache:
            return False
        if stream or return_raw:
            return False
        if kwargs.get("tools"):
            return False
        return True

    def _build_cache_key(
        self,
        model: str,
        messages: list[dict[str, Any]],
        kwargs: dict[str, Any],
    ) -> str:
        """Build stable cache key from model, messages and behavior parameters."""
        relevant_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key
            in {
                "temperature",
                "top_p",
                "max_tokens",
                "presence_penalty",
                "frequency_penalty",
                "extra_body",
                "tool_choice",
            }
        }
        payload = {
            "provider": self.config.provider,
            "model": model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": messages,
            "kwargs": relevant_kwargs,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _serialize_response(self, response) -> dict:
        """将 LLM 响应序列化为 dict。"""
        if hasattr(response, "model_dump"):
            return response.model_dump()
        if hasattr(response, "dict"):
            return response.dict()
        if isinstance(response, dict):
            return response
        return {"content": str(response)}

    def _store_cache_sync(self, cache_key: str, response_data: dict) -> None:
        if not self._cache:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._cache.aset(cache_key, response_data))
            return

        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(
                lambda: asyncio.run(self._cache.aset(cache_key, response_data))
            ).result()

    def _get_cache_sync(self, cache_key: str) -> Optional[dict]:
        if not self._cache:
            return None
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._cache.aget(cache_key))

        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(
                lambda: asyncio.run(self._cache.aget(cache_key))
            ).result()

    async def _store_cache_async(self, cache_key: str, response_data: dict) -> None:
        if self._cache:
            await self._cache.aset(cache_key, response_data)

    def _prepare_call(
        self,
        messages: List[Union[Dict, BaseMessage]],
        kwargs: dict[str, Any],
    ) -> tuple[str, list[dict[str, Any]], str, dict[str, Any]]:
        call_kwargs = dict(kwargs)
        model = call_kwargs.pop("model", self.config.model)
        messages_to_send = self._normalize_messages(messages)
        cache_key = self._build_cache_key(model, messages_to_send, call_kwargs)
        return model, messages_to_send, cache_key, call_kwargs

    def _get_cached_text_sync(self, cache_key: str) -> str | None:
        cached = self._get_cache_sync(cache_key)
        if not cached:
            return None
        self.logger.info("LLM 缓存命中")
        self._last_cache_hit = True
        return self._read_cached_response(cached)

    async def _get_cached_text_async(self, cache_key: str) -> str | None:
        if not self._cache:
            return None
        cached = await self._cache.aget(cache_key)
        if not cached:
            return None
        self.logger.info("LLM 缓存命中")
        self._last_cache_hit = True
        return self._read_cached_response(cached)

    def _read_cached_response(self, response_data: dict):
        if not response_data.get("choices") or not response_data["choices"][0].get(
            "message", {}
        ).get("content"):
            raise ValueError("缓存的 LLM 响应格式异常")
        return response_data["choices"][0]["message"]["content"]

    def chat(
        self,
        messages: List[Union[Dict, BaseMessage]],
        stream: bool = False,
        return_raw: bool = False,
        **kwargs,
    ):
        self._last_cache_hit = False
        model, messages_to_send, cache_key, call_kwargs = self._prepare_call(
            messages, kwargs
        )

        if self._should_use_cache(stream, return_raw, call_kwargs):
            try:
                cached_text = self._get_cached_text_sync(cache_key)
                if cached_text is not None:
                    return cached_text
            except Exception as e:
                self.logger.warning(f"LLM 缓存查询异常: {e}")

        response = self.client.chat(
            model=model,
            messages=messages_to_send,
            stream=stream,
            **call_kwargs,
        )

        if return_raw or stream:
            return response

        result = extract_text_response(response)
        if self._should_use_cache(stream, return_raw, call_kwargs):
            try:
                self._store_cache_sync(cache_key, self._serialize_response(response))
            except Exception as e:
                self.logger.warning(f"LLM 缓存存储异常: {e}")

        if hasattr(langfuse_context, "update_current_span"):
            langfuse_context.update_current_span(
                metadata={"cache_hit": self._last_cache_hit}
            )

        return result

    async def achat(
        self,
        messages: List[Union[Dict, BaseMessage]],
        stream: bool = False,
        return_raw: bool = False,
        **kwargs,
    ):
        """与 LLM 进行异步对话。"""
        self._last_cache_hit = False
        model, messages_to_send, cache_key, call_kwargs = self._prepare_call(
            messages, kwargs
        )

        if self._should_use_cache(stream, return_raw, call_kwargs):
            try:
                cached_text = await self._get_cached_text_async(cache_key)
                if cached_text is not None:
                    return cached_text
            except Exception as e:
                self.logger.warning(f"LLM 缓存查询异常: {e}")

        response = await self.client.achat(
            model=model,
            messages=messages_to_send,
            stream=stream,
            return_raw=True,
            **call_kwargs,
        )

        if return_raw or stream:
            return response

        result = extract_text_response(response)
        if self._should_use_cache(stream, return_raw, call_kwargs):
            try:
                await self._store_cache_async(
                    cache_key, self._serialize_response(response)
                )
            except Exception as e:
                self.logger.warning(f"LLM 缓存存储异常: {e}")

        if hasattr(langfuse_context, "update_current_span"):
            langfuse_context.update_current_span(
                metadata={"cache_hit": self._last_cache_hit}
            )

        return result
