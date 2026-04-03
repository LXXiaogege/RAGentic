# -*- coding: utf-8 -*-
"""
@Time：2025/4/12 14:17
@Auth：吕鑫
@File：llm.py
@IDE：PyCharm
"""

from typing import ClassVar, Dict, List, Optional, Type, Union, cast
import asyncio
import json
from src.configs.model_config import LLMConfig
from openai.types.chat import ChatCompletionMessageParam
from langfuse.openai import OpenAI, AsyncOpenAI
from src.configs.logger_config import setup_logger
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    FunctionMessage,
    ToolMessage,
)
from langfuse.decorators import observe, langfuse_context

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
        raise NotImplementedError  # 异步接口


class OpenAILLM(BaseLLM):
    def __init__(self, api_key: str, base_url: str):
        self.logger = setup_logger(f"{__name__}.OpenAILLM")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.aclient = AsyncOpenAI(
            api_key=api_key, base_url=base_url
        )  # <--- 初始化异步客户端

    @observe(name="LLMWrapper.chat", as_type="generation")
    def chat(self, model: str, messages: List[dict], stream: bool = False, **kwargs):
        """
        与 LLM 进行对话

        Args:
            model: 模型名称
            messages: 消息列表
            stream: 是否使用流式输出
            **kwargs: 其他参数
        """
        self.logger.info(f"开始与模型 {model} 对话...")
        self.logger.debug(f"消息数量: {len(messages)}")
        self.logger.debug(f"流式输出: {stream}")

        typed_messages = cast(List[ChatCompletionMessageParam], messages)
        try:
            response = self.client.chat.completions.create(
                model=model, messages=typed_messages, stream=stream, **kwargs
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
        """
        与 LLM 进行异步对话
        """
        typed_messages = cast(List[ChatCompletionMessageParam], messages)
        try:
            response = await self.aclient.chat.completions.create(
                model=model, messages=typed_messages, stream=stream, **kwargs
            )
            self.logger.info("成功获取模型异步响应")
            return response
        except Exception as e:
            self.logger.exception(f"调用模型时发生错误: {str(e)}")
            raise


class LLMWrapper:
    TYPE_TO_ROLE: ClassVar[Dict[Type[BaseMessage], str]] = {
        HumanMessage: "user",
        AIMessage: "assistant",
        SystemMessage: "system",
        FunctionMessage: "function",
        ToolMessage: "tool",
    }

    def __init__(self, config: LLMConfig):
        self.logger = setup_logger(f"{__name__}.LLMWrapper")
        self.config = config
        if self.config.provider == "litellm":
            from src.models.llm_litellm import LiteLLMWrapper

            self.client = LiteLLMWrapper(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                provider=self.config.model.split("/")[0] if "/" in self.config.model else "openai",
            )
        elif self.config.provider in ("openai", "minimax"):
            self.client = OpenAILLM(self.config.api_key, self.config.base_url)
        else:
            raise ValueError(f"暂不支持的模型提供商：{self.config.provider}")

        # 初始化 LLM 语义缓存
        self._cache = None
        self._last_cache_hit = False
        if self.config.use_response_cache:
            self._init_cache()

    def _init_cache(self):
        """初始化 LLM 响应缓存"""
        try:
            from src.models.llm_response_cache import LLMResponseCache
            from src.configs.config import AppConfig

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

    def _extract_last_user_message(self, messages: List[Union[Dict, BaseMessage]]) -> Optional[str]:
        """从 messages 中提取最后一个 user message 的 content"""
        if not messages:
            return None
        # messages 可能是 Dict 列表或 BaseMessage 列表
        user_messages = []
        for msg in reversed(messages):
            if isinstance(msg, BaseMessage):
                if isinstance(msg, HumanMessage):
                    user_messages.append(msg.content)
            elif isinstance(msg, dict):
                if msg.get("role") == "user":
                    user_messages.append(msg.get("content", ""))
        return user_messages[0] if user_messages else None

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

    def _should_use_cache(self, stream: bool, return_raw: bool, kwargs: dict) -> bool:
        """判断是否应该使用缓存"""
        if not self._cache:
            return False
        if stream or return_raw:
            return False
        if kwargs.get("tools"):
            return False
        return True

    def _serialize_response(self, response) -> dict:
        """将 LLM 响应序列化为 dict"""
        if hasattr(response, "model_dump"):
            return response.model_dump()
        elif hasattr(response, "dict"):
            return response.dict()
        elif isinstance(response, dict):
            return response
        else:
            return {"content": str(response)}

    def _deserialize_response(self, data: dict):
        """将 dict 反序列化为 LLM 响应对象"""
        # 简化处理：直接返回 dict，由调用方处理
        # 如需完整对象重建，可根据需求扩展
        return data

    def chat(
        self,
        messages: List[Union[Dict, BaseMessage]],
        stream: bool = False,
        return_raw: bool = False,
        **kwargs,
    ):
        # 缓存查询：仅对非 stream、非 raw、带 tools 的请求缓存
        self._last_cache_hit = False
        query_text = self._extract_last_user_message(messages)
        if self._should_use_cache(stream, return_raw, kwargs) and query_text:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 在新线程中运行异步缓存查找，避免阻塞事件循环
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        cached = list(executor.map(
                            lambda: asyncio.run(self._cache.aget(query_text)),
                            [None]
                        ))[0]
                else:
                    cached = asyncio.run(self._cache.aget(query_text))

                if cached:
                    self.logger.info(f"LLM 缓存命中: {query_text[:50]}...")
                    self._last_cache_hit = True
                    response_data = cached
                    if return_raw:
                        return self._deserialize_response(response_data)
                    if not response_data.get("choices") or not response_data["choices"][0].get("message", {}).get("content"):
                        raise ValueError("缓存的 LLM 响应格式异常")
                    return response_data["choices"][0]["message"]["content"]
            except Exception as e:
                self.logger.warning(f"LLM 缓存查询异常: {e}")

        # 如果是 LangChain 消息，先转换为 OpenAI 所需格式
        messages_to_send = self.convert_messages_to_dicts(messages) if messages and isinstance(messages[0], BaseMessage) else messages
        response = self.client.chat(
            model=self.config.model, messages=messages_to_send, stream=stream, **kwargs
        )

        if return_raw or stream:
            # 异步存储到缓存（不阻塞返回）
            if self._should_use_cache(stream, return_raw, kwargs) and query_text:
                try:
                    response_data = self._serialize_response(response)
                    asyncio.create_task(self._cache.aset(query_text, response_data))
                except Exception as e:
                    self.logger.warning(f"LLM 缓存存储异常: {e}")
            return response

        if not response.choices or response.choices[0].message.content is None:
            raise ValueError("LLM 返回了空的 choices 或 content，请检查模型服务")

        result = response.choices[0].message.content

        # 异步存储到缓存
        if self._should_use_cache(stream, return_raw, kwargs) and query_text:
            try:
                response_data = self._serialize_response(response)
                asyncio.create_task(self._cache.aset(query_text, response_data))
            except Exception as e:
                self.logger.warning(f"LLM 缓存存储异常: {e}")

        # 更新 span metadata
        if hasattr(langfuse_context, "update_current_span"):
            langfuse_context.update_current_span(metadata={"cache_hit": self._last_cache_hit})

        return result

    async def achat(
        self,
        messages: List[Union[Dict, BaseMessage]],
        stream: bool = False,
        return_raw: bool = False,
        **kwargs,
    ):
        """
        与 LLM 进行异步对话
        """
        self._last_cache_hit = False
        query_text = self._extract_last_user_message(messages)

        # 缓存查询
        if self._should_use_cache(stream, return_raw, kwargs) and query_text:
            try:
                cached = await self._cache.aget(query_text)
                if cached:
                    self.logger.info(f"LLM 缓存命中: {query_text[:50]}...")
                    self._last_cache_hit = True
                    response_data = cached
                    if return_raw:
                        return self._deserialize_response(response_data)
                    if not response_data.get("choices") or not response_data["choices"][0].get("message", {}).get("content"):
                        raise ValueError("缓存的 LLM 响应格式异常")
                    return response_data["choices"][0]["message"]["content"]
            except Exception as e:
                self.logger.warning(f"LLM 缓存查询异常: {e}")

        messages_to_send = self.convert_messages_to_dicts(messages) if messages and isinstance(messages[0], BaseMessage) else messages
        response = await self.client.achat(
            model=self.config.model, messages=messages_to_send, stream=stream, **kwargs
        )

        if return_raw or stream:
            if self._should_use_cache(stream, return_raw, kwargs) and query_text:
                try:
                    response_data = self._serialize_response(response)
                    asyncio.create_task(self._cache.aset(query_text, response_data))
                except Exception as e:
                    self.logger.warning(f"LLM 缓存存储异常: {e}")
            return response

        if not response.choices or response.choices[0].message.content is None:
            raise ValueError("LLM 返回了空的 choices 或 content，请检查模型服务")

        result = response.choices[0].message.content

        # 异步存储到缓存
        if self._should_use_cache(stream, return_raw, kwargs) and query_text:
            try:
                response_data = self._serialize_response(response)
                asyncio.create_task(self._cache.aset(query_text, response_data))
            except Exception as e:
                self.logger.warning(f"LLM 缓存存储异常: {e}")

        # 更新 span metadata
        if hasattr(langfuse_context, "update_current_span"):
            langfuse_context.update_current_span(metadata={"cache_hit": self._last_cache_hit})

        return result