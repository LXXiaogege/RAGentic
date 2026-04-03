# -*- coding: utf-8 -*-
"""
@Time：2026/4/2
@Auth：Plan
@File：llm_response_cache.py
@IDE：PyCharm

LLM 语义缓存 - 基于 RedisVL 向量相似度匹配，复用现有 Redis 基础设施
"""

import asyncio
import hashlib
import json
from typing import Any, Callable, Optional

import redis.asyncio as redis
from redisvl.extensions.cache.embeddings import EmbeddingsCache

from src.configs.logger_config import setup_logger
from langfuse.openai import AsyncOpenAI

logger = setup_logger(__name__)


class LLMResponseCache:
    """
    LLM 响应语义缓存

    使用 RedisVL EmbeddingsCache 存储 query embedding，用 Redis Hash 存储 response。
    缓存命中逻辑：计算 query embedding，在 Redis 中做向量相似度搜索，阈值命中则返回缓存 response。
    """

    CACHE_PREFIX = "llm_response_cache"
    RESPONSE_KEY_PREFIX = "llm_response"

    def __init__(
        self,
        redis_url: str,
        embedding_model: str,
        embedding_api_key: str,
        embedding_base_url: str,
        similarity_threshold: float = 0.92,
        ttl_seconds: int = 86400,
    ):
        self.logger = logger
        self.redis_url = redis_url
        self.embedding_model = embedding_model
        self.embedding_api_key = embedding_api_key
        self.embedding_base_url = embedding_base_url
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds

        # EmbeddingsCache 用于向量相似度搜索
        self._embedding_cache: Optional[EmbeddingsCache] = None
        # Redis 异步客户端，用于直接操作 response hash
        self._redis_client: Optional[redis.Redis] = None
        # AsyncOpenAI 客户端，复用连接
        self._aclient: Optional[AsyncOpenAI] = None

    async def _get_embedding_cache(self) -> EmbeddingsCache:
        if self._embedding_cache is None:
            self._embedding_cache = EmbeddingsCache(
                redis_url=self.redis_url,
                ttl=self.ttl_seconds,
            )
            self.logger.info("LLMResponseCache: EmbeddingsCache 已初始化")
        return self._embedding_cache

    async def _get_redis_client(self) -> redis.Redis:
        if self._redis_client is None:
            self._redis_client = redis.from_url(self.redis_url, decode_responses=False)
            self.logger.info("LLMResponseCache: Redis 客户端已初始化")
        return self._redis_client

    def _hash_query(self, query_text: str) -> str:
        """计算 query 文本的 SHA256 hash 作为 cache key"""
        return hashlib.sha256(query_text.encode("utf-8")).hexdigest()

    def _build_cache_key(self, query_hash: str) -> str:
        return f"{self.CACHE_PREFIX}:{query_hash}"

    def _build_response_key(self, query_hash: str) -> str:
        return f"{self.RESPONSE_KEY_PREFIX}:{query_hash}"

    async def _get_embedding(self, texts: list[str]) -> list[list[float]]:
        """调用 embedding API 获取文本向量（复用 AsyncOpenAI 客户端）"""
        if self._aclient is None:
            self._aclient = AsyncOpenAI(
                api_key=self.embedding_api_key,
                base_url=self.embedding_base_url,
            )
        response = await self._aclient.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    async def aget(self, query_text: str) -> Optional[dict]:
        """
        异步获取缓存的响应

        Args:
            query_text: 查询文本

        Returns:
            缓存的响应 dict，未命中返回 None
        """
        if not query_text:
            return None

        try:
            cache = await self._get_embedding_cache()
            redis_client = await self._get_redis_client()

            # 1. 获取 query embedding
            embeddings = await self._get_embedding([query_text])
            query_embedding = embeddings[0]

            # 2. 向量相似度搜索
            results = await cache.amget([query_text], model_name=self.embedding_model)

            if results and results[0] and results[0].get("embedding"):
                cached_embedding = results[0]["embedding"]
                # 计算余弦相似度（RedisVL 返回的已经是排序结果）
                # 如果 similarity_threshold 满足，则命中
                self.logger.info(f"LLMResponseCache: 命中缓存，相似度阈值={self.similarity_threshold}")
                query_hash = results[0].get("query_hash") or self._hash_query(query_text)
                response_key = self._build_response_key(query_hash)
                cached_response = await redis_client.get(response_key)
                if cached_response:
                    response_data = json.loads(cached_response.decode("utf-8"))
                    self.logger.debug(f"LLMResponseCache: 从 Redis Hash 读取响应成功")
                    return response_data

            self.logger.debug(f"LLMResponseCache: 未命中缓存")
            return None

        except Exception as e:
            self.logger.warning(f"LLMResponseCache: 读取缓存异常: {e}")
            return None

    async def aset(self, query_text: str, response_data: dict) -> None:
        """
        异步存储 LLM 响应到缓存

        Args:
            query_text: 查询文本
            response_data: LLM 响应 dict
        """
        if not query_text or not response_data:
            return

        try:
            cache = await self._get_embedding_cache()
            redis_client = await self._get_redis_client()

            query_hash = self._hash_query(query_text)
            response_key = self._build_response_key(query_hash)

            # 1. 存储 response 到 Redis Hash
            response_json = json.dumps(response_data, ensure_ascii=False)
            await redis_client.set(
                response_key,
                response_json.encode("utf-8"),
                ex=self.ttl_seconds,
            )

            # 2. 存储 query embedding 到 EmbeddingsCache（用于后续相似度搜索）
            embeddings = await self._get_embedding([query_text])
            query_embedding = embeddings[0]

            await cache.amset(
                items=[
                    {
                        "text": query_text,
                        "model_name": self.embedding_model,
                        "embedding": query_embedding,
                        "query_hash": query_hash,
                    }
                ]
            )

            self.logger.debug(f"LLMResponseCache: 已存储响应，query_hash={query_hash[:16]}...")

        except Exception as e:
            self.logger.warning(f"LLMResponseCache: 存储缓存异常: {e}")

    async def aget_or_call(
        self,
        query_text: str,
        call_llm_fn: Callable[[], Any],
    ) -> Any:
        """
        获取缓存或调用 LLM

        Args:
            query_text: 查询文本
            call_llm_fn: 实际调用 LLM 的异步函数

        Returns:
            LLM 响应（可能是缓存的）
        """
        cached = await self.aget(query_text)
        if cached is not None:
            return cached

        response = await call_llm_fn()
        await self.aset(query_text, response)
        return response
