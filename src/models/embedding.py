# -*- coding: utf-8 -*-
"""
@Time ： 2025/4/12 14:17
@Auth ： 吕鑫
@File ：embedding.py
@IDE ：PyCharm
"""
from typing import Union, List, Dict, Optional, cast, Any
from langfuse.openai import AsyncOpenAI
from src.configs.logger_config import setup_logger
from src.configs.model_config import EmbeddingConfig
from src.configs.database_config import RedisConfig
from src.configs.config import AppConfig
from langfuse import observe
from redisvl.extensions.cache.embeddings import EmbeddingsCache

logger = setup_logger(__name__)


class TextEmbedding:
    def __init__(self, config: AppConfig):
        self.logger = logger
        self.embedding_config: EmbeddingConfig = config.embedding
        self.redis_config: RedisConfig = config.redis
        self.use_cache = self.embedding_config.use_cache
        self.aclient = AsyncOpenAI(api_key=self.embedding_config.api_key, base_url=self.embedding_config.base_url)

        self.cache: Optional[EmbeddingsCache] = None
        if self.use_cache:
            self.cache: EmbeddingsCache = EmbeddingsCache(redis_url=self.redis_config.redis_url, ttl=None)
            self.logger.info("RedisVL 嵌入缓存已启用并初始化完成。")

    async def _aget_embeddings_from_cache(self, text: List[str]) -> List[Optional[Dict[str, Any]]]:
        if not self.use_cache or self.cache is None:
            return [None] * len(text)
        return await self.cache.amget(text, model_name=self.embedding_config.model)

    async def _acall_api_and_store(
            self,
            texts_to_embed: List[str],
            indices_to_embed: List[int],
            final_results: List[Optional[List[float]]]
    ):
        """
        对未命中缓存的文本进行批量 API 调用，更新最终结果列表，并将新嵌入存入缓存。

        Args:
            texts_to_embed: 实际需要调用 API 的文本列表。
            indices_to_embed: 它们在原始输入列表中的索引。
            final_results: 引用原始结果列表，用于更新。
        """
        response = await self.aclient.embeddings.create(
            model=self.embedding_config.model,
            input=texts_to_embed
        )

        items_to_store = []

        for i, d in enumerate(response.data):
            original_index = indices_to_embed[i]
            embedding = d.embedding
            # 1. 更新最终结果列表 (通过索引映射回正确位置)
            final_results[original_index] = embedding

            # 2. 准备批量存储到缓存的项
            if self.use_cache:
                items_to_store.append({
                    "text": texts_to_embed[i],
                    "model_name": self.embedding_config.model,
                    "embedding": embedding
                })
        # 异步批量存储 (amset)
        if items_to_store and self.cache:
            self.logger.debug(f"异步批量存储 {len(items_to_store)} 条新嵌入到缓存。")
            await self.cache.amset(items=items_to_store)

    @observe(name="TextEmbedding.aget_embedding", as_type="embedding")
    async def aget_embedding(self, text: Union[str, List[str]]) -> List[List[float]]:
        """
        异步获取文本的嵌入向量，智能使用 RedisVL 缓存进行批量查找和存储。
        """
        if isinstance(text, str):
            text = [text]
        # 1. 批量查找缓存
        cache_entries = await self._aget_embeddings_from_cache(text) if self.use_cache else [None] * len(text)
        # 2. 区分缓存命中和未命中
        final_results: List[Optional[List[float]]] = [None] * len(text)
        texts_to_embed: List[str] = []
        indices_to_embed: List[int] = []

        for i, entry in enumerate(cache_entries):
            if entry is not None and entry.get("embedding"):
                # 缓存命中
                final_results[i] = entry["embedding"]
            else:
                # 缓存未命中
                texts_to_embed.append(text[i])
                indices_to_embed.append(i)

        # 3. 如果有未命中项，则进行 API 调用和缓存存储
        if texts_to_embed:
            await self._acall_api_and_store(
                texts_to_embed,
                indices_to_embed,
                final_results
            )
        else:
            self.logger.info("所有文本均命中缓存，无需调用 API。")
        return cast(List[List[float]], final_results)
