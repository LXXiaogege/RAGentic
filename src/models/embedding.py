# -*- coding: utf-8 -*-
"""
@Time ： 2025/4/12 14:17
@Auth ： 吕鑫
@File ：embedding.py
@IDE ：PyCharm
"""
import json
import os
from typing import Union, List, Dict, Optional, cast
# from openai import OpenAI
from langfuse.openai import OpenAI, AsyncOpenAI
from src.utils.utils import get_text_hash
from src.configs.logger_config import setup_logger
from src.configs.model_config import EmbeddingConfig
from langfuse import observe
import asyncio

logger = setup_logger(__name__)


class TextEmbedding:
    def __init__(self, config: EmbeddingConfig):
        self.logger = logger
        self.config = config
        self.logger.info("初始化文本嵌入模型...")
        self.client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)
        self.aclient = AsyncOpenAI(api_key=self.config.api_key, base_url=self.config.base_url)  # <--- 新增异步客户端
        self.logger.debug(f"使用模型: {self.config.model}")
        self.logger.debug(f"缓存路径: {self.config.cache_path}")

        self.cache: Dict[str, List[float]] = self._load_cache()
        self.logger.info("文本嵌入模型初始化完成")

    def _load_cache(self) -> Dict[str, List[float]]:
        """加载嵌入向量缓存"""
        self.logger.debug("加载嵌入向量缓存...")
        if os.path.exists(self.config.cache_path):
            try:
                with open(self.config.cache_path, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                self.logger.info(f"成功加载缓存，包含 {len(cache)} 条记录")
                return cache
            except Exception as e:
                self.logger.error(f"加载缓存失败: {str(e)}")
                return {}
        else:
            self.logger.info("缓存文件不存在，创建新的缓存")
            return {}

    def _save_cache(self):
        """保存嵌入向量缓存"""
        self.logger.debug("保存嵌入向量缓存...")
        try:
            os.makedirs(os.path.dirname(self.config.cache_path), exist_ok=True)
            with open(self.config.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f)
            self.logger.info(f"成功保存缓存，包含 {len(self.cache)} 条记录")
        except Exception as e:
            self.logger.error(f"保存缓存失败: {str(e)}")

    @observe(name="TextEmbedding.get_embedding", as_type="embedding")
    def get_embedding(self, text: Union[str, List[str]]) -> List[List[float]]:
        """
        获取文本的嵌入向量
        """
        if isinstance(text, str):
            text = [text]
        results_temp: List[Optional[List[float]]] = [None] * len(text)
        texts_to_embed: List[str] = []
        indices_to_embed: List[int] = []
        for i, t in enumerate(text):
            text_hash = get_text_hash(t)
            if text_hash in self.cache:
                results_temp[i] = self.cache[text_hash]
            else:
                texts_to_embed.append(t)
                indices_to_embed.append(i)
        if not texts_to_embed:
            self.logger.info("所有文本均命中缓存，无需调用 API")
            return cast(List[List[float]], results_temp)
        try:
            self.logger.debug(f"同步调用 API 生成 {len(texts_to_embed)} 个新的嵌入向量...")
            response = self.client.embeddings.create(
                model=self.config.model,
                input=texts_to_embed
            )
            new_embeddings_data = response.data
            for i, data in enumerate(new_embeddings_data):
                original_index = indices_to_embed[i]  # 获取原始索引
                text_hash = get_text_hash(texts_to_embed[i])
                embedding = data.embedding
                results_temp[original_index] = embedding
                self.cache[text_hash] = embedding
            self.logger.debug(f"成功生成 {len(texts_to_embed)} 个新的嵌入向量")
        except Exception as e:
            self.logger.exception(f"生成嵌入向量失败: {str(e)}")
            raise
        if len(self.cache) % 100 == 0:
            self._save_cache()
        return cast(List[List[float]], results_temp)

    async def aget_embedding(self, text: Union[str, List[str]]) -> List[List[float]]:
        """
        异步获取文本的嵌入向量
        """
        if isinstance(text, str):
            text = [text]

        self.logger.info(f"开始生成 {len(text)} 个文本的嵌入向量 (异步)...")
        results = []
        texts_to_embed = []
        texts_to_embed_hash = []
        for t in text:
            text_hash = get_text_hash(t)
            if text_hash in self.cache:
                results.append(self.cache[text_hash])
            else:
                texts_to_embed.append(t)
                texts_to_embed_hash.append(text_hash)
                results.append(None)  # 占位符

        if not texts_to_embed:
            return results
        try:
            response = await self.aclient.embeddings.create(
                model=self.config.model,
                input=texts_to_embed  # 批量调用
            )
            new_embeddings = response.data
            for i, data in enumerate(new_embeddings):
                text_hash = texts_to_embed_hash[i]
                embedding = data.embedding
                self.cache[text_hash] = embedding
            final_results = []
            new_embed_idx = 0
            for r in results:
                if r is None:
                    final_results.append(new_embeddings[new_embed_idx].embedding)
                    new_embed_idx += 1
                else:
                    final_results.append(r)  # 缓存结果
        except Exception as e:
            self.logger.exception(f"生成嵌入向量失败: {str(e)}")
            raise
        if len(self.cache) % 100 == 0:  # 每100条记录保存一次
            await asyncio.to_thread(self._save_cache)  # <--- 使用 to_thread 避免在异步上下文中阻塞 I/O
        self.logger.info(f"成功生成所有文本的嵌入向量 (异步)")
        return final_results
