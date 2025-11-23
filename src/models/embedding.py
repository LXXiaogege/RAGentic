# -*- coding: utf-8 -*-
"""
@Time ： 2025/4/12 14:17
@Auth ： 吕鑫
@File ：embedding.py
@IDE ：PyCharm
"""
import json
import os
from typing import Union, List, Dict
# from openai import OpenAI
from langfuse.openai import OpenAI,AsyncOpenAI
from src.utils.utils import get_text_hash
from src.configs.logger_config import setup_logger
from src.configs.model_config import EmbeddingConfig
from langfuse import observe

logger = setup_logger(__name__)


class TextEmbedding:
    def __init__(self, config: EmbeddingConfig):
        self.logger = logger
        self.config = config
        self.logger.info("初始化文本嵌入模型...")
        self.client = AsyncOpenAI(api_key=self.config.api_key, base_url=self.config.base_url)
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
        
        Args:
            text: 单个文本或文本列表
            
        Returns:
            List[List[float]]: 嵌入向量列表
        """
        if isinstance(text, str):
            text = [text]

        self.logger.info(f"开始生成 {len(text)} 个文本的嵌入向量...")
        results = []

        for t in text:
            text_hash = get_text_hash(t)

            # 检查缓存
            if text_hash in self.cache:
                self.logger.debug(f"使用缓存的嵌入向量: {text_hash[:8]}...")
                results.append(self.cache[text_hash])
                continue

            try:
                self.logger.debug(f"生成新的嵌入向量: {text_hash[:8]}...")
                response = self.client.embeddings.create(
                    model=self.config.model,
                    input=t
                )
                embedding = response.data[0].embedding

                # 更新缓存
                self.cache[text_hash] = embedding
                results.append(embedding)
                self.logger.debug(f"成功生成嵌入向量: {text_hash[:8]}")

            except Exception as e:
                self.logger.exception(f"生成嵌入向量失败: {str(e)}")
                raise

        # 定期保存缓存
        if len(self.cache) % 100 == 0:  # 每100条记录保存一次
            self._save_cache()

        self.logger.info(f"成功生成所有文本的嵌入向量")
        return results
