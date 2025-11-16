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
from langfuse.openai import OpenAI
from src.utils.utils import get_text_hash
from src.config.logger_config import setup_logger
from langfuse import observe

logger = setup_logger(__name__)


class TextEmbedding:
    def __init__(self, api_key: str, base_url: str, model: str = "text2vec",
                 cache_path: str = "../../data/embeddings/embedding_cache.json"):
        self.logger = logger
        self.logger.info("初始化文本嵌入模型...")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.cache_path = cache_path
        self.logger.debug(f"使用模型: {model}")
        self.logger.debug(f"缓存路径: {cache_path}")

        self.cache: Dict[str, List[float]] = self._load_cache()
        self.logger.info("文本嵌入模型初始化完成")

    def _load_cache(self) -> Dict[str, List[float]]:
        """加载嵌入向量缓存"""
        self.logger.debug("加载嵌入向量缓存...")
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
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
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
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
                # 生成新的嵌入向量
                self.logger.debug(f"生成新的嵌入向量: {text_hash[:8]}...")
                response = self.client.embeddings.create(
                    model=self.model,
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


# 使用示例
if __name__ == "__main__":
    # 创建嵌入实例
    from src.config.config import QAPipelineConfig

    config = QAPipelineConfig()
    embedder = TextEmbedding(api_key=config.embedding_api_key, base_url=config.embedding_base_url,
                             model=config.embedding_model)

    # 测试单个文本
    text = "测试嵌入模型调用"
    embedding = embedder.get_embedding(text)[0]
    print(f"文本 '{text}' 的嵌入向量维度: {len(embedding)}")

    # 测试多个文本
    texts = ["第一个测试文本", "第二个测试文本"]
    embeddings = embedder.get_embedding(texts)
    print(f"处理了 {len(embeddings)} 个文本")
    print(f"每个文本的嵌入向量维度: {len(embeddings[0])}")
