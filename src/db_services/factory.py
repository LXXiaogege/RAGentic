# -*- coding: utf-8 -*-
"""
@Time：2026/3/28
@Auth：RAGentic
@File：factory.py
@IDE：PyCharm

向量库工厂函数
根据配置创建对应的 VectorStore 实现
"""

from typing import TYPE_CHECKING, Any

from src.configs.logger_config import setup_logger

if TYPE_CHECKING:
    from src.configs.config import AppConfig
    from src.models.embedding import TextEmbedding

from src.db_services.base import VectorStore
from src.db_services.milvus.store import MilvusVectorStore

logger = setup_logger(__name__)


def create_vector_store(
    config: "AppConfig",
    embeddings: "TextEmbedding",
    text_splitter: Any,
) -> VectorStore:
    """
    根据配置创建对应的 VectorStore 实现

    Args:
        config: 应用配置
        embeddings: 嵌入模型
        text_splitter: 文本分割器

    Returns:
        VectorStore 实例

    Raises:
        ValueError: 不支持的 db_type
    """
    db_type = getattr(config.milvus, "db_type", "milvus").lower()

    if db_type == "milvus":
        logger.info("创建 Milvus 向量库实例...")
        return MilvusVectorStore(
            milvus_config=config.milvus,
            search_config=config.retrieve,
            rerank_config=config.reranker,
            bm25_config=config.bm25,
            embeddings=embeddings,
            text_splitter=text_splitter,
        )
    else:
        raise ValueError(f"不支持的 db_type: {db_type}，支持的类型: milvus")


__all__ = ["create_vector_store", "VectorStore"]
