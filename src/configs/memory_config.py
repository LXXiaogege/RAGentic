# -*- coding: utf-8 -*-
"""
@Time ： 2025/11/28 10:41
@Auth ： 吕鑫
@File ：memory_config.py
@IDE ：PyCharm
"""
from pydantic import BaseModel, Field
from mem0.configs.base import LlmConfig, VectorStoreConfig, RerankerConfig, GraphStoreConfig, EmbedderConfig


class Mem0Config(BaseModel):
    """
    Mem0 配置
    """
    llm: LlmConfig = Field(default_factory=LlmConfig, description="大语言模型配置")
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig, description="向量数据库配置")
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig, description="嵌入模型配置")
    reranker: RerankerConfig = Field(default_factory=RerankerConfig, description="reranker 配置")
    graph_store: GraphStoreConfig = Field(default_factory=GraphStoreConfig, description="图数据库配置")

    enable_rerank: bool = Field(False, description="是否使用 reranker")
    enable_graph: bool = Field(False, description="是否使用图数据库")

    history_db_path: str = Field("/Users/lvxin/PycharmProjects/RAGentic/data/memory/history.db",
                                 description="本地 SQLite 数据库备份 Mem0 记忆路径")
    version: str = Field("v0.1.0", description="Mem0 版本")
