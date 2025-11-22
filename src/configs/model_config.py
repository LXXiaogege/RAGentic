# -*- coding: utf-8 -*-
"""
@Time ： 2025/11/19 17:38
@Auth ： 吕鑫
@File ：model_config.py
@IDE ：PyCharm
"""
from pydantic import BaseModel, Field
import os


class LLMConfig(BaseModel):
    """LLM 相关配置"""
    base_url: str = Field("", description="LLM API Base URL,默认从.env中加载")
    api_key: str = Field("", description="LLM API Key,默认从.env中加载")
    provider: str = Field("openai", description="LLM 提供商")
    model: str = Field("Qwen/Qwen3-32B", description="使用的模型名称")


class EmbeddingConfig(BaseModel):
    """Embedding 相关配置"""
    base_url: str = Field("", description="Embedding API Base URL,默认从.env中加载")
    api_key: str = Field("", description="Embedding API Key,默认从.env中加载")
    model: str = Field("text-embedding-v3", description="Embedding 模型名称")
    cache_path: str = Field(
        "../data/embeddings/embedding_cache.json",
        description="Embedding 本地缓存路径"
    )
