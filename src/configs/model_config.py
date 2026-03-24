# -*- coding: utf-8 -*-
"""
@Time：2025/11/19 17:38
@Auth：吕鑫
@File：model_config.py
@IDE：PyCharm
"""

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM 相关配置"""

    base_url: str = Field("", description="LLM API Base URL，默认从.env 中加载")
    api_key: str = Field("", description="LLM API Key，默认从.env 中加载")
    provider: str = Field("openai", description="LLM 提供商")
    model: str = Field("Qwen/Qwen3-32B", description="使用的模型名称")
    temperature: float = Field(0.0, ge=0, le=1, description="生成结果随机性")
    max_tokens: int = Field(2000, gt=0, description="最大 tokens 数")


class EmbeddingConfig(BaseModel):
    """Embedding 相关配置"""

    base_url: str = Field("", description="Embedding API Base URL，默认从.env 中加载")
    api_key: str = Field("", description="Embedding API Key，默认从.env 中加载")
    model: str = Field("text-embedding-v3", description="Embedding 模型名称")
    use_cache: bool = Field(True, description="是否使用缓存")


class RerankConfig(BaseModel):
    rerank_model_path: str = Field(
        default="data/models/bge-reranker-base",
        description="Rerank 模型路径，可从环境变量 RERANK__MODEL_PATH 加载",
    )
    rerank_device: str = Field("cpu", description="cpu / cuda")
    max_length: int = Field(512, gt=0)


class BM25Config(BaseModel):
    bm25_model_dir: str = Field(
        default="data/knowledge_db/bm25_model",
        description="BM25 模型目录，可从环境变量 BM25__MODEL_DIR 加载",
    )
    bm25_autofit: bool = True
    bm25_language: str = "zh"
    bm25_drop_ratio: float = Field(0.2, ge=0, le=1)
