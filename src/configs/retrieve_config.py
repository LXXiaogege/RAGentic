# -*- coding: utf-8 -*-
"""
@Time ： 2025/11/19 17:47
@Auth ： 吕鑫
@File ：retrieve_config.py
@IDE ：PyCharm
"""

from pydantic import BaseModel, Field
from typing import List, Optional


# 文本切分配置
class SplitterConfig(BaseModel):
    chunk_size: int = Field(500, ge=1)
    chunk_overlap: int = Field(50, ge=0, description="切分时的重叠长度")


# 查询改写配置
class RewriteConfig(BaseModel):
    pass


# Milvus 数据库配置
class MilvusConfig(BaseModel):
    milvus_mode: str = Field("local", description="local / remote")
    vector_db_uri: str = Field("/Users/lvxin/PycharmProjects/RAGentic/data/knowledge_db/db/rag.db",
                               description="Milvus URI 或 SQLite 文件路径")
    db_user: str = "root"
    db_password: str = "Milvus"
    db_name: str = "test"

    collection_name: str = "pys"
    vector_dimension: int = Field(1024, gt=0)
    max_text_length: int = Field(10000, gt=0)
    max_metadata_length: int = Field(256, gt=0)

    bm25_model_dir: str = "../data/knowledge_db/bm25_model"
    bm25_autofit: bool = True
    bm25_language: str = "zh"
    bm25_drop_ratio: float = Field(0.2, ge=0, le=1)

    rerank_model_path: str = "/Users/lvxin/datasets/models/bge-reranker-base"
    rerank_device: str = Field("cpu", description="cpu / cuda")


# 检索配置
class SearchConfig(BaseModel):
    search_multiplier: int = Field(2, gt=0, description="用于计算初始召回：top_k * multiplier")

    # 查询改写
    use_rewrite: bool = False
    rewrite_mode: str = Field("rewrite", description="可选: rewrite / step_back / sub_query / hyde（仅知识库查询下有效）")
    # 知识库
    use_kb: bool = False
    use_contextualize_embedding: bool = False
    kb_path: str = Field("../data/knowledge_db/psychology")

    max_context_length: int = Field(300000, ge=1)

    retriever_type: str = Field("dense", description="dense / sparse / hybrid")

    retriever_weights: List[float] = Field(
        default_factory=lambda: [0.6, 0.4],
        description="稠密+稀疏检索融合权重"
    )

    top_k: int = Field(3, ge=1)

    use_sparse: bool = False
    use_reranker: bool = False

    use_memory: bool = False
    memory_window_size: int = Field(5, ge=1)

    no_think: bool = Field(True, description="是否跳过推理过程（R1）")
    use_tool: bool = Field(False, description="是否在工具模式下执行")

    user_id: Optional[str] = Field(None, min_length=1, max_length=100)
    session_id: Optional[str] = Field(None, min_length=1, max_length=100)


# Prompt 构建配置
class MessageBuilderConfig(BaseModel):
    message_builder_model: str = "gpt-3.5-turbo"
    message_max_tokens: int = Field(3500000, ge=1)

    message_system_prompt_template: str = "{prefix}{context_hint}"

    message_context_hint_template: str = (
        "以下是外部知识库资料：\n{context}"
    )

    message_no_think_prefix: str = "/no_think "
