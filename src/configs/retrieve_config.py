# -*- coding: utf-8 -*-
"""
@Time ： 2025/11/19 17:47
@Auth ： 吕鑫
@File ：retrieve_config.py
@IDE ：PyCharm
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


# 文本切分配置
class SplitterConfig(BaseModel):
    chunk_size: int = Field(500, ge=1)
    chunk_overlap: int = Field(50, ge=0, description="切分时的重叠长度")


# 检索配置
class SearchConfig(BaseModel):
    search_multiplier: int = Field(2, gt=0, description="用于计算初始召回：top_k * multiplier")

    # 查询改写
    use_rewrite: bool = False
    rewrite_mode: str = Field("rewrite", description="可选: rewrite / step_back / sub_query / hyde（仅知识库查询下有效）")
    num_hypo: int = Field(3, ge=1, description="HyDE 模式下，生成假设答案数量")
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

    use_think: bool = Field(True, description="是否跳过推理过程（R1）")
    use_tool: bool = Field(False, description="是否在工具模式下执行")

    extra_body: Dict[str, Any] = Field(
        default_factory=lambda: {
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        },
        description="LLM 请求额外参数配置，如启用/关闭 thinking 模式"
    )

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
