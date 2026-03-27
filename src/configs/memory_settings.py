# -*- coding: utf-8 -*-
"""
@Time：2026/3/27
@Auth：AI Assistant
@File：memory_settings.py
@IDE：PyCharm
"""

from pydantic import BaseModel, Field


class MemorySettings(BaseModel):
    """混合记忆服务统一配置"""

    service_type: str = Field(
        default="hybrid",
        description="记忆服务类型: hybrid / stm_only / ltm_only",
    )

    stm_window_size: int = Field(
        default=10,
        ge=1,
        description="短期记忆窗口大小",
    )

    ltm_persist_threshold: int = Field(
        default=3,
        ge=1,
        description="短期记忆刷写到长期记忆的阈值（轮数）",
    )

    enable_stm: bool = Field(
        default=True,
        description="是否启用短期记忆",
    )

    enable_ltm: bool = Field(
        default=True,
        description="是否启用长期记忆",
    )

    stm_checkpointer_enabled: bool = Field(
        default=False,
        description="是否启用 LangGraph Checkpoint 保存短期记忆",
    )

    ltm_search_limit: int = Field(
        default=5,
        ge=1,
        description="长期记忆搜索结果数量限制",
    )

    ltm_search_threshold: float = Field(
        default=None,
        description="长期记忆搜索相似度阈值（None 表示不限制）",
    )

    class Config:
        frozen = False
