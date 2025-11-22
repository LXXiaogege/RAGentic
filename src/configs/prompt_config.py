# -*- coding: utf-8 -*-
"""
@Time ： 2025/11/19 18:37
@Auth ： 吕鑫
@File ：prompt_config.py
@IDE ：PyCharm
"""
from pydantic import BaseModel, Field


class PromptConfig(BaseModel):
    """系统和知识库 Prompt 配置"""

    kb_system_prompt_name: str = Field(
        "kb_system_prompt",
        description="知识库系统 prompt 名称"
    )
    kb_system_prompt: str = Field(
        """{prefix}你是一个资深知识问答助手，回答时需参考给定的上下文资料。请用中文作答，若无法从资料中获取信息，请如实说明。{context_hint}""",
        description="知识库系统 prompt 模板"
    )

    system_prompt_name: str = Field(
        "system_prompt",
        description="系统 prompt 名称"
    )
    system_prompt: str = Field(
        """你是一个知识问答小助手，请帮助回答用户想知道的问题。""",
        description="系统 prompt 模板"
    )
