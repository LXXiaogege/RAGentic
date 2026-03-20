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

    templates_dir: str = Field(default="templates", description="Jinja2 模板目录")

    kb_system_prompt_name: str = Field(
        "kb_system_prompt", description="知识库系统 prompt 名称"
    )
    kb_system_prompt: str = Field(
        """{prefix}你是一个资深知识问答助手，回答时需参考给定的上下文资料。请用中文作答，若无法从资料中获取信息，请如实说明。{context_hint}""",
        description="知识库系统 prompt 模板",
    )

    system_prompt_name: str = Field("system_prompt", description="系统 prompt 名称")
    system_prompt: str = Field(
        """你是一个知识问答小助手，请帮助回答用户想知道的问题。""",
        description="系统 prompt 模板",
    )

    rewrite_prompt: str = Field(
        """你是一个AI助手，任务是重新制定用户查询以改进RAG系统中的检索。给定原始查询，将其重写为更具体、更详细，并且更有可能检索相关信息。
        原始查询：{original_query}
        重写的查询：""",
        description="查询重写提示词",
    )

    step_back_prompt: str = Field(
        """你是一个 AI 助手，任务是生成更广泛、更通用的查询，以改进 RAG 系统中的上下文检索。
        给定原始查询，生成一个更通用的step-back查询，该查询可以帮助检索相关的背景信息。
        原始查询：{original_query}
        step-back查询：""",
        description="Step-back 查询提示词",
    )

    sub_query_prompt: str = Field(
        """你是一名 AI 助手，任务是将复杂的查询分解为 RAG 系统的更简单的子查询。
        给定原始查询，将其分解为 2-4 个更简单的子查询，当一起回答时，将提供对原始查询的全面响应。
        原始查询： {original_query}
        示例：气候变化对环境有哪些影响？
        子查询：
        1.气候变化对生物多样性有哪些影响？
        2.气候变化如何影响海洋？
        3.气候变化对农业有哪些影响？
        4.气候变化对人类健康有哪些影响？""",
        description="子查询分解提示词",
    )

    hyde_prompt: str = Field(
        """你是一个AI助手，任务是生成假设的答案，用于信息检索。
        根据原始查询，生成一个假设的答案，该答案将用于信息检索，但不用于最终回答。
        原始查询： {original_query}
        假设的答案：""",
        description="HyDE 提示词",
    )
    kb_system_prompt: str = Field(
        """{prefix}你是一个资深知识问答助手，回答时需参考给定的上下文资料。请用中文作答，若无法从资料中获取信息，请如实说明。{context_hint}""",
        description="知识库系统 prompt 模板",
    )

    system_prompt_name: str = Field("system_prompt", description="系统 prompt 名称")
    system_prompt: str = Field(
        """你是一个知识问答小助手，请帮助回答用户想知道的问题。""",
        description="系统 prompt 模板",
    )
