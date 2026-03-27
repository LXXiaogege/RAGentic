# -*- coding: utf-8 -*-
"""
@Time ： 2025/11/19 18:37
@Auth ： 吕鑫
@File ：prompt_config.py
@IDE ：PyCharm
"""

from pydantic import BaseModel, Field


class PromptConfig(BaseModel):
    """系统和知识库 Prompt 配置 - 重Agent架构"""

    templates_dir: str = Field(default="templates", description="Jinja2 模板目录")

    kb_system_prompt_name: str = Field(
        "kb_system_prompt", description="知识库系统 prompt 名称"
    )
    kb_system_prompt: str = Field(
        """你是一个智能助手。当你需要查询特定事实、信息或知识时，请主动调用 kb_search 工具。
可用工具：
- kb_search(query, top_k): 搜索知识库获取相关文档
- weather_get_alerts(state): 查询美国各州天气预警
- weather_get_forecast(latitude, longitude): 查询天气预报
- web_crawl(url): 爬取网页内容
- read_skill(name): 读取技能指令

请通过工具调用获取信息后，再回答用户问题。若工具返回的信息不足，请如实说明。{context_hint}""",
        description="知识库Agent系统 prompt",
    )

    system_prompt_name: str = Field("system_prompt", description="系统 prompt 名称")
    system_prompt: str = Field(
        """你是一个智能助手。请仔细理解用户问题，通过思考和工具调用来完成任务。
可用工具：
- kb_search(query, top_k): 搜索知识库获取相关文档
- weather_get_alerts(state): 查询美国各州天气预警
- weather_get_forecast(latitude, longitude): 查询天气预报
- web_crawl(url): 爬取网页内容
- read_skill(name): 读取技能指令

请主动判断是否需要调用工具来回答问题。""",
        description="Agent系统 prompt",
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

    skills_dir: str = Field("skills", description="skills md 文件目录")
