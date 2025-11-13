# -*- coding: utf-8 -*-
"""
@Time ： 2025/7/1 22:26
@Auth ： 吕鑫
@File ：__main__.py
@IDE ：PyCharm
"""
import httpx
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, InMemoryPushNotificationConfigStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from src.cores.qa_agent_executor import QAAgentExecutor

if __name__ == '__main__':
    # 定义QA Agent的基础技能
    skill = AgentSkill(
        id='qa_basic',
        name='知识问答',
        description='基于知识库和大模型的智能问答，支持上下文记忆、工具调用和多轮对话。',
        tags=['问答', '知识库', 'RAG', '多轮对话', '工具调用'],
        examples=['什么是RAG？', '请用中文回答以下问题：……', '帮我查找某个知识点'],
    )

    # 可扩展技能（如工具增强、流式输出等）
    extended_skill = AgentSkill(
        id='qa_advanced',
        name='高级知识问答',
        description='支持流式输出、外部工具调用、上下文追踪等增强功能。',
        tags=['流式', '工具', '增强', '上下文'],
        examples=['请流式输出答案', '调用工具获取最新天气'],
    )

    # 公共Agent卡片
    public_agent_card = AgentCard(
        name='智能知识问答助手',
        description='基于LangGraph和RAG的智能知识问答服务，支持中文、知识库检索、工具调用和多轮对话。',
        url='http://localhost:9999/',
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
        supports_authenticated_extended_card=True,
    )

    httpx_client = httpx.AsyncClient()
    # 扩展Agent卡片（认证用户可见）
    specific_extended_agent_card = public_agent_card.model_copy(
        update={
            'name': '智能知识问答助手（增强版）',
            'description': '面向认证用户，支持流式输出、工具增强等高级功能。',
            'version': '1.1.0',
            'skills': [skill, extended_skill],
        }
    )


    request_handler = DefaultRequestHandler(
        agent_executor=QAAgentExecutor(),
        task_store=InMemoryTaskStore(),
        push_notification_config_store=InMemoryPushNotificationConfigStore()
    )

    server = A2AStarletteApplication(
        agent_card=public_agent_card,
        http_handler=request_handler,
        # extended_agent_card=specific_extended_agent_card,
    )

    uvicorn.run(server.build(), host='0.0.0.0', port=9999)
