# -*- coding: utf-8 -*-
"""
RAGentic - Chainlit UI
纯 Async 架构版本
"""

import asyncio
import os
import uuid
from src.configs.config import AppConfig
from typing import Optional

import chainlit as cl
from chainlit.input_widget import Slider, Switch

from langfuse import Langfuse

from src.configs.logger_config import setup_logger
from src.cores.pipeline_langgraph import LangGraphQAPipeline
from src.utils.security import SecurityManager

logger = setup_logger(__name__)

_NODE_NAMES = {
    "retrieve_knowledge": "检索知识库",
    "call_tools": "调用工具",
    "transform_query": "改写查询",
    "build_context": "构建上下文",
    "parse_query": "解析查询",
    "generate_answer": "生成答案",
    "update_memory": "更新记忆",
}

_SHOW_NODES = {"retrieve_knowledge", "call_tools", "transform_query"}
_STREAM_CHUNK_SIZE = 6
_CONTEXT_PREVIEW_MAX_LEN = 500

# Pipeline 实例缓存，按会话ID缓存，实现管道复用
_pipeline_cache: dict[str, LangGraphQAPipeline] = {}


@cl.on_app_startup
async def on_startup():
    os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY", "")
    os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    os.environ["LANGFUSE_HOST"] = os.getenv(
        "LANGFUSE_HOST", "https://cloud.langfuse.com"
    )
    os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = "production"
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE", "")

    try:
        Langfuse()
        logger.info("Langfuse 客户端初始化成功")
    except Exception as e:
        logger.warning(f"Langfuse 客户端初始化失败：{e}")


@cl.on_app_shutdown
async def on_shutdown():
    logger.info("应用关闭")


@cl.on_chat_start
async def on_chat_start():
    session_id = str(uuid.uuid4())
    cl.user_session.set("session_id", session_id)

    default_settings = {
        "use_memory": False,
        "top_k": 5,
        "use_stream": True,
        "use_sparse": False,
        "use_reranker": False,
        "enable_think": False,
    }
    cl.user_session.set("settings", default_settings)

    settings = cl.ChatSettings(
        [
            Switch(
                id="use_memory",
                label="💬 上下文记忆",
                initial=False,
                description="记住多轮对话内容",
            ),
            Slider(
                id="top_k",
                label="检索数量 (Top-K)",
                initial=5,
                min=1,
                max=20,
                step=1,
                description="知识库检索返回的文档数量",
            ),
            Switch(
                id="use_stream",
                label="⚡ 流式输出",
                initial=True,
                description="逐字流式返回答案",
            ),
            Switch(
                id="use_sparse",
                label="混合检索",
                initial=False,
                description="稠密 + 稀疏 (BM25) 混合检索",
            ),
            Switch(
                id="use_reranker",
                label="🎯 重排序",
                initial=False,
                description="BGE reranker 精排",
            ),
            Switch(
                id="enable_think",
                label="🧠 深度思考",
                initial=False,
                description="启用 R1 推理模式",
            ),
        ]
    )
    await settings.send()

    actions = _make_actions()
    welcome = cl.Message(
        content=(
            "**RAGentic 智能助手**已就绪。\n\n"
            f"> 会话 ID：`{session_id}`\n\n"
            "在左侧边栏调整检索参数，然后开始提问。"
        ),
        author="RAGentic",
        actions=actions,
    )
    await welcome.send()


@cl.on_settings_update
async def on_settings_update(settings: dict):
    current = cl.user_session.get("settings") or {}
    current.update(settings)
    cl.user_session.set("settings", current)
    logger.info(f"[SETTINGS_UPDATE] new settings: {current}")


def _make_actions():
    return [
        cl.Action(
            name="new_chat",
            label="新对话",
            payload={"action": "new"},
            tooltip="开启全新会话（清除对话记录）",
        ),
        cl.Action(
            name="clear_memory",
            label="清空记忆",
            payload={"action": "clear"},
            tooltip="重置跨轮记忆，保留界面对话记录",
        ),
    ]


@cl.action_callback("new_chat")
async def on_new_chat(action: cl.Action):
    new_session_id = str(uuid.uuid4())
    cl.user_session.set("session_id", new_session_id)

    actions = _make_actions()
    await cl.Message(
        content=(f"---\n**新对话已开始。**\n> 会话 ID：`{new_session_id}`"),
        author="RAGentic",
        actions=actions,
    ).send()


@cl.action_callback("clear_memory")
async def on_clear_memory(action: cl.Action):
    new_session_id = str(uuid.uuid4())
    cl.user_session.set("session_id", new_session_id)

    actions = _make_actions()
    await cl.Message(
        content=(f"记忆已清空。\n> 新会话 ID：`{new_session_id}`"),
        author="RAGentic",
        actions=actions,
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    query = message.content.strip()

    is_valid, error_msg = SecurityManager.validate_query(query)
    if not is_valid:
        logger.warning(f"无效查询：{error_msg}")
        await cl.Message(content=f"❌ 错误：{error_msg}", author="System").send()
        return

    query = SecurityManager.sanitize_input(query)

    session_id = cl.user_session.get("session_id") or str(uuid.uuid4())
    settings = cl.user_session.get("settings") or {}
    logger.info(f"[HANDLE_MESSAGE] session_id={session_id}, settings={settings}")

    use_memory = bool(settings.get("use_memory", False))
    top_k = int(settings.get("top_k", 5))
    use_stream = bool(settings.get("use_stream", True))
    use_sparse = bool(settings.get("use_sparse", False))
    use_reranker = bool(settings.get("use_reranker", False))
    enable_think = bool(settings.get("enable_think", False))

    if use_stream:
        await _handle_stream(
            query,
            session_id,
            use_memory,
            top_k,
            use_sparse,
            use_reranker,
            enable_think,
        )
    else:
        await _handle_sync(
            query,
            session_id,
            use_memory,
            top_k,
            use_sparse,
            use_reranker,
            enable_think,
        )


async def _get_pipeline_and_config(
    session_id: str, use_memory: bool, top_k: int, use_sparse: bool, use_reranker: bool, enable_think: bool
):
    """获取 pipeline 实例和请求配置（带缓存，按会话ID复用管道实例）"""

    # 检查缓存中是否已有该会话的管道实例
    if session_id in _pipeline_cache:
        logger.info(f"复用会话 {session_id} 的管道实例")
        return _pipeline_cache[session_id]

    config = AppConfig()

    req_config = config.create_request_config(
        use_kb=True,
        use_tool=True,
        use_memory=use_memory,
        top_k=top_k,
        use_sparse=use_sparse,
        use_reranker=use_reranker,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_think}},
    )

    pipeline = LangGraphQAPipeline(req_config)
    _pipeline_cache[session_id] = pipeline
    logger.info(f"为会话 {session_id} 创建新的管道实例，缓存大小: {len(_pipeline_cache)}")
    return pipeline


async def _handle_stream(
    query: str,
    session_id: str,
    use_memory: bool,
    top_k: int,
    use_sparse: bool,
    use_reranker: bool,
    enable_think: bool,
):
    """流式处理 - 纯 async"""
    pipeline = await _get_pipeline_and_config(
        session_id, use_memory, top_k, use_sparse, use_reranker, enable_think
    )

    answer_msg = cl.Message(content="", author="RAGentic")
    answer_started = False
    active_step: Optional[cl.Step] = None

    try:
        async for event in pipeline.ask_stream(
            query=query,
            thread_id=session_id,
            langfuse_session_id=session_id,
        ):
            status = event.get("status")
            node = event.get("node", "")
            logger.info(
                f"[DEBUG STREAM] node={node}, status={status}, event_keys={list(event.keys())}"
            )

            if status == "error":
                err_text = event.get("error", "未知错误")
                if answer_started:
                    await answer_msg.stream_token(f"\n\n❌ 错误：{err_text}")
                    await answer_msg.send()
                else:
                    await cl.Message(
                        content=f"❌ 错误：{err_text}", author="System"
                    ).send()
                return

            elif status == "processing" and node in _SHOW_NODES:
                display_name = _NODE_NAMES.get(node, node)
                step_type = "retrieval" if node == "retrieve_knowledge" else "tool"
                active_step = cl.Step(
                    name=display_name,
                    type=step_type,
                )
                await active_step.__aenter__()
                state_info = event.get("state", {})
                active_step.output = _format_step_output(node, state_info)
                await active_step.__aexit__(None, None, None)

            elif status == "processing" and node == "tools_node":
                state_info = event.get("state", {})
                tool_ctx = state_info.get("tool_context", "")
                if tool_ctx:
                    tool_step = cl.Step(
                        name="🔧 工具执行结果",
                        type="tool",
                    )
                    await tool_step.__aenter__()
                    tool_step.output = tool_ctx[:500] + (
                        "..." if len(tool_ctx) > 500 else ""
                    )
                    await tool_step.__aexit__(None, None, None)

            elif status == "complete" and node in (
                "generate_answer",
                "agent_node",
                "build_context",
            ):
                answer = event.get("answer", "")
                context = event.get("context", "") or ""

                if not answer_started:
                    await answer_msg.send()
                    answer_started = True

                if answer:
                    for i in range(0, len(answer), _STREAM_CHUNK_SIZE):
                        await answer_msg.stream_token(
                            answer[i : i + _STREAM_CHUNK_SIZE]
                        )
                        await asyncio.sleep(0)

                if context:
                    ctx_preview = (
                        context[:_CONTEXT_PREVIEW_MAX_LEN] + "\n\n...(截断)"
                        if len(context) > _CONTEXT_PREVIEW_MAX_LEN
                        else context
                    )
                    answer_msg.elements = [
                        cl.Text(
                            name="📚 检索参考源",
                            content=ctx_preview,
                            display="side",
                        )
                    ]

                await answer_msg.send()

    except Exception as e:
        logger.exception("流式问答处理异常")
        await cl.Message(content=f"❌ 错误：{str(e)}", author="System").send()

    if not answer_started:
        await cl.Message(content="抱歉，未能生成答案。", author="RAGentic").send()


async def _handle_sync(
    query: str,
    session_id: str,
    use_memory: bool,
    top_k: int,
    use_sparse: bool,
    use_reranker: bool,
    enable_think: bool,
):
    """同步处理 - 纯 async"""
    pipeline = await _get_pipeline_and_config(
        session_id, use_memory, top_k, use_sparse, use_reranker, enable_think
    )

    try:
        result = await pipeline.ask(
            query=query,
            thread_id=session_id,
            langfuse_session_id=session_id,
        )
    except Exception as e:
        logger.exception("同步问答处理异常")
        await cl.Message(content=f"❌ 错误：{str(e)}", author="System").send()
        return

    if result.get("error"):
        await cl.Message(content=f"❌ 错误：{result['error']}", author="System").send()
        return

    answer = result.get("answer", "无法生成答案")
    context = result.get("context", "") or result.get("kb_context", "")

    elements = []
    if context:
        ctx_preview = (
            context[:_CONTEXT_PREVIEW_MAX_LEN] + "\n\n...(截断)"
            if len(context) > _CONTEXT_PREVIEW_MAX_LEN
            else context
        )
        elements.append(
            cl.Text(
                name="📚 检索参考源",
                content=ctx_preview,
                display="side",
            )
        )

    await cl.Message(content=answer, author="RAGentic", elements=elements).send()


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="知识库问答",
            message="请根据知识库介绍这个系统的主要功能",
        ),
        cl.Starter(
            label="联网搜索",
            message="今天有什么重要的 AI 新闻？",
        ),
        cl.Starter(
            label="技术问答",
            message="请介绍一下 RAG 技术的原理和应用场景",
        ),
    ]


def _format_step_output(node: str, state: dict) -> str:
    if node == "retrieve_knowledge":
        ctx = state.get("kb_context") or state.get("final_context", "")
        if ctx:
            return ctx[:300] + "..." if len(ctx) > 300 else ctx
        return "知识库检索中..."
    if node == "call_tools":
        ctx = state.get("tool_context", "")
        return ctx[:300] + "..." if len(ctx) > 300 else ctx if ctx else "工具调用中..."
    if node == "transform_query":
        return "正在改写/扩展查询..."
    return f"{_NODE_NAMES.get(node, node)} 执行中..."
