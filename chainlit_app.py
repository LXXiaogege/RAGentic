# -*- coding: utf-8 -*-
"""
RAGentic - Chainlit UI
Replaces web_app.py (Gradio) with a Chainlit 2.10 interface.
Preserves all features: streaming, security validation, runtime config injection,
session management, knowledge base / tool / memory toggles.
"""

import asyncio
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from src.configs.config import AppConfig
from typing import Optional

import chainlit as cl
from chainlit.input_widget import Slider, Switch

from langfuse import Langfuse

from src.configs.logger_config import setup_logger
from src.cores.pipeline_langgraph import LangGraphQAPipeline
from src.utils.security import SecurityManager

logger = setup_logger(__name__)

# ===== Module-level singletons (initialized once at startup) =====
_app_config: Optional[AppConfig] = None
_pipeline: Optional[LangGraphQAPipeline] = None
_executor = ThreadPoolExecutor(max_workers=4)
_pipeline_lock = asyncio.Lock()

# ===== Node display names (Chinese) =====
_NODE_NAMES = {
    "retrieve_knowledge": "检索知识库",
    "call_tools": "调用工具",
    "transform_query": "改写查询",
    "build_context": "构建上下文",
    "parse_query": "解析查询",
    "generate_answer": "生成答案",
    "update_memory": "更新记忆",
}

# Nodes that should be shown as Steps in the UI
_SHOW_NODES = {"retrieve_knowledge", "call_tools", "transform_query"}

# Streaming output chunk size (characters per chunk)
_STREAM_CHUNK_SIZE = 6
# Context preview max length for display
_CONTEXT_PREVIEW_MAX_LEN = 500


# ===== App Startup =====


@cl.on_app_startup
async def on_startup():
    global _app_config, _pipeline

    _app_config = AppConfig()

    # Mirror web_app.py environment setup
    os.environ["LANGFUSE_SECRET_KEY"] = _app_config.langfuse.secret_key
    os.environ["LANGFUSE_PUBLIC_KEY"] = _app_config.langfuse.public_key
    os.environ["LANGFUSE_HOST"] = _app_config.langfuse.host
    os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = "production"
    os.environ["OPENAI_API_KEY"] = _app_config.llm.api_key
    os.environ["OPENAI_API_BASE"] = _app_config.llm.base_url

    try:
        Langfuse()
        logger.info("Langfuse 客户端初始化成功")
    except Exception as e:
        logger.warning(f"Langfuse 客户端初始化失败：{e}")

    _pipeline = LangGraphQAPipeline(_app_config)
    logger.info("RAGentic pipeline 初始化完成")


@cl.on_app_shutdown
async def on_shutdown():
    """清理应用资源"""
    global _executor, _pipeline
    logger.info("开始清理应用资源...")

    if _executor:
        _executor.shutdown(wait=False)
        logger.info("ThreadPoolExecutor 已关闭")

    if _pipeline:
        _pipeline._close_mcp_client()
        logger.info("Pipeline MCP 客户端已关闭")

    logger.info("应用资源清理完成")


# ===== Chat Session Start =====


@cl.on_chat_start
async def on_chat_start():
    session_id = str(uuid.uuid4())
    cl.user_session.set("session_id", session_id)

    default_settings = {
        "use_kb": False,
        "use_tool": False,
        "use_memory": False,
        "top_k": 5,
        "use_stream": True,
        "use_sparse": False,
        "use_reranker": False,
        "enable_think": False,
    }
    cl.user_session.set("settings", default_settings)

    # Build ChatSettings panel (rendered in sidebar via config.toml)
    settings = cl.ChatSettings(
        [
            Switch(
                id="use_kb",
                label="📚 检索知识库",
                initial=False,
                description="基于本地文档回答",
            ),
            Switch(
                id="use_tool",
                label="🌐 联网/工具",
                initial=False,
                description="调用搜索或外部 API",
            ),
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
                label="🔀 混合检索",
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

    # Welcome message with action buttons
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


# ===== Settings Update =====


@cl.on_settings_update
async def on_settings_update(settings: dict):
    cl.user_session.set("settings", settings)
    logger.debug(f"设置已更新: {settings}")


# ===== Action Callbacks =====


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
    # LangGraph MemorySaver isolates history by thread_id.
    # Generating a new session_id is equivalent to clearing memory.
    new_session_id = str(uuid.uuid4())
    cl.user_session.set("session_id", new_session_id)

    actions = _make_actions()
    await cl.Message(
        content=(f"记忆已清空。\n> 新会话 ID：`{new_session_id}`"),
        author="RAGentic",
        actions=actions,
    ).send()


# ===== Main Message Handler =====


@cl.on_message
async def on_message(message: cl.Message):
    query = message.content.strip()

    # Security validation (mirrors web_app.py)
    is_valid, error_msg = SecurityManager.validate_query(query)
    if not is_valid:
        logger.warning(f"无效查询：{error_msg}")
        await cl.Message(content=f"❌ 错误：{error_msg}", author="System").send()
        return

    query = SecurityManager.sanitize_input(query)

    # Load per-session state
    session_id = cl.user_session.get("session_id") or str(uuid.uuid4())
    settings = cl.user_session.get("settings") or {}

    use_kb = bool(settings.get("use_kb", False))
    use_tool = bool(settings.get("use_tool", False))
    use_memory = bool(settings.get("use_memory", False))
    top_k = int(settings.get("top_k", 5))
    use_stream = bool(settings.get("use_stream", True))
    use_sparse = bool(settings.get("use_sparse", False))
    use_reranker = bool(settings.get("use_reranker", False))
    enable_think = bool(settings.get("enable_think", False))

    # Build per-request config (avoids deepcopy of expensive objects)
    req_config = _app_config.create_request_config(
        use_kb=use_kb,
        use_tool=use_tool,
        use_memory=use_memory,
        top_k=top_k,
        use_sparse=use_sparse,
        use_reranker=use_reranker,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_think}},
    )

    # Inject config (lock ensures no concurrent mutation)
    async with _pipeline_lock:
        _pipeline.config = req_config
        if use_stream:
            await _handle_stream(query, session_id)
        else:
            await _handle_sync(query, session_id)


# ===== Streaming Handler =====


async def _handle_stream(query: str, session_id: str):
    """
    Bridge between the synchronous ask_stream() generator and Chainlit's async loop.

    ask_stream() yields:
      {"node": str, "status": "processing", "state": {...}}  — per node
      {"node": "generate_answer", "status": "complete", "answer": str, "context": str}
      {"status": "error", "error": str}
    """
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _run_generator():
        try:
            for event in _pipeline.ask_stream(
                query=query,
                thread_id=session_id,
                langfuse_session_id=session_id,
            ):
                asyncio.run_coroutine_threadsafe(queue.put(event), loop).result()
        except Exception as e:
            logger.exception("流式问答生成器异常")
            asyncio.run_coroutine_threadsafe(
                queue.put({"status": "error", "error": str(e)}), loop
            ).result()
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

    loop.run_in_executor(_executor, _run_generator)

    answer_msg = cl.Message(content="", author="RAGentic")
    answer_started = False
    active_step: Optional[cl.Step] = None

    while True:
        event = await queue.get()
        if event is None:
            break

        status = event.get("status")
        node = event.get("node", "")

        if status == "error":
            err_text = event.get("error", "未知错误")
            if answer_started:
                await answer_msg.stream_token(f"\n\n❌ 错误：{err_text}")
                await answer_msg.send()
            else:
                await cl.Message(content=f"❌ 错误：{err_text}", author="System").send()
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

        elif status == "complete" and node == "generate_answer":
            answer = event.get("answer", "")
            context = event.get("context", "") or ""

            if not answer_started:
                await answer_msg.send()
                answer_started = True

            # Simulate streaming by sending in small chunks
            for i in range(0, len(answer), _STREAM_CHUNK_SIZE):
                await answer_msg.stream_token(answer[i : i + _STREAM_CHUNK_SIZE])
                await asyncio.sleep(0)  # yield control to event loop

            # Attach context as a side element if available
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

    if not answer_started:
        await cl.Message(content="抱歉，未能生成答案。", author="RAGentic").send()


# ===== Sync (Non-streaming) Handler =====


async def _handle_sync(query: str, session_id: str):
    """Non-streaming fallback: run pipeline.ask() in a thread pool."""
    loop = asyncio.get_event_loop()

    try:
        result = await loop.run_in_executor(
            _executor,
            lambda: _pipeline.ask(
                query=query,
                thread_id=session_id,
                langfuse_session_id=session_id,
            ),
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


# ===== Starter Examples =====


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


# ===== Helper Functions =====


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
