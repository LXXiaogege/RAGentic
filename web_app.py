# -*- coding: utf-8 -*-
"""
@Time ： 2025/11/18 18:26
@Auth ： 吕鑫
@File ：web_app.py
@IDE ：PyCharm
"""

import os
import uuid
import gradio as gr
from langfuse import get_client
import asyncio
from src.cores.pipeline import QAPipeline
from src.configs.config import AppConfig
from src.configs.logger_config import setup_logger
from src.configs.retrieve_config import SearchConfig

logger = setup_logger(__name__)

# ================= 配置初始化 (保持不变) =================
config = AppConfig()
os.environ["LANGFUSE_SECRET_KEY"] = config.langfuse.secret_key
os.environ["LANGFUSE_PUBLIC_KEY"] = config.langfuse.public_key
os.environ["LANGFUSE_HOST"] = config.langfuse.host
os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = "production"
os.environ["OPENAI_API_KEY"] = config.llm.api_key
os.environ["OPENAI_API_BASE"] = config.llm.base_url

try:
    langfuse_client = get_client()
    logger.info("Langfuse 客户端初始化成功")
except Exception as e:
    logger.warning(f"Langfuse 客户端初始化失败: {e}")
    langfuse_client = None

pipline = None


async def init_rag_client():
    global pipeline
    pipeline = QAPipeline(config, langfuse_client)
    await pipeline.init_components()
    logger.info("QAPipeline 初始化完成")


asyncio.run(init_rag_client())


async def chat_ask(query, use_kb, use_tool, use_memory, top_k, use_sparse, use_reranker, enable_think,
                   session_id, history):
    """同步问答接口"""
    if not query or not query.strip():
        return "", history
    if not session_id:
        session_id = str(uuid.uuid4())

    search_config = SearchConfig(
        use_sparse=use_sparse,
        use_reranker=use_reranker,
        use_kb=use_kb,
        use_memory=use_memory,
        use_tool=use_tool,
        top_k=top_k,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": enable_think
            }
        },
        session_id=session_id
    )
    try:
        result = await pipeline.ask(query=query, config=search_config)
        if "error" in result:
            answer = f"错误: {result['error']}"
        else:
            answer = result["answer"]
            if result.get("context"):
                context_preview = result["context"][:200] + "..." if len(result["context"]) > 200 else result["context"]
                # 使用 Details 标签折叠上下文，界面更干净
                answer = f"{answer}\n\n<details><summary>📚 检索参考源 (点击展开)</summary>\n\n{context_preview}\n</details>"
        history.append([query, answer])
        return "", history
    except Exception as e:
        logger.exception("问答处理异常")
        history.append([query, f"错误: {str(e)}"])
        return "", history


async def chat_stream(query, use_kb, use_tool, use_memory, top_k, use_sparse, use_reranker, enable_think, session_id,
                      history):
    """流式问答接口"""
    if not query or not query.strip():
        yield history
        return
    if not session_id:
        session_id = str(uuid.uuid4())
    try:
        search_config = SearchConfig(
            use_sparse=use_sparse,
            use_reranker=use_reranker,
            use_kb=use_kb,
            use_memory=use_memory,
            use_tool=use_tool,
            top_k=top_k,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": enable_think
                }
            },
            session_id=session_id
        )
        answer = ""
        history.append([query, ""])
        async for chunk in pipeline.ask_stream(query, config=search_config):
            if "error" in chunk:
                history[-1][1] = f"❌ 错误: {chunk['error']}"
                yield history
                return
            elif "delta" in chunk:
                answer += chunk["delta"]
                history[-1][1] = answer
                yield history
    except Exception as e:
        logger.exception("流式问答处理异常")
        if history and history[-1][0] == query:
            history[-1][1] = f"❌ 错误: {str(e)}"
        else:
            history.append([query, f"❌ 错误: {str(e)}"])
        yield history


def clear_memory_action():
    pipeline.clear_conversation()
    return [], ""


def clear_and_reset():
    return [], "", str(uuid.uuid4())


# ================= 现代化 UI 设计 =================

# 1. 高级 CSS 样式
modern_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

/* ============ 强制全屏无边距 ============ */
html, body, #root, .gradio-container {
    margin: 0 !important;
    padding: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    overflow: hidden !important;
    background-color: var(--bg-color) !important;
}

/* 清除 Gradio 自动生成的 wrapper */
.gradio-container > * {
    margin: 0 !important;
    padding: 0 !important;
    max-width: none !important;
    width: 100% !important;
    height: 100% !important;
}

/* ============ 原有变量和样式 ============ */
:root {
    --primary-color: #6366f1;
    --primary-hover: #4f46e5;
    --bg-color: #f3f4f6;
    --card-bg: #ffffff;
    --text-main: #1f2937;
    --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
    --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
}

/* === 移除 main-container 的限制 === */
.main-container {
    width: 100% !important;
    height: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    display: flex;
    flex-direction: row;
    gap: 0;
}

/* 左侧侧边栏卡片 - 调整高度以填满 */
.sidebar-panel {
    background: var(--card-bg);
    border-radius: 0; /* 可选：去掉圆角更贴边 */
    padding: 24px;
    border: 1px solid #e5e7eb;
    box-shadow: var(--shadow-md);
    height: 100vh; /* 填满视口高度 */
    display: flex;
    flex-direction: column;
}

/* 右侧对话区 - 同样填满 */
.chat-panel {
    display: flex;
    flex-direction: column;
    height: 100vh;
    width: 100%;
}

/* === 聊天气泡：整体容器 === */
#chatbot {
    padding: 20px !important;
    background: var(--card-bg) !important;
    overflow-y: auto;
    scroll-behavior: smooth; /* 平滑滚动 */
}

/* === 优化输入框高度和外观 === */
.input-area .textbox {
    min-height: 300px !important;       /* 关键：设置最小高度 */
    padding: 14px 16px !important;     /* 上下内边距加大 */
    font-size: 1rem !important;
    border-radius: 18px !important;
    border: 1px solid #d1d5db !important;
    box-shadow: var(--shadow-sm) !important;
    transition: all 0.2s ease;
}

.input-area .textbox:focus {
    outline: none !important;
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
}

/* 确保多行展开时也好看 */
.input-area .textbox textarea {
    min-height: 60px !important;
    resize: none; /* 禁止手动拖拽调整大小，保持整洁 */
}


/* === 用户消息 === */
.message.user {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
    border-radius: 18px 18px 4px 18px !important;
    border: none !important;
    padding: 14px 18px !important;
    margin-bottom: 16px !important;
    box-shadow: 0 2px 6px rgba(99, 102, 241, 0.2) !important;
    max-width: 85% !important;
    align-self: flex-end !important;
    word-break: break-word;
}

/* === 助手消息 === */
.message.bot {
    background: #ffffff !important;
    color: #1f2937 !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 18px 18px 18px 4px !important;
    padding: 14px 18px !important;
    margin-bottom: 16px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    max-width: 85% !important;
    align-self: flex-start !important;
    word-break: break-word;
}

* === 头像区域优化（隐藏默认头像背景）=== */
.message.user .avatar-container,
.message.bot .avatar-container {
    display: flex;
    justify-content: center;
    align-items: center;
}

.message.user .avatar,
.message.bot .avatar {
    background: transparent !important;
    border: none !important;
}

#send-btn {
    background: var(--primary-color) !important;
    color: white !important;
    border-radius: 12px !important;
    width: 80px !important;
    height: 50px !important;
    font-weight: 600 !important;
    transition: all 0.2s;
    border: none !important;
}

#send-btn:hover {
    background: var(--primary-hover) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
}

::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: transparent; 
}
::-webkit-scrollbar-thumb {
    background: #d1d5db; 
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #9ca3af; 
}
"""

# 2. 创建主题
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    text_size="lg",
    radius_size="lg"
)
with gr.Blocks(title="RAG 智能助手", theme=theme, css=modern_css) as demo:
    session_id_state = gr.State(value="")

    # 直接使用 Row，不加额外类（或保留但 CSS 已覆盖）
    with gr.Row(elem_classes="main-container"):
        # 左侧：控制面板
        with gr.Column(scale=3, min_width=300):
            with gr.Column(elem_classes="sidebar-panel"):
                # 标题区
                gr.HTML("""
                <div class="sidebar-header">
                    <h2>Knowledge Base Bot</h2>
                    <p style="color: #6b7280; font-size: 0.9rem;">基于 RAG 的企业级智能问答助手</p>
                </div>
                <hr style="margin: 15px 0; opacity: 0.2;">
                """)

                # 核心功能开关
                with gr.Group():
                    gr.Markdown("### ✨ 核心能力", elem_id="core-label")
                    use_kb = gr.Checkbox(label="📚 检索知识库", value=False, info="基于本地文档回答")
                    use_tool = gr.Checkbox(label="🌐 联网/工具", value=False, info="调用搜索或外部API")
                    use_memory_cb = gr.Checkbox(label="💬 上下文记忆", value=False, info="记住多轮对话内容")

                # 高级设置 (折叠)
                with gr.Accordion("高级参数设置", open=False):
                    top_k = gr.Slider(1, 20, value=5, step=1, label="检索数量 (Top-K)")
                    use_stream_cb = gr.Checkbox(label="流式输出", value=False)
                    use_sparse_cb = gr.Checkbox(label="混合检索", value=False)
                    use_reranker_cb = gr.Checkbox(label="重排序", value=False)
                    use_think_cb = gr.Checkbox(label="深度思考", value=False)
                    # 隐藏的 Session ID 显示，方便调试但平时不占地
                    session_id_display = gr.Textbox(label="当前会话 ID", interactive=False, text_align="right", lines=1)

                # 底部操作区
                gr.Markdown("### 🗑️ 会话管理")
                with gr.Row():
                    clear_btn = gr.Button("新对话", variant="secondary", size="sm")
                    clear_memory_btn = gr.Button("清空记忆", variant="stop", size="sm")

        # 右侧：对话主视区
        with gr.Column(scale=9, elem_classes="chat-panel"):  # 添加 class 便于控制
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                show_label=False,
                avatar_images=("https://api.dicebear.com/7.x/notionists/svg?seed=User",
                               "https://api.dicebear.com/7.x/bottts/svg?seed=RAGBot"),
                bubble_full_width=False,
                render_markdown=True,
                show_copy_button=True
            )

            with gr.Row(elem_classes="input-area"):
                msg = gr.Textbox(
                    placeholder="在此输入您的问题...",
                    lines=2,
                    max_lines=5,
                    show_label=False,
                    container=False,
                    scale=8,
                    autofocus=True
                )
                submit_btn = gr.Button("发送", elem_id="send-btn", scale=1)


    # ================= 事件绑定 =================

    # 包装响应函数以处理 State
    async def respond_wrapper(message, history, kb, tool, memory, stream, k, sparse, rerank, enable_think, sess_id):
        if not message.strip():
            yield "", history, sess_id
            return

        # 确保有 Session ID
        current_sess_id = sess_id if sess_id else str(uuid.uuid4())

        if stream:
            async for updated_history in chat_stream(
                    message, kb, tool, memory, k, sparse, rerank, enable_think, current_sess_id, history
            ):
                yield "", updated_history, current_sess_id
        else:
            _, updated_history = await chat_ask(
                message, kb, tool, memory, k, sparse, rerank, enable_think, current_sess_id, history
            )
            yield "", updated_history, current_sess_id


    # 包装重置函数
    def reset_wrapper():
        new_sid = str(uuid.uuid4())
        return [], "", new_sid, new_sid


    # 绑定 Enter 发送
    msg.submit(
        respond_wrapper,
        inputs=[msg, chatbot, use_kb, use_tool, use_memory_cb, use_stream_cb,
                top_k, use_sparse_cb, use_reranker_cb, use_think_cb, session_id_state],
        outputs=[msg, chatbot, session_id_state]
    ).then(
        lambda s: s, inputs=[session_id_state], outputs=[session_id_display]  # 更新显示的 ID
    )

    # 绑定 按钮 发送
    submit_btn.click(
        respond_wrapper,
        inputs=[msg, chatbot, use_kb, use_tool, use_memory_cb, use_stream_cb,
                top_k, use_sparse_cb, use_reranker_cb, use_think_cb, session_id_state],
        outputs=[msg, chatbot, session_id_state]
    ).then(
        lambda s: s, inputs=[session_id_state], outputs=[session_id_display]
    )

    # 绑定 清除按钮
    clear_btn.click(
        reset_wrapper,
        outputs=[chatbot, msg, session_id_state, session_id_display]
    )

    clear_memory_btn.click(
        clear_memory_action,
        outputs=[chatbot, msg]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        debug=True,
        prevent_thread_lock=True
    )
