# -*- coding: utf-8 -*-
"""
@Time：2025/6/8 10:06
@Auth：吕鑫
@File：pipeline_langgraph.py
@IDE：PyCharm

基于 LangGraph 的 QA Pipeline 重构
使用状态图来管理复杂的 RAG 工作流
"""

import asyncio
import concurrent.futures
import time
from functools import wraps
from typing import Annotated, Any, Callable, Dict, Generator, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from langfuse.decorators import langfuse_context
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from pydantic import BaseModel

from src.configs.config import AppConfig
from src.configs.logger_config import setup_logger
from src.cores.message_builder import MessageBuilder
from src.cores.query_transformer import QueryTransformer
from src.db_services.milvus.connection_manager import MilvusConnectionManager
from src.mcp.mcp_client import MCPClient, mcp_main
from src.models.embedding import TextEmbedding
from src.models.llm import LLMWrapper

logger = setup_logger(__name__)


def timing_decorator(func: Callable) -> Callable:
    """性能监控装饰器 - 记录节点执行时间"""

    @wraps(func)
    def wrapper(self, state: QAState) -> QAState:
        start = time.time()
        result = func(self, state)
        duration = time.time() - start
        self.logger.info(f"{func.__name__} 耗时：{duration:.3f}s")
        return result

    return wrapper


class QAState(BaseModel):
    """QA Pipeline的状态定义"""

    # 核心数据
    messages: Annotated[List[Any], add_messages] = []
    original_query: str
    transformed_query: Optional[str] = None

    # 检索结果
    kb_context: Optional[str] = None
    tool_context: Optional[str] = None
    final_context: Optional[str] = None

    error: Optional[str] = None


def build_prompt_messages(
    state: QAState,
    system_prompt: str,
    use_memory: bool = False,
    memory_limit: int = 10,
) -> List[BaseMessage]:
    """构建用于 LLM 调用的消息序列"""
    messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]

    if use_memory and state.messages:
        history_messages = [
            msg for msg in state.messages if not isinstance(msg, SystemMessage)
        ]
        messages.extend(history_messages[-memory_limit:])

    query_content = state.original_query

    if state.transformed_query and state.transformed_query != state.original_query:
        query_content = (
            f"原始问题：{state.original_query}\n转换后的问题：{state.transformed_query}"
        )

    if state.final_context:
        query_content = f"上下文信息：\n{state.final_context}\n\n问题：{query_content}"

    messages.append(HumanMessage(content=query_content))

    return messages


class LangGraphQAPipeline:
    """基于LangGraph的QA Pipeline"""

    def __init__(self, config: AppConfig):
        self.logger = logger
        self.config = config
        self._init_components()
        self._build_graph()

    def _init_components(self):
        """初始化所有组件"""
        self.logger.info("初始化QA Pipeline组件...")

        # 文本嵌入
        self.embeddings = TextEmbedding(self.config)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.splitter.chunk_size,
            chunk_overlap=self.config.splitter.chunk_overlap,
        )
        self.message_builder = MessageBuilder(self.config.message_builder)

        # 向量数据库
        self.db_connection_manager = MilvusConnectionManager(
            self.embeddings,
            self.text_splitter,
            self.config.milvus,
            self.config.retrieve,
            self.config.reranker,
            self.config.bm25,
        )

        # LLM包装器
        self.llm_caller = LLMWrapper(self.config.llm)

        # 查询转换器
        self.query_transformer = QueryTransformer(
            self.llm_caller,
            self.message_builder,
            self.embeddings,
            self.db_connection_manager,
        )

        # MCP客户端
        self.mcp_client = MCPClient(self.llm_caller)

        # 检查点保存器
        self.checkpointer = MemorySaver()

        # 初始化 Langfuse 客户端
        self.langfuse_client = None
        self.langfuse_handler = None
        if self.config.langfuse.secret_key:
            try:
                self.langfuse_client = Langfuse(
                    secret_key=self.config.langfuse.secret_key,
                    public_key=self.config.langfuse.public_key,
                    host=self.config.langfuse.host,
                )
                self.langfuse_handler = CallbackHandler()
                self.logger.info("Langfuse 客户端初始化完成")
            except Exception as e:
                self.logger.warning(f"Langfuse 客户端初始化失败: {e}")

    def _build_graph(self):
        """构建LangGraph工作流"""
        # 创建状态图
        workflow = StateGraph(QAState)

        # 添加节点
        workflow.add_node("parse_query", self._parse_query)  # 解析查询及配置参数
        workflow.add_node("transform_query", self._transform_query)  # 转换查询
        workflow.add_node(
            "check_retrieve_knowledge", self._check_retrieve_knowledge
        )  # 检查是否需要检索知识库
        workflow.add_node("retrieve_knowledge", self._retrieve_knowledge)
        workflow.add_node("check_call_tools", self._check_call_tools)
        workflow.add_node("call_tools", self._call_tools)
        workflow.add_node("build_context", self._build_context)
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("update_memory", self._update_memory)
        workflow.add_node("handle_error", self._handle_error)

        # 定义流程
        self._define_workflow(workflow)

        # 编译图
        self.graph = workflow.compile(checkpointer=self.checkpointer)

    def _define_workflow(self, workflow: StateGraph):
        """定义工作流程"""
        # 入口：解析查询
        workflow.add_edge(START, "parse_query")

        # 解析查询后的条件分支
        workflow.add_conditional_edges(
            "parse_query",
            self._should_transform_query,
            {
                "transform": "transform_query",
                "skip_transform": "check_retrieve_knowledge",
                "error": "handle_error",
            },
        )
        # 当执行transform_query时
        workflow.add_conditional_edges(
            "transform_query",
            self._should_retrieve_knowledge,
            {
                "do_retrieve": "retrieve_knowledge",
                "skip_retrieve": "check_call_tools",
                "error": "handle_error",
            },
        )

        workflow.add_conditional_edges(
            "check_call_tools",
            self._should_call_tools,
            {
                "call_tools": "call_tools",
                "skip_tools": "build_context",
                "error": "handle_error",
            },
        )

        # 是否需要知识库检索
        workflow.add_conditional_edges(
            "check_retrieve_knowledge",
            self._should_retrieve_knowledge,
            {
                "do_retrieve": "retrieve_knowledge",
                "skip_retrieve": "check_call_tools",
                "error": "handle_error",
            },
        )

        # 知识检索后的条件分支
        workflow.add_conditional_edges(
            "retrieve_knowledge",
            self._should_call_tools,
            {
                "call_tools": "call_tools",
                "skip_tools": "build_context",
                "error": "handle_error",
            },
        )

        # 工具调用后构建上下文
        workflow.add_edge("call_tools", "build_context")

        # 构建上下文后生成答案
        workflow.add_edge("build_context", "generate_answer")

        # 生成答案后的条件分支
        workflow.add_conditional_edges(
            "generate_answer",
            self._should_update_memory,
            {"update_memory": "update_memory", "finish": END, "error": "handle_error"},
        )

        # 更新记忆后结束
        workflow.add_edge("update_memory", END)

        # 错误处理后结束
        workflow.add_edge("handle_error", END)
        pass

    # =================== 节点实现 ===================
    @timing_decorator
    def _parse_query(self, state: QAState) -> QAState:
        """解析查询，设置默认参数"""
        try:
            self.logger.info(f"解析查询：{state.original_query}")
            defaults = dict(
                k=self.config.retrieve.top_k,
                use_sparse=self.config.retrieve.use_sparse,
                use_reranker=self.config.retrieve.use_reranker,
                use_knowledge_base=self.config.retrieve.use_kb,
                use_tools=self.config.retrieve.use_tool,
                use_memory=self.config.retrieve.use_memory,
                error=None,
            )
            merged = {
                **defaults,
                **(state.model_dump() if isinstance(state, BaseModel) else state),
            }
            state = QAState(**merged)
            return state
        except Exception as e:
            self.logger.exception(f"解析查询失败：{e}")
            state.error = f"解析查询失败：{str(e)}"
            return state

    @timing_decorator
    def _transform_query(self, state: QAState) -> QAState:
        """查询转换"""
        try:
            self.logger.info(f"转换查询：{state.original_query}")

            if self.config.retrieve.use_rewrite:
                transformed_query = self.query_transformer.transform_query(
                    state.original_query, self.config.retrieve.rewrite_mode
                )
                state.transformed_query = transformed_query
                self.logger.info(f"查询转换完成：{transformed_query}")
            else:
                state.transformed_query = ""

            return state

        except Exception as e:
            self.logger.exception(f"查询转换失败：{e}")
            state.error = f"查询转换失败：{str(e)}"
            return state

    def _check_retrieve_knowledge(self, state: QAState) -> QAState:
        return state

    def _check_call_tools(self, state: QAState) -> QAState:
        return state

    @timing_decorator
    def _retrieve_knowledge(self, state: QAState) -> QAState:
        """知识库检索"""
        try:
            if not self.config.retrieve.use_kb:
                state.kb_context = ""
                return state

            query = getattr(state, "transformed_query", "") or state.original_query
            self.logger.info(f"开始知识库检索：{query}")

            results = self._perform_search(query)
            state.kb_context = self._format_results(results)

            return state

        except Exception as e:
            self.logger.exception(f"知识库检索失败：{e}")
            state.error = f"知识库检索失败：{str(e)}"
            return state

    def _perform_search(self, query: str) -> list[dict]:
        """执行检索 - 根据配置选择 HyDE 或标准检索"""
        if (
            self.config.retrieve.use_rewrite
            and self.config.retrieve.rewrite_mode == "hyde"
        ):
            return asyncio.run(
                self.query_transformer.hyde_search(
                    query, self.config.retrieve.search_config
                )
            )
        else:
            return asyncio.run(
                self.db_connection_manager.asearch(
                    query, self.config.retrieve.search_config
                )
            )

    def _format_results(self, results: list[dict]) -> str:
        """格式化检索结果"""
        if not results:
            self.logger.warning("知识库检索未返回结果")
            return "未检索到相关知识。"

        valid_results = [doc for doc in results if doc and doc.get("text")]
        if not valid_results:
            return "未检索到有效知识。"

        kb_context = "\n".join(
            [f"【文档{i + 1}】{doc['text']}" for i, doc in enumerate(valid_results)]
        )
        self.logger.info(f"知识库检索返回 {len(results)} 条结果")
        return f"【知识库检索内容】\n{kb_context}"

    @timing_decorator
    def _call_tools(self, state: QAState) -> QAState:
        """调用外部工具"""
        try:
            if not self.config.retrieve.use_tool:
                state.tool_context = ""
                return state

            query = state.original_query
            self.logger.info(f"调用工具: {query}")

            def run_in_thread():
                return asyncio.run(mcp_main(self.mcp_client, query))

            # 在线程中执行异步任务，防止阻塞主线程
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                tool_result = future.result(timeout=30)

            # 处理工具结果
            if tool_result:
                state.tool_context = f"【工具返回内容】\n{tool_result}"
                self.logger.info(f"工具调用成功，返回内容长度: {len(tool_result)}")
            else:
                state.tool_context = ""
                self.logger.info("工具调用无返回结果")

            return state

        except Exception as e:
            self.logger.exception(f"工具调用失败：{e}")
            state.error = f"工具调用失败：{str(e)}"
            return state

    def _build_context(self, state: QAState) -> QAState:
        """构建最终上下文"""
        try:
            self.logger.info("构建最终上下文")

            context_parts = []

            if state.tool_context:
                context_parts.append(state.tool_context)

            if state.kb_context:
                context_parts.append(state.kb_context)

            state.final_context = "\n\n".join(context_parts)
            self.logger.info(f"上下文构建完成，长度：{len(state.final_context)}")

            return state

        except Exception as e:
            self.logger.exception(f"构建上下文失败：{e}")
            state.error = f"构建上下文失败：{str(e)}"
            return state

    @timing_decorator
    def _generate_answer(self, state: QAState) -> QAState:
        """生成答案"""
        try:
            self.logger.info("开始生成答案")

            # 根据是否启用知识库选择系统提示
            system_prompt = (
                self.config.prompt.kb_system_prompt
                if self.config.retrieve.use_kb
                else self.config.prompt.system_prompt
            )

            # 构建消息
            messages = build_prompt_messages(
                state,
                system_prompt=system_prompt,
                use_memory=self.config.retrieve.use_memory,
                memory_limit=self.config.retrieve.memory_window_size,
            )

            # 调用 LLM 生成答案
            answer = self.llm_caller.chat(messages=messages)

            # 添加答案到消息历史
            if not hasattr(state, "messages") or state.messages is None:
                state.messages = []
            state.messages += [
                HumanMessage(content=state.original_query),
                AIMessage(content=answer),
            ]

            self.logger.info("答案生成完成")
            return state

        except Exception as e:
            self.logger.exception(f"生成答案失败：{e}")
            state.error = f"生成答案失败：{str(e)}"
            return state

    @timing_decorator
    def _update_memory(self, state: QAState) -> QAState:
        """更新记忆"""
        try:
            if self.config.retrieve.use_memory and state.messages:
                max_history = self.config.retrieve.memory_window_size
                state.messages = state.messages[-max_history:]
                self.logger.info(f"记忆更新完成，当前历史消息数：{len(state.messages)}")
            return state
        except Exception as e:
            self.logger.exception(f"更新记忆失败：{e}")
            state.error = f"更新记忆失败：{str(e)}"
            return state

    @timing_decorator
    def _handle_error(self, state: QAState) -> QAState:
        """处理错误"""
        error_msg = state.error or "未知错误"
        self.logger.error(f"处理错误: {error_msg}")

        # 初始化消息历史，如果为空则创建列表
        if not hasattr(state, "messages") or state.messages is None:
            state.messages = []

        # 添加错误消息到对话历史
        state.messages.append(
            AIMessage(content=f"抱歉，处理您的请求时出现错误：{error_msg}")
        )

        return state

    # =================== 条件判断函数 ===================

    def _should_transform_query(self, state: QAState) -> str:
        """判断是否需要转换查询"""
        if state.error:
            return "error"

        if self.config.retrieve.use_rewrite:
            return "transform"
        else:
            return "skip_transform"

    def _should_retrieve_knowledge(self, state: QAState) -> str:
        """判断是否需要做知识库查询"""
        if state.error:
            return "error"

        if self.config.retrieve.use_kb:
            return "do_retrieve"
        else:
            return "skip_retrieve"

    def _should_call_tools(self, state: QAState) -> str:
        """判断是否需要调用工具"""
        if state.error:
            return "error"

        if self.config.retrieve.use_tool:
            return "call_tools"
        else:
            return "skip_tools"

    def _should_update_memory(self, state: QAState) -> str:
        """判断是否需要更新记忆"""
        if state.error:
            return "error"

        if self.config.retrieve.use_memory:
            return "update_memory"
        else:
            return "finish"

    def update_langfuse_trace(self, **kwargs):
        """
        根据传入参数更新 Langfuse trace 信息。
        支持字段: langfuse_session_id, langfuse_user_id
        """
        field_map = {
            "session_id": "langfuse_session_id",
            "user_id": "langfuse_user_id",
        }

        updates = {
            field: kwargs[key] for field, key in field_map.items() if key in kwargs
        }

        if updates:
            for k, v in updates.items():
                self.logger.info(f"Langfuse更新当前追踪{k}：{v}")
            if self.langfuse_client:
                langfuse_context.update_current_trace(**updates)

    # =================== 外部接口 ===================

    def ask(self, query: str, **kwargs) -> Dict[str, Any]:
        """同步问答接口"""
        # 创建配置对象
        config = RunnableConfig(
            configurable={
                "thread_id": kwargs.get("thread_id", "0"),  # 会话ID，标记第几轮对话
            }
        )
        if self.langfuse_handler:
            config["callbacks"] = [self.langfuse_handler]
            self.update_langfuse_trace(**kwargs)

        existing_messages = kwargs.get("messages", [])  # 参数传入，前台来管理

        # 初始化状态
        initial_state_dict = {
            "original_query": query,
            "messages": existing_messages,
        }
        initial_state_object = QAState(**initial_state_dict)

        try:
            result = self.graph.invoke(initial_state_object, config)

            if result.get("error"):
                return {"error": result["error"]}

            # 提取最后的AI消息作为答案
            ai_messages = [
                msg.content
                for msg in result.get("messages", [])
                if isinstance(msg, AIMessage)
            ]

            answer = ai_messages[-1] if ai_messages else "无法生成答案"

            return {
                "question": query,
                "answer": answer,
                "messages": result.get("messages", existing_messages),  # ✅ 返回给前台
                "transformed_query": result.get("transformed_query", ""),
                "context": result.get("final_context", ""),
                "kb_context": result.get("kb_context", ""),
                "tool_context": result.get("tool_context", ""),
            }

        except Exception as e:
            self.logger.error(f"问答处理失败: {e}")
            return {"error": f"处理请求时出错: {str(e)}"}

    def ask_stream(self, query: str, **kwargs) -> Generator[Dict[str, Any], None, None]:
        """流式问答接口"""

        # 创建配置对象
        config = RunnableConfig(
            configurable={
                "thread_id": kwargs.get("thread_id", "0"),  # 会话ID，标记第几轮对话
            }
        )
        if self.langfuse_handler:
            config["callbacks"] = [self.langfuse_handler]
            self.update_langfuse_trace(**kwargs)

        # 初始化状态
        initial_state_dict = {
            "original_query": query,
            "messages": [],
        }
        initial_state_object = QAState(**initial_state_dict)

        try:
            for event in self.graph.stream(initial_state_object, config):
                if not event:
                    continue
                node_name = list(event.keys())[0]
                node_state = event[node_name]

                # node_state may be a QAState object or a plain dict depending on the node
                def _get(key, default=None):
                    if isinstance(node_state, dict):
                        return node_state.get(key, default)
                    return getattr(node_state, key, default)

                # 检查错误
                if _get("error"):
                    yield {
                        "node": node_name,
                        "status": "error",
                        "error": _get("error"),
                    }
                    continue

                # 流式返回节点执行状态
                yield {
                    "node": node_name,
                    "status": "processing",
                    "state": {
                        "kb_context": _get("kb_context"),
                        "tool_context": _get("tool_context"),
                        "final_context": _get("final_context"),
                    },
                }

                # 如果是生成答案节点，实现流式生成
                if node_name == "generate_answer":
                    messages = _get("messages") or []
                    if messages:
                        # 获取最后一条AI消息
                        ai_messages = [
                            msg.content
                            for msg in messages
                            if isinstance(msg, AIMessage)
                        ]
                        if ai_messages:
                            yield {
                                "node": "generate_answer",
                                "status": "complete",
                                "answer": ai_messages[-1],
                                "context": _get("final_context"),
                            }

        except Exception as e:
            self.logger.error(f"流式问答处理失败: {e}")
            yield {"status": "error", "error": f"处理请求时出错: {str(e)}"}

    def batch_ask(self, questions: List[str], **kwargs) -> List[Dict[str, Any]]:
        """批量问答"""
        return [self.ask(q, **kwargs) for q in questions]

    def export_graph(self, output_path: str = "qa_pipeline_graph.png") -> str | None:
        """导出图结构"""
        try:
            mermaid_code = self.graph.get_graph().draw_mermaid()

            with open(output_path.replace(".png", ".mmd"), "w", encoding="utf-8") as f:
                f.write(mermaid_code)

            self.logger.info(f"图结构已导出到：{output_path}")
            return mermaid_code

        except Exception as e:
            self.logger.error(f"导出图结构失败：{e}")
            return None
