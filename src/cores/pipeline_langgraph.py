# -*- coding: utf-8 -*-
"""
@Time：2025/6/8 10:06
@Auth：吕鑫
@File：pipeline_langgraph.py
@IDE：PyCharm

基于 LangGraph 的 QA Pipeline 重构
使用状态图来管理复杂的 RAG 工作流
纯 Async 架构版本
"""

import asyncio
import os
from functools import wraps
from typing import Annotated, Any, Callable, Dict, Generator, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from langfuse.decorators import langfuse_context
from langgraph.graph import END, START, StateGraph, add_messages

from src.cores.bounded_memory_saver import BoundedMemorySaver
from pydantic import BaseModel

from src.configs.config import AppConfig
from src.configs.logger_config import setup_logger
from src.configs.memory_settings import MemorySettings
from src.cores.message_builder import MessageBuilder
from src.cores.query_transformer import QueryTransformer
from src.db_services.milvus.connection_manager import MilvusConnectionManager
from src.mcp.mcp_client import MCPClient, _find_project_root
from src.mcp.server.kb_tools import KBTools
from src.memory.hybrid_memory_service import HybridMemoryService
from src.models.embedding import TextEmbedding
from src.models.llm import LLMWrapper
from src.skills.skill_manager import SkillManager

logger = setup_logger(__name__)


def timing_decorator(func: Callable) -> Callable:
    """性能监控装饰器 - 记录节点执行时间"""

    @wraps(func)
    async def wrapper(self, state: QAState) -> QAState:
        import time

        start = time.time()
        result = await func(self, state)
        duration = time.time() - start
        self.logger.info(f"{func.__name__} 耗时：{duration:.3f}s")
        return result

    return wrapper


class QAState(BaseModel):
    """QA Pipeline的状态定义"""

    messages: Annotated[List[Any], add_messages] = []
    original_query: str
    transformed_query: Optional[str] = None

    use_knowledge_base: Optional[bool] = None
    use_tools: Optional[bool] = None
    use_memory: Optional[bool] = None

    kb_context: Optional[str] = None
    tool_context: Optional[str] = None
    final_context: Optional[str] = None

    error: Optional[str] = None

    agent_iteration: int = 0
    tool_calls_history: List[Dict[str, Any]] = []

    cache_hit: Optional[bool] = None
    hyde_vector: Optional[List[float]] = None
    stream: bool = False


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
    """基于LangGraph的QA Pipeline - 纯 Async 架构"""

    def __init__(self, config: AppConfig):
        self.logger = logger
        self.config = config
        self._init_components()
        self._build_graph()

    def _init_components(self):
        """初始化所有组件"""
        self.logger.info("初始化QA Pipeline组件...")

        self.embeddings = TextEmbedding(self.config)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.splitter.chunk_size,
            chunk_overlap=self.config.splitter.chunk_overlap,
        )
        self.message_builder = MessageBuilder(self.config.message_builder)

        self.db_connection_manager = MilvusConnectionManager(
            self.embeddings,
            self.text_splitter,
            self.config.milvus,
            self.config.retrieve,
            self.config.reranker,
            self.config.bm25,
        )

        self.llm_caller = LLMWrapper(self.config.llm)

        self.query_transformer = QueryTransformer(
            self.llm_caller,
            self.message_builder,
            self.embeddings,
            self.db_connection_manager,
        )

        self._mcp_client: Optional[MCPClient] = None
        self._mcp_connected = False
        self._mcp_tools_cache: list = []

        self.skill_manager = SkillManager(self.config.prompt.skills_dir)

        self.kb_tools = KBTools(
            milvus_connection=self.db_connection_manager,
            embeddings=self.embeddings,
            search_config=self.config.retrieve,
        )

        self.checkpointer = BoundedMemorySaver(
            max_checkpoints=self.config.retrieve.max_checkpoints,
        )

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

        self._memory_settings = MemorySettings(
            stm_window_size=self.config.retrieve.memory_window_size,
            enable_ltm=getattr(self.config.memory, "enable_ltm", True)
            if hasattr(self.config, "memory")
            else True,
        )
        self._memory_service: Optional[HybridMemoryService] = None
        self._memory_initialized = False
        self._memory_init_lock = asyncio.Lock()
        self.logger.info("记忆服务配置完成，将在首次使用时初始化")

    async def _ensure_mcp_client(self) -> MCPClient:
        """确保 MCP 客户端已连接（懒加载，复用连接）- 纯 async"""
        if self._mcp_connected and self._mcp_client is not None:
            return self._mcp_client

        async with self._memory_init_lock:
            if self._mcp_connected and self._mcp_client is not None:
                return self._mcp_client

            if self._mcp_client is None:
                self._mcp_client = MCPClient(self.llm_caller)

            if not self._mcp_connected:
                project_root = _find_project_root()
                server_script = os.path.join(project_root, "mcp_server.py")
                await self._mcp_client.connect_to_server(server_script)
                self._mcp_connected = True
                self._mcp_tools_cache = (
                    self._mcp_client._convert_mcp_tools_to_openai_format()
                )
                self.logger.info("MCP 客户端连接已建立")

        return self._mcp_client

    async def _close_mcp_client(self) -> None:
        """关闭 MCP 客户端连接 - 纯 async"""
        if self._mcp_client is not None and self._mcp_connected:
            await self._mcp_client.cleanup()
            self._mcp_connected = False
            self._mcp_client = None
            self._mcp_tools_cache = []
            self.logger.info("MCP 客户端连接已关闭")

    def _build_graph(self):
        """构建LangGraph工作流"""
        workflow = StateGraph(QAState)

        workflow.add_node("agent_node", self._agent_node)
        workflow.add_node("tools_node", self._tools_node)
        workflow.add_node("build_context", self._build_context)
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("update_memory", self._update_memory)
        workflow.add_node("handle_error", self._handle_error)

        self._define_workflow(workflow)
        self.graph = workflow.compile(checkpointer=self.checkpointer)

    def _define_workflow(self, workflow: StateGraph):
        """定义工作流程"""
        workflow.add_edge(START, "agent_node")

        workflow.add_conditional_edges(
            "agent_node",
            self._should_continue_agent_loop,
            {
                "has_tool_calls": "tools_node",
                "done": "build_context",
                "error": "handle_error",
            },
        )
        workflow.add_edge("tools_node", "agent_node")

        workflow.add_conditional_edges(
            "build_context",
            self._should_generate_answer,
            {
                "generate_answer": "generate_answer",
                "skip_generate": "update_memory",
                "finish": END,
                "error": "handle_error",
            },
        )

        workflow.add_conditional_edges(
            "generate_answer",
            self._should_update_memory,
            {"update_memory": "update_memory", "finish": END, "error": "handle_error"},
        )

        workflow.add_edge("update_memory", END)
        workflow.add_edge("handle_error", END)


    @langfuse_context.observe(name="pipeline._agent_node")
    @timing_decorator
    async def _agent_node(self, state: QAState) -> QAState:
        """Agent 节点：LLM with tools，循环调用直到无 tool_calls"""
        try:
            self.logger.info(
                f"[AGENT_NODE] >>> iteration={state.agent_iteration}, "
                f"tool_calls_history_len={len(state.tool_calls_history)}, "
                f"messages_count={len(state.messages)}, "
                f"use_tool={self.config.retrieve.use_tool}"
            )

            for i, msg in enumerate(state.messages):
                self.logger.info(
                    f"[AGENT_NODE]   state.messages[{i}] {type(msg).__name__}: content={str(msg.content)[:80] if msg.content else ''}..., tool_calls={getattr(msg, 'tool_calls', None)}"
                )

            if self.config.retrieve.use_tool:
                openai_tools = await self._get_openai_tools()
                self.logger.info(
                    f"[AGENT_NODE]   openai_tools count: {len(openai_tools)}"
                )
            else:
                openai_tools = []

            if state.agent_iteration == 0:
                system_prompt = (
                    self.config.prompt.kb_system_prompt
                    if self.config.retrieve.use_kb
                    else self.config.prompt.system_prompt
                )
                skills_block = self.skill_manager.get_skills_prompt_block()
                if skills_block:
                    system_prompt = system_prompt + "\n\n" + skills_block
                messages = build_prompt_messages(
                    state,
                    system_prompt=system_prompt,
                    use_memory=self.config.retrieve.use_memory,
                    memory_limit=self.config.retrieve.memory_window_size,
                )
                self.logger.info(
                    f"[AGENT_NODE]   built new messages from scratch, count={len(messages)}"
                )
            else:
                messages = list(state.messages)
                self.logger.info(
                    f"[AGENT_NODE]   reusing state.messages, count={len(messages)}"
                )
                for i, msg in enumerate(messages):
                    tc = getattr(msg, "tool_calls", None)
                    self.logger.info(
                        f"[AGENT_NODE]   msg[{i}] {type(msg).__name__}: content={str(msg.content)[:100] if msg.content else ''}..., tool_calls={tc}, tc_type={type(tc)}"
                    )
                    if tc:
                        for j, t in enumerate(tc):
                            self.logger.info(
                                f"[AGENT_NODE]     tc[{j}]: type={type(t)}, value={t}"
                            )

            extra_body = self.config.retrieve.extra_body
            call_kwargs = dict(
                messages=messages,
                return_raw=True,
                extra_body=extra_body,
                stream=state.stream,
            )
            if openai_tools:
                call_kwargs["tools"] = openai_tools
                call_kwargs["tool_choice"] = "auto"

            self.logger.info(
                f"[AGENT_NODE]   calling LLM with {len(messages)} messages, tools={bool(openai_tools)}, stream={state.stream}"
            )
            response = await self.llm_caller.achat(**call_kwargs)
            state.cache_hit = self.llm_caller._last_cache_hit

            if state.stream:
                # 流式模式：迭代 chunks 收集完整响应
                content_chunks = []
                tool_calls_chunks = []
                is_first_chunk = True
                async for chunk in response:
                    if hasattr(chunk, "choices") and chunk.choices:
                        delta = chunk.choices[0].delta
                        if delta:
                            if hasattr(delta, "content") and delta.content:
                                content_chunks.append(delta.content)
                            if hasattr(delta, "tool_calls") and delta.tool_calls:
                                tool_calls_chunks.append(delta.tool_calls)
                        is_first_chunk = False
                full_content = "".join(content_chunks)
                raw_message = type("RawMessage", (), {
                    "content": full_content,
                    "tool_calls": tool_calls_chunks[0] if tool_calls_chunks else None
                })()
            else:
                # 非流式模式
                raw_message = response.choices[0].message
            self.logger.info(
                f"[AGENT_NODE]   LLM response: content={str(raw_message.content)[:200] if raw_message.content else 'None'}, tool_calls={raw_message.tool_calls}"
            )

            if openai_tools and raw_message.tool_calls:
                lc_tool_calls = []
                self.logger.info(
                    f"[AGENT_NODE]   raw tool_calls count: {len(raw_message.tool_calls)}"
                )
                for tc in raw_message.tool_calls:
                    tc_type = type(tc).__name__
                    tc_id = getattr(tc, "id", "N/A")
                    tc_func = getattr(tc, "function", None)
                    func_name = getattr(tc_func, "name", "N/A") if tc_func else "N/A"
                    func_args_raw = (
                        getattr(tc_func, "arguments", None) if tc_func else "N/A"
                    )
                    self.logger.info(
                        f"[AGENT_NODE]   tc: type={tc_type}, id={tc_id}, func_name={func_name}, func_args_raw type={type(func_args_raw).__name__}, func_args_raw={func_args_raw}"
                    )
                    self.logger.info(
                        f"[AGENT_NODE]   tc: type={tc_type}, id={tc_id}, function.name={func_name}, function.arguments={func_args_raw}"
                    )
                    try:
                        if func_args_raw is None:
                            args = {}
                        elif isinstance(func_args_raw, dict):
                            args = func_args_raw
                        else:
                            args = __import__("json").loads(func_args_raw)
                        if not isinstance(args, dict):
                            args = {"raw": str(args)}
                    except Exception as e:
                        self.logger.warning(
                            f"[AGENT_NODE]   json parse error: {e}, raw_args: {func_args_raw}"
                        )
                        args = {"raw": str(func_args_raw)}
                    lc_tool_calls.append(
                        {
                            "id": tc.id,
                            "name": tc.function.name,
                            "args": args,
                            "type": "tool_call",
                        }
                    )
                    self.logger.info(
                        f"[AGENT_NODE]   appended tc: name={tc.function.name}, args type={type(args)}"
                    )
                ai_msg = AIMessage(
                    content=raw_message.content or "",
                    tool_calls=lc_tool_calls,
                )
                state.messages += [ai_msg]
                state.agent_iteration += 1
                self.logger.info(
                    f"[AGENT_NODE]   <<< HAS TOOL_CALLS: {[tc['name'] for tc in lc_tool_calls]}, "
                    f"args={[tc['args'] for tc in lc_tool_calls]}"
                )
            else:
                answer_content = raw_message.content or ""
                ai_msg = AIMessage(content=answer_content)
                state.messages += [ai_msg]
                state.agent_iteration += 1
                self.logger.info(
                    f"[AGENT_NODE]   <<< NO TOOL_CALLS, answer={str(answer_content)[:200]}..."
                )
                if state.tool_calls_history:
                    tool_results = "\n".join(
                        f"[{r['tool']}]: {r['result']}"
                        for r in state.tool_calls_history
                    )
                    state.tool_context = f"【工具返回内容】\n{tool_results}"
                    self.logger.info(
                        f"[AGENT_NODE]   <<< SET tool_context from tool_calls_history, len={len(tool_results)}"
                    )
                else:
                    self.logger.info(
                        "[AGENT_NODE]   <<< NO tool_calls_history, tool_context NOT set"
                    )
                self.logger.info("[AGENT_NODE]   <<< Agent生成最终答案")

            return state

        except Exception as e:
            self.logger.exception(f"Agent节点执行失败：{e}")
            state.error = f"Agent节点执行失败：{str(e)}"
            return state
        finally:
            # 更新 span metadata
            if hasattr(langfuse_context, "update_current_span"):
                langfuse_context.update_current_span(
                    metadata={
                        "agent_iteration": state.agent_iteration,
                        "tool_calls_count": len(getattr(state, "tool_calls_history", [])),
                        "cache_hit": getattr(state, "cache_hit", False),
                        "use_tools": self.config.retrieve.use_tool,
                    }
                )

    async def _get_openai_tools(self) -> list:
        """获取 OpenAI 格式的工具列表（复用 MCP 连接）"""
        if self._mcp_tools_cache:
            return self._mcp_tools_cache

        try:
            client = await self._ensure_mcp_client()
            self._mcp_tools_cache = client._convert_mcp_tools_to_openai_format()
            return self._mcp_tools_cache
        except Exception as e:
            self.logger.warning(f"获取 MCP 工具列表失败: {e}")
            return []

    @langfuse_context.observe(name="pipeline._tools_node")
    @timing_decorator
    async def _tools_node(self, state: QAState) -> QAState:
        """工具节点：执行 AIMessage 中的 tool_calls - 纯 async"""
        try:
            self.logger.info(
                f"[TOOLS_NODE] >>> messages_count={len(state.messages)}, tool_calls_history_len={len(state.tool_calls_history)}"
            )

            last_ai = next(
                (
                    m
                    for m in reversed(state.messages)
                    if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
                ),
                None,
            )
            if not last_ai:
                self.logger.warning("[TOOLS_NODE] 未找到待执行的 tool_calls")
                return state

            tool_calls = last_ai.tool_calls
            if not isinstance(tool_calls, list):
                self.logger.error(
                    f"[TOOLS_NODE] tool_calls is not a list! type={type(tool_calls)}, value={tool_calls}"
                )
                state.error = (
                    f"tool_calls format error: expected list, got {type(tool_calls)}"
                )
                return state
            self.logger.info(
                f"[TOOLS_NODE] 执行 {len(tool_calls)} 个工具调用: {[tc['name'] for tc in tool_calls]}"
            )
            for tc in tool_calls:
                self.logger.info(
                    f"[TOOLS_NODE]   tool_call: {tc['name']}, id={tc['id']}, args={tc['args']}"
                )

            kb_tool_calls = []
            rewrite_tool_calls = []
            mcp_tool_calls = []
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    self.logger.error(
                        f"[TOOLS_NODE] tool_call is not a dict: type={type(tc)}, value={tc}"
                    )
                    continue
                if tc.get("name") == "kb_search":
                    kb_tool_calls.append(tc)
                elif tc.get("name") == "query_rewrite":
                    rewrite_tool_calls.append(tc)
                else:
                    mcp_tool_calls.append(tc)
            self.logger.info(
                f"[TOOLS_NODE]   kb_tool_calls={len(kb_tool_calls)}, rewrite_tool_calls={len(rewrite_tool_calls)}, mcp_tool_calls={len(mcp_tool_calls)}"
            )

            results_map: Dict[str, str] = {}

            # Handle query_rewrite tool directly (like kb_search)
            if rewrite_tool_calls:
                self.logger.info(
                    f"[TOOLS_NODE]   准备执行 query_rewrite，共 {len(rewrite_tool_calls)} 个"
                )
                for tc in rewrite_tool_calls:
                    args = tc.get("args", {}) if isinstance(tc.get("args"), dict) else {}
                    query = args.get("query", "") if isinstance(args, dict) else ""
                    mode = args.get("mode", "rewrite") if isinstance(args, dict) else "rewrite"
                    try:
                        rewritten = self.query_transformer.transform_query(query, mode=mode)
                        results_map[tc["id"]] = f"改写后查询：{rewritten}"
                        self.logger.info(
                            f"[TOOLS_NODE]   query_rewrite [{query}] ({mode}) => {rewritten[:50]}..."
                        )
                    except Exception as e:
                        results_map[tc["id"]] = f"查询改写失败: {str(e)}"
                        self.logger.warning(f"[TOOLS_NODE]   query_rewrite error: {e}")

            if kb_tool_calls:
                self.logger.info(
                    f"[TOOLS_NODE]   准备执行 KB search，共 {len(kb_tool_calls)} 个"
                )
                kb_tasks = []
                for tc in kb_tool_calls:
                    args = (
                        tc.get("args", {}) if isinstance(tc.get("args"), dict) else {}
                    )
                    query = args.get("query", "") if isinstance(args, dict) else ""
                    top_k = args.get("top_k", 3) if isinstance(args, dict) else 3
                    use_hyde = args.get("use_hyde", False) if isinstance(args, dict) else False
                    # Generate HyDE vector if use_hyde=True and not already provided
                    hyde_vector = getattr(state, "hyde_vector", None)
                    if use_hyde and not hyde_vector and self.query_transformer:
                        try:
                            hyde_vector = await self.query_transformer.generate_hyde_vector(
                                query, num_hypo=self.config.retrieve.num_hypo
                            )
                            self.logger.info("HyDE vector generated in tools_node")
                        except Exception as e:
                            self.logger.warning(f"HyDE vector generation failed: {e}")
                    kb_tasks.append(self.kb_tools.kb_search(
                        query=query,
                        top_k=top_k,
                        hyde_vector=hyde_vector,
                    ))
                kb_results = await asyncio.gather(*kb_tasks)
                for tc, result in zip(kb_tool_calls, kb_results):
                    results_map[tc["id"]] = result
                    args = (
                        tc.get("args", {}) if isinstance(tc.get("args"), dict) else {}
                    )
                    query = (
                        args.get("query", "") if isinstance(args, dict) else str(args)
                    )
                    self.logger.info(
                        f"[TOOLS_NODE]   KB [{query}] => {result[:100] if result else '(空)'}..."
                    )

            if mcp_tool_calls:
                self.logger.info(
                    f"[TOOLS_NODE]   准备执行 MCP 工具，共 {len(mcp_tool_calls)} 个"
                )
                client = await self._ensure_mcp_client()
                mcp_results = await client.execute_tool_calls(mcp_tool_calls)
                for tc, result in zip(mcp_tool_calls, mcp_results):
                    results_map[tc["id"]] = result
                    self.logger.info(
                        f"[TOOLS_NODE]   MCP [{tc['name']}] => {result[:100] if result else '(空)'}..."
                    )

            ordered_results = [results_map[tc["id"]] for tc in tool_calls]

            tool_messages = [
                ToolMessage(content=result, tool_call_id=tc["id"])
                for tc, result in zip(tool_calls, ordered_results)
            ]
            state.messages += tool_messages
            self.logger.info(
                f"[TOOLS_NODE]   添加 {len(tool_messages)} 个 ToolMessage 到 messages"
            )

            history = list(state.tool_calls_history)
            for tc, result in zip(tool_calls, ordered_results):
                history.append(
                    {"tool": tc["name"], "args": tc["args"], "result": result}
                )
            # Enforce max_tool_calls_history limit with FIFO eviction
            max_history = self.config.retrieve.max_tool_calls_history
            if len(history) > max_history:
                evicted = len(history) - max_history
                history = history[-max_history:]
                self.logger.debug(
                    f"[TOOLS_NODE]   tool_calls_history truncated, evicted {evicted} oldest entries, "
                    f"remaining {len(history)}"
                )
            state.tool_calls_history = history
            self.logger.info(
                f"[TOOLS_NODE]   tool_calls_history 更新后长度: {len(history)}"
            )

            if history:
                tool_results = "\n".join(
                    f"[{r['tool']}]: {r['result']}" for r in history
                )
                state.tool_context = f"【工具返回内容】\n{tool_results}"
                self.logger.info(
                    f"[TOOLS_NODE]   <<< SET tool_context, len={len(tool_results)}"
                )
            else:
                self.logger.info("[TOOLS_NODE]   <<< NO history, tool_context NOT set")

            return state

        except Exception as e:
            self.logger.exception(f"工具节点执行失败：{e}")
            state.error = f"工具节点执行失败：{str(e)}"
            return state
        finally:
            if hasattr(langfuse_context, "update_current_span"):
                tool_history = getattr(state, "tool_calls_history", []) or []
                langfuse_context.update_current_span(
                    metadata={
                        "tool_calls_count": len(tool_history),
                        "tools_executed": [t["tool"] for t in tool_history] if tool_history else [],
                    }
                )

    @langfuse_context.observe(name="pipeline._generate_answer")
    @timing_decorator
    async def _generate_answer(self, state: QAState) -> QAState:
        """生成答案（use_tool=False 时的纯 LLM 生成路径）"""
        try:
            self.logger.info("开始生成答案")

            system_prompt = (
                self.config.prompt.kb_system_prompt
                if self.config.retrieve.use_kb
                else self.config.prompt.system_prompt
            )

            messages = build_prompt_messages(
                state,
                system_prompt=system_prompt,
                use_memory=self.config.retrieve.use_memory,
                memory_limit=self.config.retrieve.memory_window_size,
            )

            response = await self.llm_caller.achat(messages=messages, stream=state.stream)
            state.cache_hit = self.llm_caller._last_cache_hit

            if state.stream:
                # 流式模式：迭代 chunks 收集完整响应
                content_chunks = []
                async for chunk in response:
                    if hasattr(chunk, "choices") and chunk.choices:
                        delta = chunk.choices[0].delta
                        if delta and hasattr(delta, "content") and delta.content:
                            content_chunks.append(delta.content)
                answer = "".join(content_chunks)
            else:
                answer = response

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
        finally:
            if hasattr(langfuse_context, "update_current_span"):
                langfuse_context.update_current_span(
                    metadata={"cache_hit": getattr(state, "cache_hit", False)}
                )

    async def _ensure_memory_initialized(self) -> None:
        """确保记忆服务已初始化（async，线程安全）"""
        if self._memory_initialized:
            return

        async with self._memory_init_lock:
            if self._memory_initialized:
                return
            await self._init_memory()

    async def _init_memory(self) -> None:
        """实际初始化记忆服务的逻辑"""
        try:
            if self._memory_service is None:
                user_id = self.config.retrieve.user_id or "default"
                self._memory_service = HybridMemoryService(
                    config=self.config,
                    user_id=user_id,
                    stm_window_size=self._memory_settings.stm_window_size,
                    ltm_persist_threshold=self._memory_settings.ltm_persist_threshold,
                    enable_stm=self._memory_settings.enable_stm,
                    enable_ltm=self._memory_settings.enable_ltm,
                )

            await self._memory_service.initialize()
            self._memory_initialized = True
            self.logger.info("记忆服务初始化完成")
        except Exception as e:
            self.logger.warning(f"记忆服务初始化失败: {e}，将使用简单记忆模式")

    @langfuse_context.observe(name="pipeline._build_context")
    @timing_decorator
    async def _build_context(self, state: QAState) -> QAState:
        """构建最终上下文 - 纯 async"""
        try:
            self.logger.info(
                f"[BUILD_CONTEXT] >>> agent_iteration={state.agent_iteration}, tool_context={state.tool_context[:100] if state.tool_context else None}..., kb_context={state.kb_context[:100] if state.kb_context else None}..."
            )

            context_parts = []

            if state.tool_context:
                context_parts.append(state.tool_context)
                self.logger.info(
                    f"[BUILD_CONTEXT]   added tool_context, len={len(state.tool_context)}"
                )

            if state.kb_context:
                context_parts.append(state.kb_context)
                self.logger.info(
                    f"[BUILD_CONTEXT]   added kb_context, len={len(state.kb_context)}"
                )
            elif self.config.retrieve.use_kb:
                query = state.transformed_query or state.original_query
                self.logger.info(
                    f"[BUILD_CONTEXT]   use_kb=True, kb_context is empty, searching kb with query={query}"
                )
                try:
                    kb_results = await self.kb_tools.kb_search(
                        query,
                        top_k=self.config.retrieve.top_k,
                        hyde_vector=getattr(state, "hyde_vector", None),
                    )
                    if kb_results:
                        state.kb_context = kb_results
                        context_parts.append(kb_results)
                        self.logger.info(
                            f"[BUILD_CONTEXT]   kb search returned {len(kb_results)} chars"
                        )
                except Exception as e:
                    self.logger.warning(f"[BUILD_CONTEXT]   kb search failed: {e}")

            if self.config.retrieve.use_memory and self._memory_settings.enable_ltm:
                await self._ensure_memory_initialized()
                if self._memory_service and self._memory_service.is_initialized:
                    try:
                        ltm_results = await self._memory_service.search(
                            query=state.original_query,
                            user_id=self.config.retrieve.user_id or "default",
                            limit=self._memory_settings.ltm_search_limit,
                            threshold=self._memory_settings.ltm_search_threshold,
                        )
                        if ltm_results:
                            ltm_context = self._memory_service.format_ltm_for_context(
                                ltm_results, prefix="【长期记忆】"
                            )
                            context_parts.append(ltm_context)
                            self.logger.info(
                                f"添加 {len(ltm_results)} 条长期记忆到上下文"
                            )
                    except Exception as e:
                        self.logger.warning(f"搜索长期记忆失败: {e}")

            state.final_context = "\n\n".join(context_parts)
            self.logger.info(f"上下文构建完成，长度：{len(state.final_context)}")

            return state

        except Exception as e:
            self.logger.exception(f"构建上下文失败：{e}")
            state.error = f"构建上下文失败：{str(e)}"
            return state
        finally:
            if hasattr(langfuse_context, "update_current_span"):
                langfuse_context.update_current_span(
                    metadata={
                        "has_kb_context": bool(getattr(state, "kb_context", None)),
                        "kb_context_length": len(getattr(state, "kb_context", "") or ""),
                        "has_memory_context": bool(getattr(state, "tool_context", None)),
                    }
                )

    @timing_decorator
    async def _update_memory(self, state: QAState) -> QAState:
        """更新记忆 - 纯 async"""
        try:
            if self.config.retrieve.use_memory and state.messages:
                await self._ensure_memory_initialized()
                if self._memory_service and self._memory_service.is_initialized:
                    try:
                        user_id = self.config.retrieve.user_id or "default"
                        stm_messages = [
                            {
                                "role": "user"
                                if isinstance(m, HumanMessage)
                                else "assistant",
                                "content": m.content,
                            }
                            for m in state.messages
                            if isinstance(m, (HumanMessage, AIMessage))
                            and getattr(m, "content", None)
                        ]

                        await self._memory_service.add(
                            messages=stm_messages,
                            user_id=user_id,
                            persist_immediately=True,
                        )
                        self.logger.info("记忆已存入混合记忆服务")
                    except Exception as e:
                        self.logger.warning(f"存入记忆失败: {e}")

                max_history = self.config.retrieve.memory_window_size
                state.messages = state.messages[-max_history:]
                self.logger.info(f"记忆更新完成，当前历史消息数：{len(state.messages)}")
            return state
        except Exception as e:
            self.logger.exception(f"更新记忆失败：{e}")
            state.error = f"更新记忆失败：{str(e)}"
            return state

    @timing_decorator
    async def _handle_error(self, state: QAState) -> QAState:
        """处理错误"""
        error_msg = state.error or "未知错误"
        self.logger.error(f"处理错误: {error_msg}")

        if not hasattr(state, "messages") or state.messages is None:
            state.messages = []

        state.messages.append(
            AIMessage(content=f"抱歉，处理您的请求时出现错误：{error_msg}")
        )

        return state

    def _should_continue_agent_loop(self, state: QAState) -> str:
        """判断 agent loop 是否继续"""
        self.logger.info(
            f"[LOOP_CHECK] >>> agent_iteration={state.agent_iteration}, max={self.config.retrieve.max_agent_iterations}, error={state.error}"
        )

        if state.error:
            self.logger.info("[LOOP_CHECK] <<< returning 'error'")
            return "error"

        if state.agent_iteration >= self.config.retrieve.max_agent_iterations:
            self.logger.warning(
                "[LOOP_CHECK] <<< MAX ITERATIONS REACHED, returning 'done'"
            )
            return "done"

        last_ai = None
        for m in reversed(state.messages):
            self.logger.info(f"[LOOP_CHECK] checking msg type={type(m).__name__}")
            if isinstance(m, AIMessage):
                last_ai = m
                break
        if last_ai is None:
            self.logger.info("[LOOP_CHECK] No AIMessage found")
            return "done"
        has_tc = getattr(last_ai, "tool_calls", None)
        self.logger.info(
            f"[LOOP_CHECK] last_ai type={type(last_ai).__name__}, has_tc type={type(has_tc).__name__}, has_tc={has_tc}"
        )
        if last_ai and has_tc and isinstance(has_tc, list) and len(has_tc) > 0:
            self.logger.info(
                "[LOOP_CHECK] <<< HAS TOOL_CALLS, returning 'has_tool_calls'"
            )
            return "has_tool_calls"
        self.logger.info("[LOOP_CHECK] <<< NO TOOL_CALLS, returning 'done'")
        return "done"

    def _should_generate_answer(self, state: QAState) -> str:
        """判断 build_context 后是否需要 LLM 生成答案"""
        self.logger.info(
            f"[GEN_ANSWER_CHECK] >>> agent_iteration={state.agent_iteration}, error={state.error}, use_memory={self.config.retrieve.use_memory}"
        )
        if state.error:
            self.logger.info("[GEN_ANSWER_CHECK] <<< returning 'error'")
            return "error"

        if state.agent_iteration > 0:
            if self.config.retrieve.use_memory:
                self.logger.info("[GEN_ANSWER_CHECK] <<< returning 'skip_generate'")
                return "skip_generate"
            self.logger.info("[GEN_ANSWER_CHECK] <<< returning 'finish'")
            return "finish"

        self.logger.info("[GEN_ANSWER_CHECK] <<< returning 'generate_answer'")
        return "generate_answer"

    def _should_update_memory(self, state: QAState) -> str:
        """判断是否需要更新记忆"""
        if state.error:
            return "error"

        if self.config.retrieve.use_memory:
            return "update_memory"
        else:
            return "finish"

    def update_langfuse_trace(self, **kwargs):
        """根据传入参数更新 Langfuse trace 信息"""
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

    async def ask(self, query: str, **kwargs) -> Dict[str, Any]:
        """异步问答接口"""
        config = RunnableConfig(
            configurable={
                "thread_id": kwargs.get("thread_id", "0"),
            }
        )
        if self.langfuse_handler:
            config["callbacks"] = [self.langfuse_handler]
            self.update_langfuse_trace(**kwargs)

        existing_messages = kwargs.get("messages", [])

        initial_state_object = QAState(
            original_query=query,
            messages=existing_messages,
            use_knowledge_base=self.config.retrieve.use_kb,
            use_memory=self.config.retrieve.use_memory,
        )

        try:
            result = await self.graph.ainvoke(initial_state_object, config)

            if result.get("error"):
                return {"error": result["error"]}

            ai_messages = [
                msg.content
                for msg in result.get("messages", [])
                if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None)
            ]

            answer = ai_messages[-1] if ai_messages else "无法生成答案"

            ret = {
                "question": query,
                "answer": answer,
                "messages": result.get("messages", existing_messages),
                "transformed_query": result.get("transformed_query", ""),
                "context": result.get("final_context", ""),
                "kb_context": result.get("kb_context", ""),
                "tool_context": result.get("tool_context", ""),
                "tool_calls_history": result.get("tool_calls_history", []),
                "agent_iterations": result.get("agent_iteration", 0),
                "cache_hit": result.get("cache_hit", False),
            }

            # trace_url 仅在 debug=True 且 Langfuse 启用时返回
            if kwargs.get("debug") and self.langfuse_client:
                if hasattr(langfuse_context, "get_trace_url"):
                    ret["trace_url"] = langfuse_context.get_trace_url()
                elif hasattr(langfuse_context, "get_trace_id"):
                    trace_id = langfuse_context.get_trace_id()
                    if trace_id and self.langfuse_client.host:
                        ret["trace_url"] = f"{self.langfuse_client.host}/project/1/traces/{trace_id}"

            return ret

        except Exception as e:
            self.logger.error(f"问答处理失败: {e}")
            return {"error": f"处理请求时出错: {str(e)}"}

    def ask_sync(self, query: str, **kwargs) -> Dict[str, Any]:
        """同步问答接口（内部使用 asyncio.run）"""
        return asyncio.run(self.ask(query, **kwargs))

    async def ask_stream(
        self, query: str, **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """流式问答接口 - 纯 async，支持 token 级流式输出"""

        config = RunnableConfig(
            configurable={
                "thread_id": kwargs.get("thread_id", "0"),
            }
        )
        if self.langfuse_handler:
            config["callbacks"] = [self.langfuse_handler]
            self.update_langfuse_trace(**kwargs)

        initial_state_object = QAState(
            original_query=query,
            messages=[],
            stream=True,
            use_knowledge_base=self.config.retrieve.use_kb,
            use_memory=self.config.retrieve.use_memory,
        )

        current_answer = []
        current_node = None

        try:
            async for event in self.graph.astream_events(initial_state_object, config):
                if not event:
                    continue

                event_type = event.get("event")
                self.logger.debug(f"[STREAM] event_type={event_type}")

                # Token 级流式：LLM 返回的每个 token chunk
                if event_type == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "delta") and chunk.delta:
                        delta = chunk.delta
                        if hasattr(delta, "content") and delta.content:
                            token = delta.content
                            current_answer.append(token)
                            yield {
                                "node": current_node or "agent_node",
                                "status": "chunk",
                                "content": token,
                            }

                # 节点开始
                elif event_type == "on_chain_start":
                    node_name = event.get("name")
                    if node_name:
                        current_node = node_name
                        self.logger.info(f"[STREAM] >>> node start: {node_name}")

                # 节点结束
                elif event_type == "on_chain_end":
                    node_name = event.get("name")
                    if node_name in ("agent_node", "generate_answer") and current_answer:
                        full_answer = "".join(current_answer)
                        self.logger.info(f"[STREAM] <<< node end: {node_name}, answer_len={len(full_answer)}")
                        complete_event = {
                            "node": node_name,
                            "status": "complete",
                            "answer": full_answer,
                            "context": "",
                        }
                        if kwargs.get("debug") and self.langfuse_client:
                            if hasattr(langfuse_context, "get_trace_url"):
                                complete_event["trace_url"] = langfuse_context.get_trace_url()
                            elif hasattr(langfuse_context, "get_trace_id"):
                                trace_id = langfuse_context.get_trace_id()
                                if trace_id and self.langfuse_client.host:
                                    complete_event["trace_url"] = f"{self.langfuse_client.host}/project/1/traces/{trace_id}"
                        yield complete_event
                        current_answer = []

                # 错误处理
                elif event_type == "on_chain_error" or event_type == "on_error":
                    self.logger.error(f"[STREAM] error: {event}")
                    yield {
                        "node": current_node or "unknown",
                        "status": "error",
                        "error": str(event.get("error", "Unknown error")),
                    }

        except Exception as e:
            self.logger.exception(f"流式问答处理失败: {e}")
            yield {"status": "error", "error": f"处理请求时出错: {str(e)}"}
        finally:
            # Ensure streaming buffer is cleaned up on any exit path
            current_answer.clear()

    async def batch_ask(self, questions: List[str], **kwargs) -> List[Dict[str, Any]]:
        """批量问答 - 纯 async"""
        tasks = [self.ask(q, **kwargs) for q in questions]
        return await asyncio.gather(*tasks)

    def batch_ask_sync(self, questions: List[str], **kwargs) -> List[Dict[str, Any]]:
        """批量问答同步包装器"""
        return asyncio.run(self.batch_ask(questions, **kwargs))

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
