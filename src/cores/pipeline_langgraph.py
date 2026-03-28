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
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
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

        self.checkpointer = MemorySaver()

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

        workflow.add_node("parse_query", self._parse_query)
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
        workflow.add_edge(START, "parse_query")

        workflow.add_conditional_edges(
            "parse_query",
            self._should_call_tools,
            {
                "call_agent": "agent_node",
                "skip_tools": "build_context",
                "error": "handle_error",
            },
        )

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

    @timing_decorator
    async def _parse_query(self, state: QAState) -> QAState:
        """解析查询，设置默认参数"""
        try:
            self.logger.info(f"解析查询：{state.original_query}")
            if state.use_knowledge_base is None:
                state.use_knowledge_base = self.config.retrieve.use_kb
            if state.use_tools is None:
                state.use_tools = self.config.retrieve.use_tool
            if state.use_memory is None:
                state.use_memory = self.config.retrieve.use_memory
            state.error = None
            return state
        except Exception as e:
            self.logger.exception(f"解析查询失败：{e}")
            state.error = f"解析查询失败：{str(e)}"
            return state

    @timing_decorator
    async def _agent_node(self, state: QAState) -> QAState:
        """Agent 节点：LLM with tools，循环调用直到无 tool_calls"""
        try:
            self.logger.info(
                f"Agent节点迭代 {state.agent_iteration}，"
                f"当前 tool_calls_history 长度: {len(state.tool_calls_history)}，"
                f"messages 数量: {len(state.messages)}"
            )

            if self.config.retrieve.use_tool:
                openai_tools = await self._get_openai_tools()
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
            else:
                messages = list(state.messages)

            extra_body = self.config.retrieve.extra_body
            call_kwargs = dict(
                messages=messages,
                return_raw=True,
                extra_body=extra_body,
            )
            if openai_tools:
                call_kwargs["tools"] = openai_tools
                call_kwargs["tool_choice"] = "auto"

            response = self.llm_caller.chat(**call_kwargs)
            raw_message = response.choices[0].message

            if openai_tools and raw_message.tool_calls:
                lc_tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "args": __import__("json").loads(tc.function.arguments),
                        "type": "tool_call",
                    }
                    for tc in raw_message.tool_calls
                ]
                ai_msg = AIMessage(
                    content=raw_message.content or "",
                    tool_calls=lc_tool_calls,
                )
                state.messages += [ai_msg]
                state.agent_iteration += 1
                self.logger.info(
                    f"Agent决定调用工具: {[tc['name'] for tc in lc_tool_calls]}"
                )
            else:
                answer_content = raw_message.content or ""
                ai_msg = AIMessage(content=answer_content)
                state.messages += [ai_msg]
                state.agent_iteration += 1
                if state.tool_calls_history:
                    tool_results = "\n".join(
                        f"[{r['tool']}]: {r['result']}"
                        for r in state.tool_calls_history
                    )
                    state.tool_context = f"【工具返回内容】\n{tool_results}"
                self.logger.info("Agent生成最终答案")

            return state

        except Exception as e:
            self.logger.exception(f"Agent节点执行失败：{e}")
            state.error = f"Agent节点执行失败：{str(e)}"
            return state

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

    @timing_decorator
    async def _tools_node(self, state: QAState) -> QAState:
        """工具节点：执行 AIMessage 中的 tool_calls - 纯 async"""
        try:
            last_ai = next(
                (
                    m
                    for m in reversed(state.messages)
                    if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
                ),
                None,
            )
            if not last_ai:
                self.logger.warning("tools_node: 未找到待执行的 tool_calls")
                return state

            tool_calls = last_ai.tool_calls
            self.logger.info(f"执行 {len(tool_calls)} 个工具调用")

            kb_tool_calls = [tc for tc in tool_calls if tc["name"] == "kb_search"]
            mcp_tool_calls = [tc for tc in tool_calls if tc["name"] != "kb_search"]

            results_map: Dict[str, str] = {}

            if kb_tool_calls:
                kb_tasks = [
                    self.kb_tools.kb_search(
                        query=tc["args"].get("query", ""),
                        top_k=tc["args"].get("top_k", 3),
                    )
                    for tc in kb_tool_calls
                ]
                kb_results = await asyncio.gather(*kb_tasks)
                for tc, result in zip(kb_tool_calls, kb_results):
                    results_map[tc["id"]] = result
                    self.logger.info(
                        f"KB search [{tc['args'].get('query', '')}] 返回: {result[:200] if result else '(空)'}"
                    )

            if mcp_tool_calls:
                client = await self._ensure_mcp_client()
                mcp_results = await client.execute_tool_calls(mcp_tool_calls)
                for tc, result in zip(mcp_tool_calls, mcp_results):
                    results_map[tc["id"]] = result

            ordered_results = [results_map[tc["id"]] for tc in tool_calls]

            tool_messages = [
                ToolMessage(content=result, tool_call_id=tc["id"])
                for tc, result in zip(tool_calls, ordered_results)
            ]
            state.messages += tool_messages

            history = list(state.tool_calls_history)
            for tc, result in zip(tool_calls, ordered_results):
                history.append(
                    {"tool": tc["name"], "args": tc["args"], "result": result}
                )
                self.logger.info(
                    f"工具 [{tc['name']}] 结果: {result[:200] if result else '(空)'}"
                )
            state.tool_calls_history = history

            self.logger.info(
                f"工具调用完成，共 {len(results_map)} 条结果，tool_calls_history 长度: {len(history)}"
            )
            return state

        except Exception as e:
            self.logger.exception(f"工具节点执行失败：{e}")
            state.error = f"工具节点执行失败：{str(e)}"
            return state

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

            answer = self.llm_caller.chat(messages=messages)

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

    @timing_decorator
    async def _build_context(self, state: QAState) -> QAState:
        """构建最终上下文 - 纯 async"""
        try:
            self.logger.info("构建最终上下文")

            context_parts = []

            if state.tool_context:
                context_parts.append(state.tool_context)

            if state.kb_context:
                context_parts.append(state.kb_context)
            elif state.agent_iteration == 0 and self.config.retrieve.use_kb:
                query = state.transformed_query or state.original_query
                try:
                    kb_results = await self.kb_tools.kb_search(
                        query, top_k=self.config.retrieve.top_k
                    )
                    if kb_results:
                        state.kb_context = kb_results
                        context_parts.append(kb_results)
                        self.logger.info(
                            f"主动检索知识库返回: {kb_results[:200] if kb_results else '(空)'}..."
                        )
                except Exception as e:
                    self.logger.warning(f"知识库检索失败: {e}")

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

    def _should_call_tools(self, state: QAState) -> str:
        """判断是否需要调用工具"""
        if state.error:
            return "error"

        if self.config.retrieve.use_tool:
            return "call_agent"
        elif self.config.retrieve.use_kb:
            return "skip_tools"
        else:
            return "skip_tools"

    def _should_continue_agent_loop(self, state: QAState) -> str:
        """判断 agent loop 是否继续"""
        if state.error:
            return "error"

        if state.agent_iteration >= self.config.retrieve.max_agent_iterations:
            self.logger.warning(
                f"Agent达到最大迭代次数 {self.config.retrieve.max_agent_iterations}，强制结束"
            )
            return "done"

        last_ai = next(
            (m for m in reversed(state.messages) if isinstance(m, AIMessage)),
            None,
        )
        self.logger.info(
            f"_should_continue_agent_loop: messages={len(state.messages)}, "
            f"last_ai={type(last_ai).__name__ if last_ai else None}, "
            f"last_ai.tool_calls={getattr(last_ai, 'tool_calls', None)}"
        )
        if last_ai and getattr(last_ai, "tool_calls", None):
            return "has_tool_calls"
        return "done"

    def _should_generate_answer(self, state: QAState) -> str:
        """判断 build_context 后是否需要 LLM 生成答案"""
        if state.error:
            return "error"

        if state.agent_iteration > 0:
            if self.config.retrieve.use_memory:
                return "skip_generate"
            return "finish"

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

            return {
                "question": query,
                "answer": answer,
                "messages": result.get("messages", existing_messages),
                "transformed_query": result.get("transformed_query", ""),
                "context": result.get("final_context", ""),
                "kb_context": result.get("kb_context", ""),
                "tool_context": result.get("tool_context", ""),
                "tool_calls_history": result.get("tool_calls_history", []),
                "agent_iterations": result.get("agent_iteration", 0),
            }

        except Exception as e:
            self.logger.error(f"问答处理失败: {e}")
            return {"error": f"处理请求时出错: {str(e)}"}

    def ask_sync(self, query: str, **kwargs) -> Dict[str, Any]:
        """同步问答接口（内部使用 asyncio.run）"""
        return asyncio.run(self.ask(query, **kwargs))

    async def ask_stream(
        self, query: str, **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """流式问答接口 - 纯 async"""

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
        )

        try:
            async for event in self.graph.astream(initial_state_object, config):
                if not event:
                    continue
                node_name = list(event.keys())[0]
                node_state = event[node_name]

                def _get(key, default=None):
                    if isinstance(node_state, dict):
                        return node_state.get(key, default)
                    return getattr(node_state, key, default)

                if _get("error"):
                    yield {
                        "node": node_name,
                        "status": "error",
                        "error": _get("error"),
                    }
                    continue

                yield {
                    "node": node_name,
                    "status": "processing",
                    "state": {
                        "kb_context": _get("kb_context"),
                        "tool_context": _get("tool_context"),
                        "final_context": _get("final_context"),
                    },
                }

                if node_name in ("agent_node", "generate_answer"):
                    messages = _get("messages") or []
                    ai_msgs = [
                        msg.content
                        for msg in messages
                        if isinstance(msg, AIMessage)
                        and not getattr(msg, "tool_calls", None)
                    ]
                    if ai_msgs:
                        yield {
                            "node": node_name,
                            "status": "complete",
                            "answer": ai_msgs[-1],
                            "context": _get("final_context"),
                        }

        except Exception as e:
            self.logger.error(f"流式问答处理失败: {e}")
            yield {"status": "error", "error": f"处理请求时出错: {str(e)}"}

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
