# -*- coding: utf-8 -*-
"""
@Time ： 2025/6/8 10:06
@Auth ： 吕鑫
@File ：qa_agent.py
@IDE ：PyCharm
"""

from typing import List, Dict, Optional, Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

import asyncio
import concurrent.futures

from src.configs.logger_config import setup_logger
from src.configs.config import AppConfig
from src.models.embedding import TextEmbedding
from src.db_services.milvus.connection_manager import MilvusConnectionManager
from src.models.llm import LLMWrapper
from src.cores.query_transformer import QueryTransformer
from src.mcp.mcp_client import MCPClient, mcp_main
from src.cores.message_builder import MessageBuilder
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = setup_logger(__name__)


class QAState(TypedDict):
    """QA Pipeline的状态定义"""
    # 核心数据
    messages: Annotated[List, add_messages]  # 对话消息
    original_query: str  # 原始查询
    transformed_query: Optional[str]  # 转换后的查询

    # 检索结果
    kb_context: Optional[str]  # 知识库上下文
    tool_context: Optional[str]  # 工具调用结果
    final_context: Optional[str]  # 最终上下文

    # 配置参数
    rewrite_config: bool
    use_knowledge_base: bool
    use_tools: bool
    use_memory: bool
    k: int  # 检索数量
    use_sparse: bool
    use_reranker: bool
    stream: bool

    # 流程控制
    next_step: Optional[str]  # 下一步操作
    error: Optional[str]  # 错误信息

    # 评估相关
    evaluation_mode: bool
    ground_truth: Optional[str]


def build_prompt_messages(
        state: QAState,
        system_prompt: str,
        memory_limit: int = 10
) -> List:
    """构建用于 LLM 调用的消息序列"""

    messages = [SystemMessage(content=system_prompt)]

    # 添加系统提示

    # 添加历史消息（如有）
    if state.get("use_memory") and state.get("messages"):
        history_messages = [
            msg for msg in state["messages"]
            if not isinstance(msg, SystemMessage)
        ]
        messages.extend(history_messages[-memory_limit:])

    # 构建用户消息（包含上下文 + 原始/转换查询）
    query_content = state["original_query"]

    if (
            state.get("transformed_query")
            and state["transformed_query"] != state["original_query"]
    ):
        query_content = f"原始问题：{state['original_query']}\n转换后的问题：{state['transformed_query']}"

    if state.get("final_context"):
        query_content = f"上下文信息：\n{state['final_context']}\n\n问题：{query_content}"

    messages.append(HumanMessage(content=query_content))

    return messages


class QA_Agent:
    """基于LangGraph的QA Pipeline"""

    def __init__(self, config: AppConfig):
        self.config = config
        self._init_components()
        self._build_graph()

    def _init_components(self):
        """初始化所有组件"""
        logger.info("初始化QA Pipeline组件...")

        # 文本嵌入
        self.embeddings = TextEmbedding(self.config.embedding)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        self.message_builder = MessageBuilder(self.config.message_builder)

        # 向量数据库
        self.db_connection_manager = MilvusConnectionManager(self.embeddings, self.text_splitter, self.config.milvus)

        # LLM包装器
        self.llm_caller = LLMWrapper(self.config.llm)

        # 查询转换器
        self.query_transformer = QueryTransformer(self.llm_caller, self.message_builder, self.embeddings,
                                                  self.db_connection_manager)

        # MCP客户端
        self.mcp_client = MCPClient(self.llm_caller)

        # 检查点保存器
        self.checkpointer = MemorySaver()

    def _build_graph(self):
        """构建LangGraph工作流"""
        # 创建状态图
        workflow = StateGraph(QAState)

        # 添加节点
        workflow.add_node("parse_query", self._parse_query)  # 解析查询及配置参数
        workflow.add_node("transform_query", self._transform_query)  # 转换查询
        workflow.add_node("check_retrieve_knowledge", self._check_retrieve_knowledge)  # 检查是否需要检索知识库
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
                "error": "handle_error"
            }
        )
        # 当执行transform_query时
        workflow.add_conditional_edges(
            "transform_query",
            self._should_retrieve_knowledge,
            {
                "do_retrieve": "retrieve_knowledge",
                "skip_retrieve": "check_call_tools",
                "error": "handle_error"
            }
        )

        workflow.add_conditional_edges(
            "check_call_tools",
            self._should_call_tools,
            {
                "call_tools": "call_tools",
                "skip_tools": "build_context",
                "error": "handle_error"
            }
        )

        # 是否需要知识库检索
        workflow.add_conditional_edges(
            "check_retrieve_knowledge",
            self._should_retrieve_knowledge,
            {
                "do_retrieve": "retrieve_knowledge",
                "skip_retrieve": "check_call_tools",
                "error": "handle_error"
            }
        )

        # 知识检索后的条件分支
        workflow.add_conditional_edges(
            "retrieve_knowledge",
            self._should_call_tools,
            {
                "call_tools": "call_tools",
                "skip_tools": "build_context",
                "error": "handle_error"
            }
        )

        # 工具调用后构建上下文
        workflow.add_edge("call_tools", "build_context")

        # 构建上下文后生成答案
        workflow.add_edge("build_context", "generate_answer")

        # 生成答案后的条件分支
        workflow.add_conditional_edges(
            "generate_answer",
            self._should_update_memory,
            {
                "update_memory": "update_memory",
                "finish": END,
                "error": "handle_error"
            }
        )

        # 更新记忆后结束
        workflow.add_edge("update_memory", END)

        # 错误处理后结束
        workflow.add_edge("handle_error", END)

    # =================== 节点实现 ===================

    def _parse_query(self, state: QAState) -> QAState:
        """解析查询，设置默认参数"""
        try:
            logger.info(f"解析查询: {state['original_query']}")

            # 设置默认参数
            state.update({
                "k": state.get("k", self.config.default_top_k),
                "use_sparse": state.get("use_sparse", self.config.use_sparse),
                "use_reranker": state.get("use_reranker", self.config.use_reranker),
                "use_knowledge_base": state.get("use_knowledge_base", self.config.use_knowledge_base),
                "use_tools": state.get("use_tools", self.config.use_tools),
                "use_memory": state.get("use_memory", self.config.use_memory),
                "error": None
            })

            return state

        except Exception as e:
            logger.error(f"解析查询失败: {e}")
            state["error"] = f"解析查询失败: {str(e)}"
            return state

    def _transform_query(self, state: QAState) -> QAState:
        """查询转换"""
        try:
            original_query = state["original_query"]
            logger.info(f"转换查询: {original_query}")
            if state['rewrite_config']:
                mode = self.config.rewrite_config["mode"]
                transformed_query = self.query_transformer.transform_query(
                    original_query, mode
                )
                state["transformed_query"] = transformed_query
                logger.info(f"查询转换完成: {transformed_query}")
            else:
                state["transformed_query"] = ''

            return state

        except Exception as e:
            logger.error(f"查询转换失败: {e}")
            state["error"] = f"查询转换失败: {str(e)}"
            return state

    def _check_retrieve_knowledge(self, state: QAState) -> QAState:
        return state

    def _check_call_tools(self, state: QAState) -> QAState:
        return state

    def _retrieve_knowledge(self, state: QAState) -> QAState:
        """知识库检索"""
        try:
            if not state["use_knowledge_base"]:
                state["kb_context"] = ""
                return state

            query = state.get("transformed_query", '') or state["original_query"]
            logger.info(f"开始知识库检索: {query}")

            # 选择检索方法
            if (state.get("rewrite_config") and self.config.rewrite_config["mode"] == "hyde"):
                results = self.query_transformer.hyde_search(query, state["k"])
            else:
                results = self.db_manager.search(
                    query=query,
                    k=state["k"],
                    use_sparse=state["use_sparse"],
                    use_reranker=state["use_reranker"]
                )

            # 处理检索结果
            if not results:
                state["kb_context"] = "未检索到相关知识。"
                logger.warning("知识库检索未返回结果")
            else:
                valid_results = [doc for doc in results if doc and doc.get('text')]
                if not valid_results:
                    state["kb_context"] = "未检索到有效知识。"
                else:
                    kb_context = "\n".join([
                        f"【文档{i + 1}】{doc['text']}"
                        for i, doc in enumerate(valid_results)
                    ])
                    state["kb_context"] = f"【知识库检索内容】\n{kb_context}"
                logger.info(f"知识库检索返回 {len(results)} 条结果")

            return state

        except Exception as e:
            logger.error(f"知识库检索失败: {e}")
            state["error"] = f"知识库检索失败: {str(e)}"
            return state

    def _call_tools(self, state: QAState) -> QAState:
        """调用外部工具"""
        try:
            if not state["use_tools"]:
                state["tool_context"] = ""
                return state

            query = state["original_query"]
            logger.info(f"调用工具: {query}")

            def run_in_thread():
                return asyncio.run(mcp_main(self.mcp_client, query))

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                tool_result = future.result(timeout=30)

            if tool_result:
                state["tool_context"] = f"【工具返回内容】\n{tool_result}"
                logger.info(f"工具调用成功，返回内容长度: {len(tool_result)}")
            else:
                state["tool_context"] = ""
                logger.info("工具调用无返回结果")

            return state

        except Exception as e:
            logger.error(f"工具调用失败: {e}")
            state["error"] = f"工具调用失败: {str(e)}"
            return state

    def _build_context(self, state: QAState) -> QAState:
        """构建最终上下文"""
        try:
            logger.info("构建最终上下文")

            context_parts = []

            # 添加工具结果
            if state.get("tool_context"):
                context_parts.append(state["tool_context"])

            # 添加知识库内容
            if state.get("kb_context"):
                context_parts.append(state["kb_context"])

            state["final_context"] = "\n\n".join(context_parts)
            logger.info(f"上下文构建完成，长度: {len(state['final_context'])}")

            return state

        except Exception as e:
            logger.error(f"构建上下文失败: {e}")
            state["error"] = f"构建上下文失败: {str(e)}"
            return state

    def _generate_answer(self, state: QAState) -> QAState:
        """生成答案"""
        try:
            logger.info("开始生成答案")
            system_prompt = (
                self.config.kb_system_prompt
                if state["use_knowledge_base"]
                else self.config.system_prompt
            )
            messages = build_prompt_messages(
                state,
                system_prompt=system_prompt,
                memory_limit=10
            )
            answer = self.llm_caller.chat(messages=messages,
                                          extra_body={"chat_template_kwargs": {"enable_thinking": False}})

            state["messages"] = state.get("messages", []) + [
                HumanMessage(content=state["original_query"]),
                AIMessage(content=answer)
            ]

            logger.info("答案生成完成")
            return state

        except Exception as e:
            logger.error(f"生成答案失败: {e}")
            state["error"] = f"生成答案失败: {str(e)}"
            return state

    def _update_memory(self, state: QAState) -> QAState:
        """更新记忆"""
        try:
            if state["use_memory"] and state.get("messages"):
                # 确保消息历史不超过最大长度
                max_history = 10  # 可以根据需要调整
                state["messages"] = state["messages"][-max_history:]
                logger.info(f"记忆更新完成，当前历史消息数: {len(state['messages'])}")
            return state
        except Exception as e:
            logger.error(f"更新记忆失败: {e}")
            state["error"] = f"更新记忆失败: {str(e)}"
            return state

    def _handle_error(self, state: QAState) -> QAState:
        """处理错误"""
        error_msg = state.get("error", "未知错误")
        logger.error(f"处理错误: {error_msg}")

        # 添加错误消息到对话历史
        state["messages"] = state.get("messages", []) + [
            AIMessage(content=f"抱歉，处理您的请求时出现错误：{error_msg}")
        ]

        return state

    # =================== 条件判断函数 ===================

    def _should_transform_query(self, state: QAState) -> str:
        """判断是否需要转换查询"""
        if state.get("error"):
            return "error"

        if state.get('rewrite_config'):
            return "transform"
        else:
            return "skip_transform"

    def _should_retrieve_knowledge(self, state: dict) -> str:
        """判断是否需要做知识库查询"""
        if state.get("error"):
            return "error"

        if state.get("use_knowledge_base"):
            return "do_retrieve"
        else:
            return "skip_retrieve"

    def _should_call_tools(self, state: QAState) -> str:
        """判断是否需要调用工具"""
        if state.get("error"):
            return "error"

        if state["use_tools"]:
            return "call_tools"
        else:
            return "skip_tools"

    def _should_update_memory(self, state: QAState) -> str:
        """判断是否需要更新记忆"""
        if state.get("error"):
            return "error"

        if state["use_memory"]:
            return "update_memory"
        else:
            return "finish"

    # =================== 外部接口 ===================
    def _build_config_and_state(self, query: str, **kwargs) -> (RunnableConfig, dict):
        """
        构建 configs 和初始 state

        :param query: 用户输入的查询
        :param kwargs: 其他动态参数
        :return: (configs, initial_state)
        """
        config = RunnableConfig(
            configurable={
                "thread_id": kwargs.get("context_id", ""),
            }
        )

        existing_messages = kwargs.get("messages", [])

        initial_state = {
            "original_query": query,
            "messages": existing_messages,
            "rewrite_config": kwargs.get("rewrite_config", self.config.rewrite_config),
            "use_knowledge_base": kwargs.get("use_knowledge_base", self.config.use_knowledge_base),
            "use_tools": kwargs.get("use_tools", self.config.use_tools),
            "use_memory": kwargs.get("use_memory", self.config.use_memory),
            "k": kwargs.get("k", self.config.default_top_k),
            "use_sparse": kwargs.get("use_sparse", self.config.use_sparse),
            "use_reranker": kwargs.get("use_reranker", self.config.use_reranker)
        }

        return config, initial_state

    async def ask(self, query: str, **kwargs) -> Dict:
        """同步问答接口"""
        # 创建配置对象
        config, initial_state = self._build_config_and_state(query, **kwargs)

        existing_messages = kwargs.get("messages", [])  # 参数传入，前台来管理

        # 初始化状态
        try:
            result = self.graph.invoke(initial_state, config)

            if result.get("error"):
                return {"error": result["error"]}

            # 提取最后的AI消息作为答案
            ai_messages = [
                msg.content for msg in result.get("messages", [])
                if isinstance(msg, AIMessage)
            ]

            answer = ai_messages[-1] if ai_messages else "无法生成答案"

            logger.info(f"最终答案: {answer}")
            return {
                "question": query,
                "answer": answer,
                "messages": result.get("messages", existing_messages),  # ✅ 返回给前台
                "transformed_query": result.get("transformed_query", ""),
                "context": result.get("final_context", ""),
                "kb_context": result.get("kb_context", ""),
                "tool_context": result.get("tool_context", "")
            }

        except Exception as e:
            logger.error(f"问答处理失败: {e}")
            return {"error": f"处理请求时出错: {str(e)}"}

    async def ask_stream(self, query: str, **kwargs):
        """流式问答接口 - 改进版本"""

        # 创建配置对象
        config, initial_state = self._build_config_and_state(query, **kwargs)
        existing_messages = kwargs.get("messages", [])
        try:
            # 流式执行图
            rewrite_flag = False
            kb_flag = False
            tool_flag = False
            messages_flag = False
            for event in self.graph.stream(initial_state, config, stream_mode='values'):
                if event.get('transformed_query') and event.get('rewrite_config') and not rewrite_flag:  # 完成查询重写
                    rewrite_flag = True
                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": event.get('transformed_query'),
                        "status": "transformed_query"
                    }
                if event.get('kb_context') and event.get('use_knowledge_base') and not kb_flag:  # 获取知识库内容
                    kb_flag = True
                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": event.get('kb_context'),
                        "status": "kb_context"
                    }
                if event.get('tool_context') and event.get('use_tools') and not tool_flag:  # 获取工具内容
                    tool_flag = True
                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": event.get('tool_context'),
                        "status": "tool_context"
                    }
                if event.get('messages') and not messages_flag:  # 生成最后答案
                    messages_flag = True
                    last_msg = event['messages'][-1]
                    if isinstance(last_msg, AIMessage):
                        yield {
                            "is_task_complete": False,
                            "require_user_input": False,
                            "content": last_msg.content,  # 取纯字符串
                            "status": "generated_answer"
                        }

            yield self._get_final_response(config, query, existing_messages)

        except Exception as e:
            logger.error(f"流式问答处理失败: {e}")
            yield {
                "is_task_complete": False,
                "require_user_input": True,
                "content": f"处理请求时出错: {str(e)}",
                "status": "error"
            }

    def _get_final_response(self, config, query, existing_messages):
        try:
            current_state = self.graph.get_state(config)
            if not current_state or not current_state.values:
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": "无法获取处理结果，请重试。",
                    "status": "error"
                }

            state_values = current_state.values
            if state_values.get("error"):
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": f"处理过程中出现错误: {state_values['error']}",
                    "status": "error"
                }

            messages = state_values.get("messages", existing_messages)
            ai_messages = [msg.content for msg in messages if isinstance(msg, AIMessage)]
            if not ai_messages:
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": "无法生成答案，请重试。",
                    "status": "error"
                }

            answer = ai_messages[-1]
            return {
                "is_task_complete": True,
                "require_user_input": False,
                "content": answer,
                "status": "completed",
                "metadata": {
                    "question": query,
                    "answer": answer,
                    "messages": messages,
                    "transformed_query": state_values.get("transformed_query", ""),
                    "context": state_values.get("final_context", ""),
                    "kb_context": state_values.get("kb_context", ""),
                    "tool_context": state_values.get("tool_context", "")
                }
            }

        except Exception as e:
            logger.error(f"获取最终响应失败: {e}")
            return {
                "is_task_complete": False,
                "require_user_input": True,
                "content": f"获取最终结果时出错: {str(e)}",
                "status": "error"
            }

    def export_graph(self, output_path: str = "qa_pipeline_graph.png"):
        """导出图结构"""
        try:
            # 生成Mermaid图
            mermaid_code = self.graph.get_graph().draw_mermaid()

            # 保存到文件
            with open(output_path.replace('.png', '.mmd'), 'w', encoding='utf-8') as f:
                f.write(mermaid_code)

            logger.info(f"图结构已导出到: {output_path}")
            return mermaid_code

        except Exception as e:
            logger.error(f"导出图结构失败: {e}")
            return None
