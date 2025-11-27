# -*- coding: utf-8 -*-
"""
@Time    : 2025/4/23
@Author  : 吕鑫
@File    : qa_pipeline.py
@Desc    : 支持配置化、批量问答、自动评估、接口调用的知识库问答模块封装
"""
from src.models.embedding import TextEmbedding
from src.db_services.milvus.connection_manager import MilvusConnectionManager
from src.agent.memory import ConversationMemory
from src.models.llm import LLMWrapper
from src.evaluate.evaluate import QAEvaluator
from src.evaluate.rag import RAGASEvaluator
from src.cores.message_builder import MessageBuilder
from src.configs.config import AppConfig
from src.configs.retrieve_config import SearchConfig
from src.cores.query_transformer import QueryTransformer
from src.configs.logger_config import setup_logger
from src.mcp.mcp_client import MCPClient, mcp_main
from src.utils.prompt import PromptManager

from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict
import asyncio
from langfuse import get_client, observe

logger = setup_logger(__name__)

from typing import Optional


class QAPipeline:
    def __init__(self, config: AppConfig, langfuse_client: Optional[get_client] = None):
        self.logger = logger
        self.config = config
        self.langfuse_client = langfuse_client
        self._init_components()

        self.logger.info("QAPipeline 初始化完成")

    def _init_components(self):
        self.logger.info("初始化 QAPipeline...")

        self.logger.info("初始化文本嵌入模型...")
        self.embeddings = TextEmbedding(self.config.embedding)

        self.logger.info("初始化 LLM 包装器...")
        self.llm_caller = LLMWrapper(self.config.llm)

        self.logger.info("初始化文本分割器...")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.splitter.chunk_size,
            chunk_overlap=self.config.splitter.chunk_overlap
        )

        self.logger.info("初始化消息构建器...")
        self.message_builder = MessageBuilder(self.config.message_builder)

        self.logger.info("初始化向量数据库管理器...")
        # 使用统一的 MilvusDB 入口类
        self.db_connection_manager = MilvusConnectionManager(self.embeddings, self.text_splitter, self.config.milvus,
                                                             self.config.retrieve)

        self.logger.info("初始化对话记忆...")
        self.memory = ConversationMemory(self.config.retrieve)
        self.logger.info("初始化 MCP 客户端...")
        self.mcp_client = MCPClient(self.llm_caller)

        self.logger.info("初始化评估器...")
        self.evaluator = QAEvaluator(self, self.config.evaluation, self.config.retrieve)
        self.ragas_evaluator = RAGASEvaluator(config=self.config.evaluation)
        self.logger.info("初始化工具管理器...")
        self.logger.info("初始化查询转换器...")
        self.query_transformer = QueryTransformer(self.llm_caller, self.message_builder, self.embeddings,
                                                  self.db_connection_manager, self.config.rewrite)
        # 初始化 Langfuse 客户端
        self.logger.info("初始化 Langfuse ...")
        self.logger.info("初始化 prompt manager")
        self.prompt_manager = PromptManager(self.langfuse_client)

    @observe(name="QAPipline.build_knowledge_base")
    async def build_knowledge_base(self, data_dir: str):
        self.logger.info(f"开始构建知识库，数据目录: {data_dir}")
        await self.db_connection_manager.add_documents_from_dir(data_dir)
        self.logger.info("知识库构建完成")

    @observe(name="QAPipeline._build_messages")
    async def _build_messages(self, query):
        system_prompt = self.prompt_manager.get_prompt(
            "kb_system_prompt") if self.config.retrieve.use_kb else self.prompt_manager.get_prompt("system_prompt")

        context, _ = await self._prepare_context(query)
        messages = self.message_builder.build(
            query=query,
            context=context,
            use_memory=self.config.retrieve.use_memory,
            memory_items=list(self.memory.memory) if self.config.retrieve.use_memory else [],
            system_prompt_template=system_prompt,
            no_think=self.config.retrieve.no_think,
            max_tokens=self.config.message_builder.message_max_tokens,
            max_history_turns=self.config.retrieve.memory_window_size
        )
        return messages, context

    async def _retrieve_kb_context(self, query: str) -> tuple[str, str]:
        """
        执行知识库检索，返回格式化的上下文和原始检索内容。
        如果配置未开启知识库，返回空元组。
        """
        retrieve_config = self.config.retrieve
        if not retrieve_config.use_kb:
            return "", ""

        self.logger.info("🟢 开始知识库检索")

        # 确定并执行检索逻辑
        use_hyde = retrieve_config.use_rewrite and retrieve_config.rewrite_mode == 'hyde'

        if use_hyde:
            results = await self.query_transformer.hyde_search(query, retrieve_config)
        else:
            results = await self.db_connection_manager.asearch(query, retrieve_config)

        # 处理检索结果
        if results and results[0]:
            kb_context_docs = "\n".join([f"【文档{i + 1}】{doc.get('text', '内容缺失')}"
                                         for i, doc in enumerate(results)])
            return f"【知识库检索内容】\n{kb_context_docs}", kb_context_docs
        else:
            return "【知识库检索内容】\n未检索到相关知识。", ""

    async def _retrieve_tool_context(self, query: str) -> str:
        """
        执行工具调用，并返回格式化的上下文字符串。
        如果配置未开启工具，返回空字符串。
        """
        if not self.config.retrieve.use_tool:
            return ""
        try:
            tool_result = await mcp_main(self.mcp_client, query)
        except Exception as e:
            self.logger.error(f"❌ 工具调用失败: {e}", exc_info=True)
            return ""

        if tool_result:
            self.logger.debug(f"✅ 工具返回内容长度: {len(tool_result)}")
            return f"【工具返回内容】\n{tool_result}"

        return ""

    @observe(name="QAPipline._prepare_context")
    async def _prepare_context(self, query) -> (str, str):
        context_blocks = []
        # --- Step 1: 工具调用 ---
        tool_formatted_context = await self._retrieve_tool_context(query)
        tool_context_raw = ""
        if tool_formatted_context:
            context_blocks.append(tool_formatted_context)
            tool_context_raw = tool_formatted_context.split('\n', 1)[-1]

        # --- Step 2: 知识库检索 ---
        kb_formatted_context, kb_context_raw = await self._retrieve_kb_context(query)
        if kb_formatted_context:
            context_blocks.append(kb_formatted_context)

        # --- Step 3: 结果组装 ---
        final_context = "\n\n".join(context_blocks)
        summary_context = kb_context_raw or tool_context_raw
        return final_context, summary_context

    @observe(name="QAPipline.ask", as_type="chain")
    async def ask(self, query: str, config: SearchConfig = None) -> Dict:

        if config:
            self.config.retrieve = config

        if self.config.retrieve.session_id:
            self.logger.info(
                f"更新当前追踪用户及会话ID: {self.config.retrieve.user_id}, {self.config.retrieve.session_id}")
            self.langfuse_client.update_current_trace(user_id=self.config.retrieve.user_id,
                                                      session_id=self.config.retrieve.session_id)

        self.logger.info(f"收到问题: {query}")
        if not query.strip():
            self.logger.error("问题不能为空")
            return {"error": "问题不能为空"}

        try:
            if self.config.retrieve.use_rewrite and self.config.retrieve.rewrite_mode != "hyde":
                self.logger.info(f"使用 {self.config.retrieve.rewrite_mode} 模式重写查询")
                query = self.query_transformer.transform_query(query, self.config.retrieve.rewrite_mode)

            self.logger.debug(
                f"查询参数: k={self.config.retrieve.top_k}, use_sparse={self.config.retrieve.use_sparse}, use_reranker={self.config.retrieve.use_reranker}")
            messages, context = await self._build_messages(query)
            self.logger.info("开始调用 LLM 生成回答")
            answer = await self.llm_caller.achat(messages)
            self.logger.info("LLM 回答生成完成")

            if self.config.retrieve.use_memory:
                self.logger.debug("更新对话记忆")
                self.memory.add(query, answer)

            return {"question": query, "answer": answer, "context": context}
        except Exception as e:
            self.logger.exception("处理请求时发生异常")
            return {"error": f"处理请求时出错: {str(e)}"}

    @observe(name="QAPipline.ask_stream")
    async def ask_stream(self, query: str, config: SearchConfig = None):

        if config:
            self.config.retrieve = config
        self.logger.info(f"收到流式问题: {query}")
        if not query.strip():
            self.logger.error("问题不能为空")
            yield {"error": "问题不能为空"}
            return

        try:
            if self.config.retrieve.use_rewrite:
                self.logger.info(f"使用 {self.config.retrieve.rewrite_mode} 模式重写查询")
                query = self.query_transformer.transform_query(query, self.config.retrieve.rewrite_mode)

            self.logger.debug(
                f"流式查询参数: k={self.config.retrieve.top_k}, use_sparse={self.config.retrieve.use_sparse}, use_reranker={self.config.retrieve.use_reranker}")
            messages, context = await self._build_messages(query)
            self.logger.info("开始流式调用 LLM")
            stream = await self.llm_caller.achat(messages, stream=True)
            answer = ""
            async for chunk in stream:
                delta = getattr(chunk.choices[0].delta, 'content', None)
                if delta:
                    answer += delta
                    yield {"delta": delta}

            self.logger.info("流式回答生成完成")
            if self.config.retrieve.use_memory:
                self.logger.debug("更新对话记忆")
                self.memory.add(query, answer)

        except Exception as e:
            self.logger.exception("处理流式请求时发生异常")
            yield {"error": f"处理请求时出错: {str(e)}"}

    @observe(name="QAPipline.batch_ask")
    async def batch_ask(self, questions: List[str]) -> List[Dict]:
        self.logger.info(f"开始批量处理 {len(questions)} 个问题")
        # 创建一系列异步任务
        tasks = [self.ask(q) for q in questions]
        # 使用 asyncio.gather 并发执行所有任务
        results = await asyncio.gather(*tasks)
        self.logger.info("批量处理完成")
        return results

    @observe(name="QAPipline.evaluate")
    async def evaluate(self, qa_pairs):
        """
        评估问答对
        
        Args:
            qa_pairs: 问答对列表
        """
        self.logger.info(f"开始评估，使用 {self.config.evaluation.eval_method} 方法")

        if self.config.evaluation.eval_method == "ragas":
            self.logger.info("使用 RAGAS 评估方法")

            # 准备 RAGAS 评估数据
            async def run_and_format_ask(pair):
                """内部异步函数，用于执行 ask 并格式化 RAGAS 需要的数据"""
                result = await self.ask(pair["question"])
                return {
                    "query": pair["question"],
                    "prediction": result["answer"],
                    # 使用 get() 增加健壮性，防止 context 为 None
                    "contexts": result.get("context", "").split("\n") if result.get("context") else [],
                    "ground_truths": [pair["answer"]]
                }

            # 使用 asyncio.gather 并发执行所有问答任务
            tasks = [run_and_format_ask(pair) for pair in qa_pairs]
            qa_data = await asyncio.gather(*tasks)
            # 执行 RAGAS 评估
            results = self.ragas_evaluator.evaluate(qa_data)
            self.logger.info("RAGAS 评估完成")
            return results
        else:
            # 使用原有评估方法
            results = self.evaluator.evaluate(qa_pairs=qa_pairs)
            self.logger.info("评估完成")
            return results

    @observe(name="QAPipline.clear_memory")
    def clear_memory(self):
        """清除对话历史"""
        self.logger.info("清除对话记忆")
        self.memory.clear()
