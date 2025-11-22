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
from typing import List, Dict, Optional
import asyncio
from langfuse import get_client, observe

logger = setup_logger(__name__)

from pydantic import BaseModel, Field, validator
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
        self.db_connection_manager = MilvusConnectionManager(self.embeddings, self.text_splitter, self.config.milvus)

        self.logger.info("初始化对话记忆...")
        self.memory = ConversationMemory(self.config.retrieve)
        self.logger.info("初始化 MCP 客户端...")
        self.mcp_client = MCPClient(self.llm_caller)

        self.logger.info("初始化评估器...")
        self.evaluator = QAEvaluator(self)
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
    def build_knowledge_base(self, data_dir: str):
        self.logger.info(f"开始构建知识库，数据目录: {data_dir}")
        self.db_connection_manager.add_documents_from_dir(data_dir)
        self.logger.info("知识库构建完成")

    @observe(name="QAPipeline._build_messages")
    def _build_messages(self, query):
        config = self.config.retrieve
        system_prompt = self.prompt_manager.get_prompt(
            "kb_system_prompt") if config.use_kb else self.prompt_manager.get_prompt("system_prompt")

        context, _ = self._prepare_context(query)
        messages = self.message_builder.build(
            query=query,
            context=context,
            use_memory=config.use_memory,
            memory_items=list(self.memory.memory) if config.use_memory else [],
            system_prompt_template=system_prompt,
            no_think=config.no_think,
            max_tokens=self.config.message_builder.message_max_tokens,
            max_history_turns=config.memory_window_size  # todo 记忆管理大小与message大小是否重复？
        )
        return messages, context

    @observe(name="QAPipline._prepare_context")
    def _prepare_context(self, query) -> (str, str):

        config = self.config.retrieve
        context_blocks = []
        # Step 1: 工具调用
        tool_context = ''
        if config.use_tool:
            tool_result = asyncio.run(mcp_main(self.mcp_client, query))
            if tool_result:
                context_blocks.append(f"【工具返回内容】\n{tool_result}")
                self.logger.debug(f"工具返回内容长度: {len(tool_result)}")

        # Step 2: 知识库检索
        kb_context = ''
        if config.use_kb:
            self.logger.info("开始知识库检索")
            if config.use_rewrite and config.rewrite_mode == 'hyde':
                self.logger.info("使用 HyDE 方法进行检索")
                results = self.query_transformer.hyde_search(query,config.top_k)
            else:
                results = self.db_connection_manager.search(query=query, k=config.top_k, use_sparse=config.use_sparse,
                                                            use_reranker=config.use_reranker)
            if not results or not results[0]:
                kb_context = "未检索到相关知识。"
                self.logger.warning("知识库检索未返回结果")
            else:
                kb_context = "\n".join([f"【文档{i + 1}】{doc['text']}" for i, doc in enumerate(results)])
                self.logger.info(f"知识库检索返回 {len(results)} 条结果")
            context_blocks.append(f"【知识库检索内容】\n{kb_context}")

        return "\n\n".join(context_blocks), kb_context or tool_context

    @observe(name="QAPipline.ask", as_type="chain")
    def ask(self, query: str) -> Dict:

        config = self.config.retrieve
        if config.session_id:
            self.logger.info(f"更新当前追踪用户及会话ID: {config.user_id}, {config.session_id}")
            self.langfuse_client.update_current_trace(user_id=config.user_id, session_id=config.session_id)

        self.logger.info(f"收到问题: {query}")
        if not query.strip():
            self.logger.error("问题不能为空")
            return {"error": "问题不能为空"}

        try:
            if config.use_rewrite and config.rewrite_config["mode"] != "hyde":
                self.logger.info(f"使用 {config.rewrite_mode} 模式重写查询")
                query = self.query_transformer.transform_query(query, config.rewrite_config["mode"])

            self.logger.debug(
                f"查询参数: k={config.top_k}, use_sparse={config.use_sparse}, use_reranker={config.use_reranker}")
            messages, context = self._build_messages(query)
            self.logger.info("开始调用 LLM 生成回答")
            answer = self.llm_caller.chat(messages)
            self.logger.info("LLM 回答生成完成")

            if config.use_memory:
                self.logger.debug("更新对话记忆")
                self.memory.add(query, answer)

            return {"question": query, "answer": answer, "context": context}
        except Exception as e:
            self.logger.exception("处理请求时发生异常")
            return {"error": f"处理请求时出错: {str(e)}"}

    @observe(name="QAPipline.ask_stream")
    def ask_stream(self, query: str):

        config = self.config.retrieve
        self.logger.info(f"收到流式问题: {query}")
        if not query.strip():
            self.logger.error("问题不能为空")
            yield {"error": "问题不能为空"}
            return

        try:
            if config.use_rewrite:
                self.logger.info(f"使用 {config.rewrite_mode} 模式重写查询")
                query = self.query_transformer.transform_query(query, config.rewrite_mode)

            self.logger.debug(
                f"流式查询参数: k={config.top_k}, use_sparse={config.use_sparse}, use_reranker={config.use_reranker}")
            messages, context = self._build_messages(query, config.use_knowledge_base, config.k, config.use_sparse,
                                                     config.use_reranker, config.use_tool, config.use_memory,
                                                     config.no_think)
            self.logger.info("开始流式调用 LLM")
            stream = self.llm_caller.chat(messages, stream=True)

            answer = ""
            for chunk in stream:
                delta = getattr(chunk.choices[0].delta, 'content', None)
                if delta:
                    answer += delta
                    yield {"delta": delta}

            self.logger.info("流式回答生成完成")
            if config.use_memory:
                self.logger.debug("更新对话记忆")
                self.memory.add(query, answer)

        except Exception as e:
            self.logger.exception("处理流式请求时发生异常")
            yield {"error": f"处理请求时出错: {str(e)}"}

    @observe(name="QAPipline.batch_ask")
    def batch_ask(self, questions: List[str]) -> List[Dict]:
        self.logger.info(f"开始批量处理 {len(questions)} 个问题")
        results = [self.ask(q) for q in questions]
        self.logger.info("批量处理完成")
        return results

    @observe(name="QAPipline.evaluate")
    def evaluate(self, qa_pairs, method: Optional[str] = None, k: Optional[int] = None):
        """
        评估问答对
        
        Args:
            qa_pairs: 问答对列表
            method: 评估方法 (rouge/bert/gpt/ragas)
            k: 检索数量
        """
        self.logger.info(f"开始评估，使用 {method or self.config.default_eval_method} 方法")
        method = method or self.config.default_eval_method
        k = k or self.config.default_eval_limit

        if method == "ragas":
            self.logger.info("使用 RAGAS 评估方法")
            # 准备 RAGAS 评估数据
            qa_data = []
            for pair in qa_pairs:
                result = self.ask(pair["question"], k=k, use_knowledge_base=True)
                qa_data.append({
                    "query": pair["question"],
                    "prediction": result["answer"],
                    "contexts": result["context"].split("\n") if result["context"] else [],
                    "ground_truths": [pair["answer"]]
                })

            # 执行 RAGAS 评估
            results = self.ragas_evaluator.evaluate(qa_data)
            self.logger.info("RAGAS 评估完成")
            return results
        else:
            # 使用原有评估方法
            results = self.evaluator.evaluate(method=method, qa_pairs=qa_pairs, k=k)
            self.logger.info("评估完成")
            return results

    @observe(name="QAPipline.clear_memory")
    def clear_memory(self):
        """清除对话历史"""
        self.logger.info("清除对话记忆")
        self.memory.clear()
