# -*- coding: utf-8 -*-
"""
@Time    : 2025/4/23
@Author  : 吕鑫
@File    : query_transformer.py
@Desc    : 查询转换模块
"""
from typing import List, Dict, Optional
from src.models.llm import LLMWrapper
from src.configs.retrieve_config import SearchConfig

from src.cores.message_builder import MessageBuilder
from src.models.embedding import TextEmbedding
from src.db_services.milvus.connection_manager import MilvusConnectionManager
from src.configs.logger_config import setup_logger
import numpy as np

logger = setup_logger(__name__)


class QueryTransformer:
    def __init__(self, llm: LLMWrapper, message_builder: MessageBuilder, embeddings: TextEmbedding,
                 db_connection_manager: MilvusConnectionManager):
        self.logger = logger
        self.logger.info("初始化查询转换器...")
        self.llm = llm
        self.message_builder = message_builder
        self.embeddings = embeddings
        self.db_connection_manager = db_connection_manager
        self.logger.info("查询转换器初始化完成")

    def get_templates(self, mode: str) -> str:
        """获取查询转换模板"""
        self.logger.debug(f"获取 {mode} 模式的查询转换模板")
        templates = {
            "rewrite": """
                你是一个AI助手，任务是重新制定用户查询以改进RAG系统中的检索。给定原始查询，将其重写为更具体、更详细，并且更有可能检索相关信息。
                原始查询：{original_query}
                重写的查询：
            """,
            "step_back": """
                你是一个 AI 助手，任务是生成更广泛、更通用的查询，以改进 RAG 系统中的上下文检索。
                给定原始查询，生成一个更通用的step-back查询，该查询可以帮助检索相关的背景信息。
                原始查询：{original_query}
                step-back查询：
            """,
            "sub_query": """
                你是一名 AI 助手，任务是将复杂的查询分解为 RAG 系统的更简单的子查询。
                给定原始查询，将其分解为 2-4 个更简单的子查询，当一起回答时，将提供对原始查询的全面响应。
                原始查询： {original_query}
                示例：气候变化对环境有哪些影响？
                子查询：
                1.气候变化对生物多样性有哪些影响？
                2.气候变化如何影响海洋？
                3.气候变化对农业有哪些影响？
                4.气候变化对人类健康有哪些影响？
            """,
            "hyde": """
                你是一个AI助手，任务是生成假设的答案，用于信息检索。
                根据原始查询，生成一个假设的答案，该答案将用于信息检索，但不用于最终回答。
                原始查询： {original_query}
                假设的答案：
            """
        }
        if mode not in templates:
            self.logger.error(f"不支持的查询转换模式: {mode}")
            raise ValueError(f"不支持的查询转换模式: {mode}")
        self.logger.debug(f"成功获取 {mode} 模式的模板")
        return templates[mode]

    def transform_query(self, query: str, mode: str = "rewrite") -> str:
        """
        转换查询
        
        Args:
            query: 原始查询
            mode: 转换模式
                - rewrite: 查询重写
                - step_back: 生成 step-back 查询
                - sub_query: 生成子查询列表（返回 list）
        """
        self.logger.info(f"开始转换查询，模式: {mode}")
        self.logger.debug(f"原始查询: {query}")

        if mode == "hyde":
            self.logger.error("HyDE 模式请使用 generate_hyde_vector 或 hyde_search 方法")
            raise ValueError("请使用 generate_hyde_vector 或 hyde_search 方法处理 HyDE 模式")

        try:
            prompt_template = self.get_templates(mode).format(original_query=query)
            self.logger.debug("已构建提示模板")

            messages = self.message_builder.build(query, system_prompt_template=prompt_template)
            self.logger.debug("已构建消息")

            rewrite_query = self.llm.chat(messages,
                                          extra_body={"chat_template_kwargs": {"enable_thinking": False}}).strip()
            self.logger.info(f"查询转换完成，新查询: {rewrite_query[:100]}...")
            return rewrite_query

        except Exception as e:
            self.logger.exception(f"查询转换失败: {str(e)}")
            raise

    def generate_hyde_vector(self, query: str, num_hypo: int = 3) -> Optional[List[float]]:
        """
        生成 HyDE 融合向量：原始查询向量 + 多个假设答案向量
        
        Args:
            query: 原始查询
            num_hypo: 假设答案数量
        """
        self.logger.info(f"开始生成 HyDE 向量，假设答案数量: {num_hypo}")
        self.logger.debug(f"原始查询: {query}")

        try:
            # 生成假设答案
            hypo_docs = []
            prompt_template = self.get_templates("hyde").format(original_query=query)
            self.logger.debug("已构建 HyDE 提示模板")

            for i in range(num_hypo):
                self.logger.debug(f"生成第 {i + 1}/{num_hypo} 个假设答案")
                messages = self.message_builder.build(query, system_prompt_template=prompt_template)
                hypo = self.llm.chat(messages, extra_body={"chat_template_kwargs": {"enable_thinking": False}}).strip()
                hypo_docs.append(hypo)
                self.logger.debug(f"假设答案 {i + 1}: {hypo[:100]}...")

            # 生成向量
            self.logger.info("开始生成向量...")
            hypo_vectors = self.embeddings.get_embedding(hypo_docs)
            query_vector = self.embeddings.get_embedding(query)[0]

            # 融合向量
            combined_vector = [np.mean(np.vstack([*hypo_vectors, query_vector]), axis=0).tolist()]
            self.logger.info("HyDE 向量生成完成")
            self.logger.debug(f"向量维度: {len(combined_vector[0])}")

            return combined_vector[0]

        except Exception as e:
            self.logger.exception(f"HyDE 向量生成失败: {str(e)}")
            return None

    async def hyde_search(self, query: str, config: SearchConfig) -> List[Dict]:
        """
        使用 HyDE 向量进行文档检索
        """

        try:
            vector = self.generate_hyde_vector(query)
            docs = await self.db_connection_manager.asearch(query=vector, search_config=config)
            return docs

        except Exception as e:
            self.logger.exception(f"HyDE 搜索失败: {str(e)}")
            raise
