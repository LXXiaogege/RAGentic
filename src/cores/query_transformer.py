# -*- coding: utf-8 -*-
"""
@Time    : 2025/4/23
@Author  : 吕鑫
@File    : query_transformer.py
@Desc    : 查询转换模块
"""

from typing import Any, Dict, List, Optional

import numpy as np

from src.configs.logger_config import setup_logger
from src.configs.prompt_config import PromptConfig
from src.configs.retrieve_config import SearchConfig
from src.cores.message_builder import MessageBuilder
from src.db_services.milvus.connection_manager import MilvusConnectionManager
from src.models.embedding import TextEmbedding
from src.models.llm import LLMWrapper

logger = setup_logger(__name__)


class QueryTransformer:
    def __init__(
        self,
        llm: LLMWrapper,
        message_builder: MessageBuilder,
        embeddings: TextEmbedding,
        db_connection_manager: MilvusConnectionManager,
        prompt_config: Optional[PromptConfig] = None,
    ):
        self.logger = logger
        self.logger.info("初始化查询转换器...")
        self.llm = llm
        self.message_builder = message_builder
        self.embeddings = embeddings
        self.db_connection_manager = db_connection_manager
        self.prompt_config = prompt_config or PromptConfig()
        self.logger.info("查询转换器初始化完成")

    def get_templates(self, mode: str) -> str:
        """获取查询转换模板"""
        self.logger.debug(f"获取 {mode} 模式的查询转换模板")
        templates = {
            "rewrite": self.prompt_config.rewrite_prompt,
            "step_back": self.prompt_config.step_back_prompt,
            "sub_query": self.prompt_config.sub_query_prompt,
            "hyde": self.prompt_config.hyde_prompt,
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
            self.logger.error(
                "HyDE 模式请使用 generate_hyde_vector 或 hyde_search 方法"
            )
            raise ValueError(
                "请使用 generate_hyde_vector 或 hyde_search 方法处理 HyDE 模式"
            )

        try:
            prompt_template = self.get_templates(mode).format(original_query=query)
            self.logger.debug("已构建提示模板")

            messages = self.message_builder.build(
                query, system_prompt_template=prompt_template
            )
            self.logger.debug("已构建消息")

            rewrite_query = self.llm.chat(
                messages,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            ).strip()
            self.logger.info(f"查询转换完成，新查询: {rewrite_query[:100]}...")
            return rewrite_query

        except Exception as e:
            self.logger.exception(f"查询转换失败: {str(e)}")
            raise

    async def generate_hyde_vector(
        self, query: str, num_hypo: int = 3
    ) -> Optional[List[float]]:
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
                messages = self.message_builder.build(
                    query, system_prompt_template=prompt_template
                )
                hypo = self.llm.chat(
                    messages,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                ).strip()
                hypo_docs.append(hypo)
                self.logger.debug(f"假设答案 {i + 1}: {hypo[:100]}...")

            # 生成向量
            self.logger.info("开始生成向量...")
            hypo_vectors = await self.embeddings.aget_embedding(hypo_docs)
            query_embeddings_result = await self.embeddings.aget_embedding(query)
            query_vector = query_embeddings_result[0]

            # 融合向量
            combined_vector = [
                np.mean(np.vstack([*hypo_vectors, query_vector]), axis=0).tolist()
            ]
            self.logger.info("HyDE 向量生成完成")
            self.logger.debug(f"向量维度: {len(combined_vector[0])}")

            return combined_vector[0]

        except Exception as e:
            self.logger.exception(f"HyDE 向量生成失败: {str(e)}")
            return None

    async def hyde_search(
        self, query: str, config: SearchConfig
    ) -> List[Dict[str, Any]]:
        """
        使用 HyDE 向量进行文档检索
        """

        try:
            vector = await self.generate_hyde_vector(query)
            docs = await self.db_connection_manager.asearch(
                query=vector, search_config=config
            )
            return docs

        except Exception as e:
            self.logger.exception(f"HyDE 搜索失败: {str(e)}")
            raise
