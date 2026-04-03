# -*- coding: utf-8 -*-
"""
@Time：2026/3/28
@Auth：RAGentic
@File：retrieval.py
@IDE：PyCharm

Milvus 检索流水线
职责：多路召回（稠密+稀疏）、混合归一、重排序
"""

import asyncio
from typing import Any, Dict, List, Union

import numpy as np
from langfuse.decorators import observe
from pymilvus import AnnSearchRequest, RRFRanker

from src.configs.logger_config import setup_logger
from src.configs.retrieve_config import SearchConfig
from src.db_services.base import RetrievalPipeline, SearchResult
from src.db_services.milvus.collection import MilvusCollectionManager
from src.db_services.milvus.data import MilvusDataService
from src.models.embedding import TextEmbedding

logger = setup_logger(__name__)


class HybridRetrievalPipeline(RetrievalPipeline):
    """
    混合检索流水线

    组合稠密检索、稀疏检索（BM25）、RRF 归一化和 CrossEncoder 重排序
    """

    def __init__(
        self,
        collection_manager: MilvusCollectionManager,
        data_service: MilvusDataService,
        embeddings: TextEmbedding,
        search_config: SearchConfig,
    ):
        self.logger = logger
        self.collection_manager = collection_manager
        self.data_service = data_service
        self.embeddings = embeddings
        self.search_config = search_config

        self._dense_retriever = None
        self._sparse_retriever = None
        self._reranker = None

    @property
    def bm25_func(self):
        return self.data_service.bm25_func

    @property
    def rerank_function(self):
        return self.data_service.rerank_function

    @property
    def collection_name(self):
        return self.data_service.milvus_config.collection_name

    async def aretrieve(
        self, query: Union[str, List[float], np.ndarray], config: SearchConfig
    ) -> List[Dict[str, Any]]:
        """
        多路召回

        Args:
            query: 查询字符串或向量
            config: 搜索配置

        Returns:
            原始召回结果
        """
        dense_vec = await self._get_dense_vector(query)

        sparse_vec = None
        if config.use_sparse:
            if self.bm25_func is None:
                self.logger.warning(
                    "配置要求 use_sparse，但当前未启用 BM25，将改为仅 dense 检索"
                )
            else:
                self.logger.debug("生成查询的稀疏向量...")
                sparse_vec = self.bm25_func.encode_queries([query])

        def run_search():
            req_list = []

            dense_search_param = AnnSearchRequest(
                data=[dense_vec],
                anns_field="dense_vec",
                param={"metric_type": "COSINE"},
                limit=config.top_k * config.search_multiplier,
            )
            req_list.append(dense_search_param)

            if config.use_sparse and sparse_vec is not None:
                sparse_search_param = AnnSearchRequest(
                    data=[sparse_vec],
                    anns_field="bm25_vec",
                    param={"metric_type": "IP"},
                    limit=config.top_k * config.search_multiplier,
                )
                req_list.append(sparse_search_param)

            if len(req_list) > 1:
                return self.collection_manager.client.hybrid_search(
                    self.collection_name,
                    req_list,
                    RRFRanker(),
                    config.top_k,
                    output_fields=["id", "text", "metadata"],
                )
            else:
                return self.collection_manager.client.search(
                    self.collection_name,
                    data=[dense_vec],
                    anns_field="dense_vec",
                    limit=config.top_k,
                    output_fields=["id", "text", "metadata"],
                )

        docs = await asyncio.to_thread(run_search)
        return docs

    async def _get_dense_vector(
        self, query: Union[str, List[float], np.ndarray]
    ) -> np.ndarray:
        """获取稠密向量"""
        if isinstance(query, str):
            dense_vec = (await self.embeddings.aget_embedding(query))[0]
        elif isinstance(query, (list, np.ndarray)):
            dense_vec = np.array(query).flatten()
        else:
            raise TypeError("query must be a string, list of floats, or numpy array")
        return dense_vec

    def rerank(
        self, query: str, docs: List[Dict[str, Any]], config: SearchConfig
    ) -> List[Dict[str, Any]]:
        """
        重排序

        Args:
            query: 查询字符串
            docs: 召回结果
            config: 搜索配置

        Returns:
            重排后的结果
        """
        if self.rerank_function is None:
            raise RuntimeError("当前环境不支持 reranker")

        rerank_texts = []
        for hit in docs[0]:
            if config.use_contextualize_embedding:
                rerank_texts.append(
                    f"{hit['entity']['text']}\n\n{hit['entity']['context']}"
                )
            else:
                rerank_texts.append(hit["entity"]["text"])

        scores = self.rerank_function(query, rerank_texts)
        reranked = sorted(zip(docs[0], scores), key=lambda x: x[1].score, reverse=True)
        docs[0] = [item[0] for item in reranked[: config.top_k]]
        return docs

    async def arerank(
        self, query: str, docs: List[Dict[str, Any]], config: SearchConfig
    ) -> List[Dict[str, Any]]:
        """异步重排序"""
        self.logger.info("开始异步重排序...")
        reranked_docs = await asyncio.to_thread(self.rerank, query, docs, config)
        self.logger.info("异步重排序完成")
        return reranked_docs

    @observe(name="HybridRetrieval.search", as_type="retriever")
    async def search(
        self, query: Union[str, List[float], np.ndarray], config: SearchConfig
    ) -> List[SearchResult]:
        """
        完整搜索流程：召回 + 重排

        Args:
            query: 查询字符串或向量
            config: 搜索配置

        Returns:
            SearchResult 列表
        """
        self.logger.info("开始搜索...")

        docs = await self.aretrieve(query, config)

        if config.use_reranker:
            if self.rerank_function is None:
                self.logger.warning(
                    "配置要求 use_reranker，但当前未启用 reranker，将跳过 rerank"
                )
                docs = docs[: config.top_k]
            else:
                docs = await self.arerank(query, docs, config)
        else:
            docs = docs[: config.top_k]

        results = []
        if not docs or not docs[0]:
            self.logger.warning("搜索未返回任何结果")
            return results

        for hit in docs[0][: config.top_k]:
            entity = hit["entity"]
            results.append(
                SearchResult(
                    text=entity.get("text", ""),
                    score=hit["score"] if "score" in hit else None,
                    id=entity["id"],
                    metadata=entity.get("metadata"),
                )
            )

        self.logger.info(f"搜索完成，返回 {len(results)} 条结果")
        return results
