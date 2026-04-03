# -*- coding: utf-8 -*-
"""
@Time：2026/3/28
@Auth：RAGentic
@File：base.py
@IDE：PyCharm

向量数据库抽象接口定义
定义 VectorStore 和 RetrievalPipeline 核心接口
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, TypedDict

from src.configs.retrieve_config import SearchConfig


class SearchResult(TypedDict):
    """搜索结果数据结构"""

    text: str
    score: float
    id: str
    metadata: Optional[Dict[str, Any]]


class VectorStore(ABC):
    """
    向量数据库抽象接口

    定义所有向量存储实现必须遵循的接口契约
    """

    @abstractmethod
    async def asearch(
        self, query: str | List[float], config: SearchConfig
    ) -> List[SearchResult]:
        """
        异步向量搜索

        Args:
            query: 查询字符串或向量
            config: 搜索配置

        Returns:
            SearchResult 列表，按相关性排序
        """
        ...

    @abstractmethod
    async def aadd_documents(self, documents: List[Any]) -> None:
        """
        异步批量添加文档

        Args:
            documents: 文档列表
        """
        ...

    @abstractmethod
    async def adelete(self, ids: List[str]) -> None:
        """
        异步删除文档

        Args:
            ids: 文档ID列表
        """
        ...

    @abstractmethod
    async def aclose(self) -> None:
        """关闭连接"""
        ...

    @abstractmethod
    def build_collection(self) -> None:
        """构建集合（如果不存在则创建）"""
        ...


class RetrievalPipeline(ABC):
    """
    检索流水线抽象接口

    定义多路召回 + 重排序的检索流水线
    """

    @abstractmethod
    async def aretrieve(
        self, query: str | List[float], config: SearchConfig
    ) -> List[Dict[str, Any]]:
        """
        多路召回

        Args:
            query: 查询字符串或向量
            config: 搜索配置

        Returns:
            原始召回结果（未重排）
        """
        ...

    @abstractmethod
    async def arerank(
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
        ...


class DenseEncoder(Protocol):
    """稠密向量编码器协议"""

    async def aget_embedding(self, texts: List[str]) -> List[List[float]]:
        """获取文本的稠密向量表示"""
        ...


class SparseEncoder(Protocol):
    """稀疏向量编码器协议（BM25等）"""

    def encode_queries(self, texts: List[str]) -> Any:
        """编码查询文本为稀疏向量"""
        ...

    def encode_documents(self, texts: List[str]) -> Any:
        """编码文档为稀疏向量"""
        ...


class Reranker(Protocol):
    """重排序器协议"""

    def __call__(self, query: str, texts: List[str]) -> List[Any]:
        """
        对文本列表进行重排

        Args:
            query: 查询字符串
            texts: 文本列表

        Returns:
            带分数的排序结果
        """
        ...
