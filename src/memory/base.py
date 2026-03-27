# -*- coding: utf-8 -*-
"""
@Time：2026/3/27
@Auth：AI Assistant
@File：base.py
@IDE：PyCharm
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class MemoryService(ABC):
    """Memory 服务抽象接口"""

    @abstractmethod
    async def add(
        self,
        messages: List[Dict[str, str]],
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """添加记忆

        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            user_id: 用户ID
            metadata: 额外的元数据
            **kwargs: 其他参数（如 agent_id, run_id 等）

        Returns:
            dict: 包含添加结果的字典
        """
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
        threshold: Optional[float] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """搜索相关记忆

        Args:
            query: 搜索查询文本
            user_id: 用户ID
            limit: 返回结果数量限制
            threshold: 相似度阈值
            **kwargs: 其他参数（如 filters 等）

        Returns:
            List[Dict]: 记忆条目列表
        """
        pass

    @abstractmethod
    async def get_history(
        self,
        user_id: str,
        limit: int = 10,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """获取历史记忆

        Args:
            user_id: 用户ID
            limit: 返回结果数量限制
            **kwargs: 其他参数

        Returns:
            List[Dict]: 记忆条目列表
        """
        pass

    @abstractmethod
    async def clear(self, user_id: str, **kwargs) -> None:
        """清除用户记忆

        Args:
            user_id: 用户ID
            **kwargs: 其他参数（如 agent_id, run_id 等）
        """
        pass


class ShortTermMemory(ABC):
    """短期记忆抽象接口"""

    @abstractmethod
    def add(self, messages: List[Dict[str, str]]) -> None:
        """添加消息到短期记忆"""
        pass

    @abstractmethod
    def get_recent(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """获取最近的 N 条消息

        Args:
            limit: 消息数量限制，None 表示全部

        Returns:
            List[Dict]: 消息列表
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """清除短期记忆"""
        pass


class LongTermMemory(ABC):
    """长期记忆抽象接口"""

    @abstractmethod
    async def initialize(self) -> None:
        """初始化长期记忆客户端"""
        pass

    @abstractmethod
    async def add(
        self,
        messages: List[Dict[str, str]],
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """添加长期记忆"""
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
        threshold: Optional[float] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """搜索长期记忆"""
        pass
