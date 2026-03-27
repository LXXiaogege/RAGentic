# -*- coding: utf-8 -*-
"""
@Time：2026/3/27
@Auth：AI Assistant
@File：long_term_memory.py
@IDE：PyCharm
"""

from typing import Any, Dict, List, Optional

from src.configs.config import AppConfig
from src.configs.logger_config import setup_logger
from src.memory.mem0_manager import Mem0Manager

logger = setup_logger(__name__)


class LongTermMemory:
    """长期记忆 - 封装 Mem0Manager"""

    def __init__(
        self,
        config: AppConfig,
        user_id: Optional[str] = None,
    ):
        self.config = config
        self.default_user_id = user_id
        self._mem0_manager: Optional[Mem0Manager] = None
        self._initialized = False
        logger.info("初始化长期记忆管理器")

    async def initialize(self) -> None:
        """初始化 Mem0 客户端"""
        if self._initialized:
            return
        try:
            self._mem0_manager = Mem0Manager(
                config=self.config,
                user_id=self.default_user_id,
            )
            await self._mem0_manager.init_memory_client()
            self._initialized = True
            logger.info("长期记忆客户端初始化完成")
        except Exception as e:
            logger.error(f"初始化长期记忆客户端失败: {e}")
            raise

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def _ensure_initialized(self) -> None:
        if not self._initialized or self._mem0_manager is None:
            raise RuntimeError("长期记忆客户端尚未初始化，请先调用 await initialize()")

    async def add(
        self,
        messages: List[Dict[str, str]],
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """添加长期记忆

        Args:
            messages: 消息列表
            user_id: 用户ID
            metadata: 额外元数据
            infer: 是否使用 LLM 提取事实
            **kwargs: 其他参数（如 agent_id, run_id 等）
        """
        self._ensure_initialized()
        final_user_id = user_id or self.default_user_id

        mem_metadata = metadata or {}
        mem_metadata["source"] = "hybrid_memory_service"

        result = await self._mem0_manager.add(
            messages=messages,
            user_id=final_user_id,
            metadata=mem_metadata,
            infer=infer,
            agent_id=kwargs.get("agent_id"),
            run_id=kwargs.get("run_id"),
        )
        logger.debug(f"添加长期记忆，用户: {final_user_id}，结果: {result}")
        return result

    async def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 5,
        threshold: Optional[float] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """搜索长期记忆

        Args:
            query: 搜索查询
            user_id: 用户ID
            limit: 返回数量限制
            threshold: 相似度阈值
            **kwargs: 其他参数（如 filters 等）
        """
        self._ensure_initialized()
        final_user_id = user_id or self.default_user_id

        result = await self._mem0_manager.search(
            query=query,
            user_id=final_user_id,
            limit=limit,
            threshold=threshold,
            filters=kwargs.get("filters"),
        )
        results = result.get("results", [])
        logger.debug(
            f"搜索长期记忆，用户: {final_user_id}，查询: {query}，结果数: {len(results)}"
        )
        return results

    async def get_history(
        self,
        user_id: Optional[str] = None,
        limit: int = 10,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """获取历史记忆"""
        self._ensure_initialized()
        final_user_id = user_id or self.default_user_id

        results = await self._mem0_manager.get_all(
            user_id=final_user_id,
            limit=limit,
            agent_id=kwargs.get("agent_id"),
        )
        logger.debug(f"获取历史记忆，用户: {final_user_id}，结果数: {len(results)}")
        return results

    async def delete(
        self,
        memory_id: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """删除指定记忆"""
        self._ensure_initialized()
        result = await self._mem0_manager.delete(memory_id)
        logger.debug(f"删除长期记忆: {memory_id}")
        return result

    async def clear(
        self,
        user_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """清除用户所有记忆"""
        self._ensure_initialized()
        final_user_id = user_id or self.default_user_id
        await self._mem0_manager.delete_all(
            user_id=final_user_id,
            agent_id=kwargs.get("agent_id"),
        )
        logger.info(f"清除用户所有长期记忆: {final_user_id}")

    async def reset(self) -> None:
        """重置所有记忆存储"""
        self._ensure_initialized()
        await self._mem0_manager.reset()
        logger.warning("重置所有长期记忆存储")

    def format_for_context(self, results: List[Dict[str, Any]]) -> str:
        """将搜索结果格式化为上下文字符串

        Args:
            results: 搜索结果列表

        Returns:
            格式化的上下文字符串
        """
        if not results:
            return ""

        formatted_parts = []
        for i, item in enumerate(results):
            text = item.get("text", "")
            if text:
                formatted_parts.append(f"[记忆{i + 1}] {text}")

        return "\n".join(formatted_parts) if formatted_parts else ""
