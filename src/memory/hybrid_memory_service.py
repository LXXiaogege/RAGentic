# -*- coding: utf-8 -*-
"""
@Time：2026/3/27
@Auth：AI Assistant
@File：hybrid_memory_service.py
@IDE：PyCharm

混合记忆服务 - 统一管理短期记忆(LangGraph Checkpoint)和长期记忆(Mem0)
"""

from typing import Any, Dict, List, Optional

from src.configs.config import AppConfig
from src.configs.logger_config import setup_logger
from src.memory.base import MemoryService
from src.memory.long_term_memory import LongTermMemory
from src.memory.short_term_memory import ShortTermMemory

logger = setup_logger(__name__)


class HybridMemoryService(MemoryService):
    """混合记忆服务 - 统一管理 STM + LTM

    策略：
    1. 每次对话：消息存入 STM，并判断是否需要刷 LTM
    2. 每次查询：优先从 LTM 检索相关记忆作为上下文
    3. STM 作为当前会话的快速缓存，LLM 直接消费
    4. LTM 作为持久化存储，支持夸会话记忆
    """

    def __init__(
        self,
        config: AppConfig,
        user_id: str,
        stm_window_size: int = 10,
        ltm_persist_threshold: int = 3,
        enable_stm: bool = True,
        enable_ltm: bool = True,
    ):
        self.config = config
        self.user_id = user_id
        self.stm_window_size = stm_window_size
        self.ltm_persist_threshold = ltm_persist_threshold
        self.enable_stm = enable_stm
        self.enable_ltm = enable_ltm

        self._stm: Optional[ShortTermMemory] = None
        self._ltm: Optional[LongTermMemory] = None
        self._initialized = False

        self._conversation_count = 0
        self._pending_stm_messages: List[Dict[str, str]] = []

        logger.info(
            f"初始化混合记忆服务: user_id={user_id}, "
            f"stm_window={stm_window_size}, ltm_threshold={ltm_persist_threshold}, "
            f"enable_stm={enable_stm}, enable_ltm={enable_ltm}"
        )

    async def initialize(self) -> None:
        """初始化所有记忆组件"""
        if self._initialized:
            return

        if self.enable_stm:
            self._stm = ShortTermMemory(window_size=self.stm_window_size)
            self._stm.set_context(user_id=self.user_id, thread_id=self.user_id)
            logger.info("短期记忆初始化完成")

        if self.enable_ltm:
            self._ltm = LongTermMemory(config=self.config, user_id=self.user_id)
            await self._ltm.initialize()
            logger.info("长期记忆初始化完成")

        self._initialized = True
        logger.info("混合记忆服务初始化完成")

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def _ensure_stm(self) -> ShortTermMemory:
        if self._stm is None:
            raise RuntimeError("短期记忆未启用或未初始化")
        return self._stm

    def _ensure_ltm(self) -> LongTermMemory:
        if self._ltm is None:
            raise RuntimeError("长期记忆未启用或未初始化")
        return self._ltm

    async def add(
        self,
        messages: List[Dict[str, str]],
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        persist_immediately: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """添加记忆

        策略：
        1. 消息先存入 STM
        2. 积累到阈值后批量刷入 LTM
        3. 也可以选择立即刷入 LTM

        Args:
            messages: 消息列表
            user_id: 用户ID
            metadata: 额外元数据
            persist_immediately: 是否立即刷入 LTM
            **kwargs: 其他参数
        """
        if not messages:
            return {"results": [], "stm_count": 0, "ltm_persisted": False}

        if self.enable_stm:
            stm = self._ensure_stm()
            stm.add(messages)
            self._pending_stm_messages.extend(messages)
            logger.debug(f"添加 {len(messages)} 条消息到 STM")

        ltm_persisted = False
        result = {"results": [], "stm_count": len(self._pending_stm_messages)}

        if self.enable_ltm:
            should_persist = (
                persist_immediately
                or len(self._pending_stm_messages) >= self.ltm_persist_threshold
            )

            if should_persist and self._pending_stm_messages:
                ltm_result = await self._ensure_ltm().add(
                    messages=self._pending_stm_messages,
                    user_id=user_id,
                    metadata=metadata,
                    infer=True,
                    **kwargs,
                )
                result["results"] = ltm_result.get("results", [])
                self._pending_stm_messages.clear()
                ltm_persisted = True
                self._conversation_count += 1
                logger.info(f"已刷 {len(result['results'])} 条记忆到 LTM")

        result["ltm_persisted"] = ltm_persisted
        return result

    async def search(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
        threshold: Optional[float] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """搜索记忆（从 LTM）

        Args:
            query: 搜索查询
            user_id: 用户ID
            limit: 返回数量限制
            threshold: 相似度阈值
            **kwargs: 其他参数

        Returns:
            记忆条目列表
        """
        if not self.enable_ltm:
            logger.debug("LTM 未启用，返回空搜索结果")
            return []

        results = await self._ensure_ltm().search(
            query=query,
            user_id=user_id,
            limit=limit,
            threshold=threshold,
            **kwargs,
        )
        return results

    def get_stm_messages(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """获取短期记忆消息

        Args:
            limit: 消息数量限制

        Returns:
            消息列表
        """
        if not self.enable_stm:
            return []
        return self._ensure_stm().get_recent(limit=limit)

    async def get_history(
        self,
        user_id: str,
        limit: int = 10,
        include_stm: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """获取历史记忆

        Args:
            user_id: 用户ID
            limit: 返回数量限制
            include_stm: 是否包含短期记忆
            **kwargs: 其他参数

        Returns:
            包含 stm 和 ltm 历史的字典
        """
        result = {
            "stm": [],
            "ltm": [],
        }

        if include_stm and self.enable_stm:
            result["stm"] = self._ensure_stm().get_recent(limit=limit)

        if self.enable_ltm:
            result["ltm"] = await self._ensure_ltm().get_history(
                user_id=user_id,
                limit=limit,
                **kwargs,
            )

        return result

    async def clear(self, user_id: str, **kwargs) -> None:
        """清除所有记忆

        Args:
            user_id: 用户ID
            **kwargs: 其他参数（如 agent_id）
        """
        if self.enable_stm and self._stm:
            self._stm.clear()
            logger.info(f"清除 STM，用户: {user_id}")

        if self.enable_ltm:
            await self._ensure_ltm().clear(user_id=user_id, **kwargs)
            logger.info(f"清除 LTM，用户: {user_id}")

        self._pending_stm_messages.clear()
        self._conversation_count = 0

    async def persist_pending(self, user_id: str, **kwargs) -> Dict[str, Any]:
        """立即刷写待定的 STM 消息到 LTM

        Args:
            user_id: 用户ID
            **kwargs: 其他参数

        Returns:
            刷写结果
        """
        if not self._pending_stm_messages:
            return {"results": [], "count": 0}

        if self.enable_ltm:
            result = await self._ensure_ltm().add(
                messages=self._pending_stm_messages,
                user_id=user_id,
                infer=True,
                **kwargs,
            )
            count = len(self._pending_stm_messages)
            self._pending_stm_messages.clear()
            self._conversation_count += 1
            logger.info(f"强制刷写 {count} 条消息到 LTM")
            return {"results": result.get("results", []), "count": count}

        return {"results": [], "count": 0}

    def format_ltm_for_context(
        self,
        results: List[Dict[str, Any]],
        prefix: str = "【长期记忆】",
    ) -> str:
        """将 LTM 搜索结果格式化为上下文字符串

        Args:
            results: 搜索结果列表
            prefix: 前缀标签

        Returns:
            格式化的上下文字符串
        """
        if not results:
            return ""

        if self._ltm:
            return self._ltm.format_for_context(results)

        formatted_parts = []
        for i, item in enumerate(results):
            text = item.get("text", "")
            if text:
                formatted_parts.append(f"[记忆{i + 1}] {text}")

        if formatted_parts:
            return prefix + "\n" + "\n".join(formatted_parts)
        return ""

    def get_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        return {
            "user_id": self.user_id,
            "enable_stm": self.enable_stm,
            "enable_ltm": self.enable_ltm,
            "stm_window_size": self.stm_window_size,
            "ltm_persist_threshold": self.ltm_persist_threshold,
            "pending_stm_messages": len(self._pending_stm_messages),
            "conversation_count": self._conversation_count,
            "initialized": self._initialized,
        }
