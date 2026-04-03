# -*- coding: utf-8 -*-
"""
@Time：2026/3/27
@Auth：AI Assistant
@File：hybrid_memory_service.py
@IDE：PyCharm

混合记忆服务 - 统一管理短期记忆(LangGraph Checkpoint)和长期记忆(Mem0)
纯 Async 架构版本
"""

import asyncio
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
        self._init_lock = asyncio.Lock()

        self._conversation_count = 0
        self._pending_stm_messages: List[Dict[str, str]] = []

        # 加载 MemorySettings（包含 max_memories, memory_ttl_days）
        from src.configs.memory_settings import MemorySettings

        memory_settings = MemorySettings()
        self.max_memories = memory_settings.max_memories
        self.memory_ttl_days = memory_settings.memory_ttl_days

        logger.info(
            f"初始化混合记忆服务: user_id={user_id}, "
            f"stm_window={stm_window_size}, ltm_threshold={ltm_persist_threshold}, "
            f"enable_stm={enable_stm}, enable_ltm={enable_ltm}, "
            f"max_memories={self.max_memories}, memory_ttl_days={self.memory_ttl_days}"
        )

    async def initialize(self) -> None:
        """初始化所有记忆组件 - 简化版本"""
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            if self.enable_stm:
                self._stm = ShortTermMemory(window_size=self.stm_window_size)
                self._stm.set_context(user_id=self.user_id, thread_id=self.user_id)
                # 4.2: 初始化时尝试从 checkpoint 恢复 STM
                self._stm.load_from_checkpoint(config={})
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

    def _clear_checkpoint(self) -> None:
        """清除 STM checkpoint"""
        if not self.enable_stm or not self._stm:
            return
        try:
            from langgraph.checkpoint.memory import MemorySaver

            checkpointer = MemorySaver()
            checkpoint_config = {
                "configurable": {
                    "thread_id": self._stm._thread_id or "default",
                    "user_id": self._stm._user_id,
                }
            }
            checkpointer.delete(config=checkpoint_config)
            logger.debug("STM checkpoint 已清除")
        except Exception as e:
            logger.warning(f"清除 checkpoint 失败: {e}")

    async def _cleanup_expired_memories(self, user_id: str) -> int:
        """删除已过期的 LTM 记忆，返回删除数量"""
        if not self.enable_ltm:
            return 0

        all_memories = await self._ensure_ltm().get_history(user_id=user_id, limit=1000)
        if not all_memories:
            return 0

        from datetime import datetime, timezone, timedelta

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=self.memory_ttl_days)
        deleted_count = 0

        for mem in all_memories:
            created_at_str = mem.get("created_at")
            if not created_at_str:
                continue
            try:
                # Mem0 返回的 created_at 可能是 ISO 格式字符串
                if isinstance(created_at_str, str):
                    created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                else:
                    created_at = created_at_str
                if created_at < cutoff:
                    mem_id = mem.get("id")
                    if mem_id:
                        await self._ensure_ltm().delete(mem_id)
                        deleted_count += 1
            except Exception as e:
                logger.warning(f"解析记忆创建时间失败: {created_at_str}, {e}")

        if deleted_count > 0:
            logger.info(f"清理了 {deleted_count} 条过期记忆")
        return deleted_count

    async def _consolidate_memories(self, user_id: str) -> None:
        """当记忆数量超过 max_memories 时，对最旧的记忆进行摘要整合"""
        if not self.enable_ltm:
            return

        all_memories = await self._ensure_ltm().get_history(user_id=user_id, limit=1000)
        if len(all_memories) <= self.max_memories:
            return

        # 按创建时间排序，取最旧的 N 条（超出 max_memories 的部分）
        sorted_memories = sorted(
            all_memories,
            key=lambda m: m.get("created_at", ""),
        )
        excess_count = len(all_memories) - self.max_memories
        old_memories = sorted_memories[:excess_count]

        if not old_memories:
            return

        logger.info(f"触发记忆摘要: {len(old_memories)} 条旧记忆将合并为摘要")

        # 构建摘要文本
        memory_texts = []
        for mem in old_memories:
            text = mem.get("text", "")
            if text:
                memory_texts.append(text)
        if not memory_texts:
            return

        original_text = "\n---\n".join(memory_texts)

        # 调用 LLM 进行 summarization
        summary = await self._summarize_memories(original_text)

        # 删除原记忆
        for mem in old_memories:
            mem_id = mem.get("id")
            if mem_id:
                await self._ensure_ltm().delete(mem_id)

        # 添加摘要记忆
        await self._ensure_ltm().add(
            messages=[{"role": "user", "content": summary}],
            user_id=user_id,
            metadata={"source": "hybrid_memory_service", "type": "consolidated_summary"},
            infer=False,
        )
        logger.info(f"记忆摘要已添加: {len(old_memories)} 条旧记忆合并为 1 条摘要")

    def _filter_expired_memories(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """从检索结果中过滤 TTL 过期的记忆"""
        from datetime import datetime, timezone, timedelta

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=self.memory_ttl_days)

        filtered = []
        for mem in results:
            created_at_str = mem.get("created_at")
            if not created_at_str:
                # 没有创建时间的保留
                filtered.append(mem)
                continue
            try:
                if isinstance(created_at_str, str):
                    created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                else:
                    created_at = created_at_str
                if created_at >= cutoff:
                    filtered.append(mem)
            except Exception:
                # 解析失败时保留
                filtered.append(mem)

        expired_count = len(results) - len(filtered)
        if expired_count > 0:
            logger.debug(f"检索时过滤了 {expired_count} 条过期记忆")
        return filtered

    async def _summarize_memories(self, text: str) -> str:
        """调用 LLM 对记忆文本进行 summarization"""
        from src.models.llm import LLMWrapper
        from src.configs.model_config import LLMConfig

        llm_config = LLMConfig()
        llm = LLMWrapper(config=llm_config)

        prompt = (
            "你是一个记忆摘要助手。请将以下多条记忆合并为一条简洁摘要，"
            "保留关键事实、用户偏好和重要上下文。摘要应该流畅、连贯，不要罗列要点。\n\n"
            f"原始记忆：\n{text}\n\n"
            "请生成简洁摘要："
        )

        messages = [{"role": "user", "content": prompt}]
        result = await llm.achat(model=llm_config.model, messages=messages)
        return result if result else text

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
        if not self._initialized:
            await self.initialize()

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
            # 3.1/3.2: 添加前先清理过期记忆
            await self._cleanup_expired_memories(user_id)

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

                # 2.1: 持久化后检查是否需要摘要整合
                await self._consolidate_memories(user_id)

                # 4.1: 持久化成功后保存 STM checkpoint
                if self.enable_stm and self._stm:
                    self._stm.save_to_checkpoint(config={})

                # 4.3: 清理 checkpoint（成功后删除）
                self._clear_checkpoint()

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
        if not self._initialized:
            await self.initialize()

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
        # 3.3: 过滤 TTL 过期记忆
        results = self._filter_expired_memories(results)
        return results

    def get_stm_messages(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """获取短期记忆消息

        Args:
            limit: 消息数量限制

        Returns:
            消息列表
        """
        if not self.enable_stm or not self._initialized:
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
        if not self._initialized:
            await self.initialize()

        result = {
            "stm": [],
            "ltm": [],
        }

        if include_stm and self.enable_stm and self._stm:
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

        if self.enable_ltm and self._ltm:
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
        if not self._initialized:
            await self.initialize()

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

            # 持久化后检查是否需要摘要整合
            await self._consolidate_memories(user_id)

            # 4.1: 持久化成功后保存 STM checkpoint
            if self.enable_stm and self._stm:
                self._stm.save_to_checkpoint(config={})

            # 4.3: 清理 checkpoint
            self._clear_checkpoint()

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
            "max_memories": self.max_memories,
            "memory_ttl_days": self.memory_ttl_days,
            "pending_stm_messages": len(self._pending_stm_messages),
            "conversation_count": self._conversation_count,
            "initialized": self._initialized,
        }
