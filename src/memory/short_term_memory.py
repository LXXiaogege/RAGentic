# -*- coding: utf-8 -*-
"""
@Time：2026/3/27
@Auth：AI Assistant
@File：short_term_memory.py
@IDE：PyCharm
"""

from collections import deque
from typing import Any, Dict, List, Optional

from langgraph.checkpoint.memory import MemorySaver

from src.configs.logger_config import setup_logger

logger = setup_logger(__name__)


class ShortTermMemory:
    """短期记忆 - 基于滑动窗口 + LangGraph Checkpoint"""

    def __init__(
        self,
        window_size: int = 10,
        checkpointer: Optional[MemorySaver] = None,
    ):
        self.window_size = window_size
        self._messages: deque = deque(maxlen=window_size)
        self.checkpointer = checkpointer or MemorySaver()
        self._user_id: Optional[str] = None
        self._thread_id: Optional[str] = None
        logger.info(f"初始化短期记忆，窗口大小: {window_size}")

    def set_context(self, user_id: str, thread_id: str) -> None:
        """设置上下文信息用于 Checkpoint"""
        self._user_id = user_id
        self._thread_id = thread_id

    def add(self, messages: List[Dict[str, str]]) -> None:
        """添加消息到短期记忆"""
        if not messages:
            return
        for msg in messages:
            if msg.get("role") and msg.get("content"):
                self._messages.append(msg)
        logger.debug(
            f"添加 {len(messages)} 条消息到短期记忆，当前窗口: {len(self._messages)}"
        )

    def get_recent(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """获取最近的 N 条消息"""
        if limit is None:
            return list(self._messages)
        return list(self._messages)[-limit:]

    def get_history(self) -> str:
        """格式化返回文本历史（兼容旧接口）"""
        if not self._messages:
            return ""
        history = []
        for i, msg in enumerate(self._messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            history.append(f"{role}: {content}")
        return "\n".join(history)

    def get_messages_for_llm(self) -> List[Dict[str, str]]:
        """导出为 LLM 消息格式（不包含 system 消息）"""
        return [msg for msg in self._messages if msg.get("role") != "system"]

    def clear(self) -> None:
        """清除短期记忆"""
        self._messages.clear()
        logger.info("清除所有短期记忆")

    def save_to_checkpoint(self, config: Dict[str, Any]) -> None:
        """保存状态到 checkpoint"""
        try:
            checkpoint_config = {
                "configurable": {
                    "thread_id": self._thread_id or "default",
                    "user_id": self._user_id,
                }
            }
            self.checkpointer.put(
                config=checkpoint_config,
                checkpoint={"messages": list(self._messages)},
            )
            logger.debug("短期记忆已保存到 checkpoint")
        except Exception as e:
            logger.warning(f"保存 checkpoint 失败: {e}")

    def load_from_checkpoint(self, config: Dict[str, Any]) -> None:
        """从 checkpoint 加载状态"""
        try:
            checkpoint_config = {
                "configurable": {
                    "thread_id": self._thread_id or "default",
                    "user_id": self._user_id,
                }
            }
            checkpoint = self.checkpointer.get(config=checkpoint_config)
            if checkpoint and "messages" in checkpoint:
                self._messages = deque(checkpoint["messages"], maxlen=self.window_size)
                logger.debug(f"从 checkpoint 加载了 {len(self._messages)} 条消息")
        except Exception as e:
            logger.warning(f"加载 checkpoint 失败: {e}")

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return f"ShortTermMemory(window_size={self.window_size}, messages={len(self._messages)})"
