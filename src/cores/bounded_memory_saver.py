# -*- coding: utf-8 -*-
"""
@Time：2026/4/3
@Auth：吕鑫
@File：bounded_memory_saver.py
@IDE：PyCharm

Bounded MemorySaver wrapper for LangGraph - enforces max checkpoint limit with FIFO eviction.
"""

import asyncio
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata

from src.configs.logger_config import setup_logger

logger = setup_logger(__name__)


class BoundedMemorySaver(BaseCheckpointSaver):
    """
    A wrapper around LangGraph's MemorySaver that enforces a maximum number
    of checkpoints per thread using FIFO eviction.
    """

    def __init__(self, max_checkpoints: int = 1000):
        """
        Initialize BoundedMemorySaver.

        Args:
            max_checkpoints: Maximum number of checkpoints to retain per thread.
                           Older checkpoints are evicted FIFO when limit is exceeded.
        """
        super().__init__()
        self._saver = MemorySaver()
        self._max_checkpoints = max_checkpoints
        self._lock = asyncio.Lock()
        logger.info(f"BoundedMemorySaver initialized with max_checkpoints={max_checkpoints}")

    def _get_thread_id_from_config(self, config: Dict[str, Any]) -> str:
        """Extract thread_id from config."""
        return config.get("configurable", {}).get("thread_id", "")

    def _get_checkpoint_ns_from_config(self, config: Dict[str, Any]) -> str:
        """Extract checkpoint_ns from config."""
        return config.get("configurable", {}).get("checkpoint_ns", "")

    def _evict_oldest(self, config: Dict[str, Any]) -> None:
        """Evict oldest checkpoint for a thread if limit exceeded (must hold lock)."""
        thread_id = self._get_thread_id_from_config(config)
        checkpoint_ns = self._get_checkpoint_ns_from_config(config)
        storage = self._saver.storage

        if thread_id not in storage:
            return

        ns_data = storage[thread_id]
        if checkpoint_ns not in ns_data:
            return

        checkpoints = ns_data[checkpoint_ns]
        if isinstance(checkpoints, dict) and len(checkpoints) > self._max_checkpoints:
            # FIFO: pop the oldest (first) item
            oldest_key = next(iter(checkpoints))
            del checkpoints[oldest_key]
            logger.warning(
                f"BoundedMemorySaver: evicted oldest checkpoint for thread={thread_id}, "
                f"ns={checkpoint_ns}, key={oldest_key}, remaining={len(checkpoints)}"
            )

    async def aput(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Dict[str, Any],
    ) -> None:
        """Async put - delegates to MemorySaver after enforcing bound."""
        thread_id = self._get_thread_id_from_config(config)
        async with self._lock:
            await self._saver.aput(config, checkpoint, metadata, new_versions)
            self._evict_oldest(config)
        logger.debug(f"BoundedMemorySaver: stored checkpoint for thread={thread_id}")

    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Dict[str, Any],
    ) -> None:
        """Sync put - delegates to MemorySaver after enforcing bound."""
        thread_id = self._get_thread_id_from_config(config)
        self._saver.put(config, checkpoint, metadata, new_versions)
        self._evict_oldest(config)
        logger.debug(f"BoundedMemorySaver: stored checkpoint for thread={thread_id}")

    @property
    def storage(self):
        """Access underlying storage for inspection (read-only)."""
        return self._saver.storage

    # Delegate all other methods directly to the internal MemorySaver
    async def aput_writes(
        self,
        config: Dict[str, Any],
        writes: List[Tuple[str, Any]],
        task_id: Optional[str] = None,
    ) -> None:
        """Async put writes - delegates to MemorySaver."""
        await self._saver.aput_writes(config, writes, task_id)

    async def aget(
        self, config: Dict[str, Any]
    ) -> Optional[Checkpoint]:
        """Async get - delegates to MemorySaver."""
        return await self._saver.aget(config)

    async def aget_tuple(
        self, config: Dict[str, Any]
    ) -> Optional[Tuple[Checkpoint, CheckpointMetadata]]:
        """Async get tuple - delegates to MemorySaver."""
        return await self._saver.aget_tuple(config)

    async def adelete_thread(self, thread_id: str) -> None:
        """Async delete thread - delegates to MemorySaver."""
        await self._saver.adelete_thread(thread_id)

    async def alist(
        self,
        config: Optional[Dict[str, Any]] = None,
        before: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[Tuple[Checkpoint, CheckpointMetadata]]:
        """Async list - delegates to MemorySaver."""
        return await self._saver.alist(config, before, limit)

    def put_writes(
        self,
        config: Dict[str, Any],
        writes: List[Tuple[str, Any]],
        task_id: Optional[str] = None,
    ) -> None:
        """Sync put writes - delegates to MemorySaver."""
        self._saver.put_writes(config, writes, task_id)

    def get(
        self, config: Dict[str, Any]
    ) -> Optional[Checkpoint]:
        """Sync get - delegates to MemorySaver."""
        return self._saver.get(config)

    def get_tuple(
        self, config: Dict[str, Any]
    ) -> Optional[Tuple[Checkpoint, CheckpointMetadata]]:
        """Sync get tuple - delegates to MemorySaver."""
        return self._saver.get_tuple(config)

    def delete_thread(self, thread_id: str) -> None:
        """Sync delete thread - delegates to MemorySaver."""
        self._saver.delete_thread(thread_id)

    def list(
        self,
        config: Optional[Dict[str, Any]] = None,
        before: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[Tuple[Checkpoint, CheckpointMetadata]]:
        """Sync list - delegates to MemorySaver."""
        return self._saver.list(config, before, limit)

    def get_next_version(
        self, current_version: Optional[str], channel: Dict[str, Any]
    ) -> str:
        """Get next version - delegates to MemorySaver."""
        return self._saver.get_next_version(current_version, channel)
