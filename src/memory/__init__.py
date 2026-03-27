# -*- coding: utf-8 -*-
"""
@Time：2026/3/27
@Auth：AI Assistant
@File：__init__.py.py
@IDE：PyCharm

Memory 模块 - 分层记忆服务架构
"""

from src.memory.base import MemoryService, ShortTermMemory, LongTermMemory
from src.memory.short_term_memory import ShortTermMemory as ShortTermMemoryImpl
from src.memory.long_term_memory import LongTermMemory as LongTermMemoryImpl
from src.memory.hybrid_memory_service import HybridMemoryService
from src.memory.mem0_manager import Mem0Manager
from src.memory.memory_adapter import adapt_appconfig_to_mem0

__all__ = [
    "MemoryService",
    "ShortTermMemory",
    "LongTermMemory",
    "ShortTermMemoryImpl",
    "LongTermMemoryImpl",
    "HybridMemoryService",
    "Mem0Manager",
    "adapt_appconfig_to_mem0",
]
