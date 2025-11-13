# -*- coding: utf-8 -*-
"""
@Time ： 2025/10/10 14:53
@Auth ： 吕鑫
@File ：prompt.py
@IDE ：PyCharm
"""

from src.config.logger_config import setup_logger
from typing import Optional, Union, Literal, Dict, Any

logger = setup_logger(__name__)


class PromptManager:
    """
    通用 Prompt 管理类

    功能特性：
        ✅ 优先从 Langfuse 获取 Prompt（自动缓存 + 回退）
        ✅ 可配置优先策略（langfuse_first / local_first）
        ✅ 自动编译 PromptClient 对象为可用文本
        ✅ 支持 text/chat 两种 Prompt 类型
        ✅ 允许传入本地 fallback prompt
    """

    def __init__(
            self,
            langfuse_client=None,
            config: Optional[object] = None,
            strategy: str = "langfuse_first",
            default_cache_ttl: int = 300,  # 默认缓存 5 分钟
    ):
        """
        初始化 PromptManager

        Args:
            langfuse_client: Langfuse 客户端（可为 None）
            config: 本地配置对象（或 dict）
            strategy: prompt 加载策略 ["langfuse_first" | "local_first"]
            default_cache_ttl: Langfuse 内置缓存 TTL（秒）
        """
        self.logger = logger
        self.langfuse_client = langfuse_client
        self.config = config or {}
        self.strategy = strategy
        self.default_cache_ttl = default_cache_ttl

    # ----------------------------------------------------------------------
    def get_prompt(
            self,
            name: str,
            *,
            version: Optional[int] = None,
            label: Optional[str] = None,
            type: Literal["chat", "text"] = "text",
            fallback: Optional[Union[str, list]] = None,
            cache_ttl_seconds: Optional[int] = None,
            compile_vars: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        获取指定名称的 Prompt。

        Args:
            name: prompt 名称（如 "system_prompt"、"kb_prompt"、"agent_router"）
            version: 指定版本（可选）
            label: 指定标签（可选）
            type: prompt 类型 ("text" 或 "chat")
            fallback: 本地 fallback prompt（可为 str 或 ChatMessage 列表）
            cache_ttl_seconds: 覆盖默认缓存 TTL（秒）
            compile_vars: prompt参数， 不带直接返回原生prompt
        Returns:
            prompt 文本字符串
        """
        ttl = cache_ttl_seconds or self.default_cache_ttl

        # 优先级策略
        if self.strategy == "langfuse_first":
            prompt_text = (
                    self._try_langfuse(name, version, label, type, fallback, ttl, compile_vars)
                    or self._try_local(name, compile_vars)
            )
        else:  # local_first
            prompt_text = (
                    self._try_local(name, compile_vars)
                    or self._try_langfuse(name, version, label, type, fallback, ttl, compile_vars)
            )

        if not prompt_text:
            logger.warning(f"[PromptManager] 未找到有效 prompt: {name}")

        return prompt_text

    def _try_langfuse(
            self,
            name: str,
            version: Optional[int],
            label: Optional[str],
            type: Literal["chat", "text"],
            fallback: Optional[Union[str, list]],
            cache_ttl_seconds: int,
            compile_vars: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        从 Langfuse 获取 prompt
        if type is "chat": returns a message object, e.g., [{"role": "system", "content": "hello"}]
        if type is "text": returns a string, e.g., "xxx"
        """
        if not self.langfuse_client:
            return None
        try:
            prompt_client = self.langfuse_client.get_prompt(
                name=name,
                version=version,
                label=label,
                type=type,
                cache_ttl_seconds=cache_ttl_seconds,
                fallback=fallback,
            )
            # return prompt_client.prompt
            if compile_vars:
                final_prompt = prompt_client.compile(**compile_vars)
            else:
                final_prompt = prompt_client.prompt
            return final_prompt
        except Exception as e:
            logger.error(f"[PromptManager] Langfuse 获取失败: {e}")
        return None

    def _try_local(self, name: str, compile_vars: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """从本地 config 获取（并用 compile_vars 格式化字符串）"""
        if not self.config:
            return None
        local_prompt = self.config.get(name)
        if local_prompt is None or compile_vars is None:
            return local_prompt
        return local_prompt.format(**compile_vars)
