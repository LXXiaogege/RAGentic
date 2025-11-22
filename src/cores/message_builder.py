# -*- coding: utf-8 -*-
"""
@Time    : 2025/4/23
@Author  : 吕鑫
@File    : message_builder.py
@Desc    : 构造多轮消息的统一模块
"""

from typing import List, Dict, Optional
import tiktoken
from src.configs.retrieve_config import MessageBuilderConfig
from src.configs.logger_config import setup_logger

logger = setup_logger(__name__)


class MessageBuilder:
    def __init__(self, config: MessageBuilderConfig):
        self.logger = logger
        self.logger.info("初始化消息构建器...")
        self.config = config
        self.model_name = config.message_builder_model
        self.logger.debug(f"使用模型: {self.model_name}")

        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model_name)
            self.logger.info("成功初始化 tokenizer")
        except Exception as e:
            self.logger.error(f"初始化 tokenizer 失败: {str(e)}")
            raise

    def num_tokens(self, text: str) -> int:
        """计算文本的 token 数量"""
        try:
            tokens = len(self.tokenizer.encode(text))
            self.logger.debug(f"文本 token 数量: {tokens}")
            return tokens
        except Exception as e:
            self.logger.error(f"计算 token 数量失败: {str(e)}")
            raise

    def build(
            self,
            query: str,
            context: str = "",
            system_prompt_template: Optional[str] = None,
            use_memory: bool = None,
            memory_items: Optional[List[Dict[str, str]]] = None,
            no_think: bool = True,
            max_tokens: Optional[int] = None,
            max_history_turns: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        构建完整的对话消息列表
        """
        self.logger.info("开始构建消息...")
        self.logger.debug(f"参数: use_memory={use_memory}, no_think={no_think}, max_history_turns={max_history_turns}")

        # 初始化
        prefix = self.config.message_no_think_prefix if no_think else ""
        context_hint = self._build_context_hint(context)
        system_prompt = self._build_system_prompt(system_prompt_template, prefix, context_hint)
        max_tokens = max_tokens or self.config.message_max_tokens
        self.logger.debug(f"系统提示词长度: {len(system_prompt)} 字符")

        # 初始化消息
        messages = [{"role": "system", "content": system_prompt}]
        total_tokens = self.num_tokens(system_prompt)
        self.logger.debug(f"系统提示词 token 数: {total_tokens}")

        # 添加历史消息
        if use_memory and memory_items:
            self.logger.info(f"添加历史消息，共 {len(memory_items)} 条")
            messages, total_tokens = self._append_memory_messages(
                messages, memory_items, total_tokens, max_tokens, max_history_turns
            )
            self.logger.debug(f"添加历史消息后的总 token 数: {total_tokens}")

        # 添加当前用户问题
        query = query.strip()
        messages.append({"role": "user", "content": query})
        query_tokens = self.num_tokens(query)
        total_tokens += query_tokens
        self.logger.debug(f"用户问题 token 数: {query_tokens}")
        self.logger.info(f"消息构建完成，总 token 数: {total_tokens}")

        return messages

    def _build_context_hint(self, context: str) -> str:
        """构建上下文提示"""
        if context.strip():
            hint = self.config.message_context_hint_template.format(context=context.strip())
            self.logger.debug(f"构建上下文提示，长度: {len(hint)} 字符")
            return hint
        self.logger.debug("无上下文信息")
        return "当前无外部知识库上下文。"

    def _build_system_prompt(self, template: Optional[str], prefix: str, context_hint: str) -> str:
        """构建系统提示词"""
        template = template or self.config.message_system_prompt_template
        try:
            if "{prefix}" in template:
                prompt = template.format(prefix=prefix, context_hint=context_hint)
            else:
                prompt = prefix + template.format(context_hint=context_hint)
            self.logger.debug(f"构建系统提示词，长度: {len(prompt)} 字符")
            return prompt
        except KeyError as e:
            self.logger.error(f"系统提示词模板缺失变量: {e}")
            raise ValueError(f"system_prompt 模板缺失变量：{e}")

    def _append_memory_messages(
            self,
            messages: List[Dict[str, str]],
            memory_items: List[Dict[str, str]],
            total_tokens: int,
            max_tokens: int,
            max_history_turns: Optional[int]
    ) -> (List[Dict[str, str]], int):
        """添加历史消息"""
        history = memory_items[::-1]  # 逆序排列，最新的优先
        if max_history_turns is not None:
            history = history[:max_history_turns]
            self.logger.debug(f"限制历史消息轮数: {max_history_turns}")

        added_turns = 0
        for item in history:
            question = item.get("question", "").strip()
            answer = item.get("answer", "").strip()
            q_tokens = self.num_tokens(question)
            a_tokens = self.num_tokens(answer)
            turn_tokens = q_tokens + a_tokens

            if total_tokens + turn_tokens >= max_tokens:
                self.logger.debug(f"达到 token 限制 ({max_tokens})，停止添加历史消息")
                break

            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
            total_tokens += turn_tokens
            added_turns += 1
            self.logger.debug(f"添加历史消息轮次 {added_turns}，当前总 token 数: {total_tokens}")

        self.logger.info(f"成功添加 {added_turns} 轮历史消息")
        return messages, total_tokens
