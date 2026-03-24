# -*- coding: utf-8 -*-
"""
@Time    : 2025/4/23
@Author  : 吕鑫
@File    : message_builder.py
@Desc    : 构造多轮消息的统一模块
"""

import os
from typing import Dict, List, Optional

import tiktoken
from jinja2 import Environment, FileSystemLoader

from src.configs.logger_config import setup_logger
from src.configs.retrieve_config import MessageBuilderConfig

logger = setup_logger(__name__)


class MessageBuilder:
    def __init__(self, config: MessageBuilderConfig):
        self.logger = logger
        self.logger.info("初始化消息构建器...")
        self.config = config
        self.model_name = config.message_builder_model
        self.logger.debug(f"使用模型: {self.model_name}")

        template_dir = (
            config.templates_dir if hasattr(config, "templates_dir") else "templates"
        )
        if not os.path.isabs(template_dir):
            template_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                template_dir,
            )
        env = Environment(loader=FileSystemLoader(template_dir))
        self.template = env.get_template("user_prompt.j2")

        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model_name)
            self.logger.info("成功初始化 tokenizer")
        except Exception as e:
            self.logger.error(f"初始化 tokenizer 失败: {str(e)}")
            raise

    def num_tokens(self, text: str) -> int:
        """计算文本的 token 数量"""
        tokens = len(self.tokenizer.encode(text))
        return tokens

    def build(
        self,
        query: str,
        context: str = "",
        system_prompt_template: Optional[str] = None,
        stm: Optional[List[Dict[str, str]]] = None,
        ltm: Optional[List[str]] = None,
    ) -> List[Dict[str, str]]:
        """
        构建完整的对话消息列表

        Args:
            query: 用户查询
            context: 上下文信息
            system_prompt_template: 系统提示模板
            stm: 短期记忆消息列表
            ltm: 长期记忆内容列表

        Returns:
            构建好的消息列表
        """
        # 1. 系统提示词 (System Prompt)
        messages = [{"role": "system", "content": system_prompt_template}]
        # 2. 长期记忆 (Long-Term Memory)
        if ltm:
            memory_content = "关于用户的长期记忆信息：\n" + "\n".join(
                [f"- {m}" for m in ltm]
            )
            messages.append({"role": "system", "content": memory_content})
        # 3. 短期记忆: 当前会话的最近 N 轮对话。
        if stm:
            messages.extend(stm)
        # 4. 检索上下文与当前查询 (Context + Query),为了防止模型产生幻觉，通常将 Context 和 Query 组合在最后一条 User 消息中。
        final_user_content = self.template.render(query=query, context=context)

        messages.append({"role": "user", "content": final_user_content})
        return messages
