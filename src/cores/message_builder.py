# -*- coding: utf-8 -*-
"""
@Time    : 2025/4/23
@Author  : 吕鑫
@File    : message_builder.py
@Desc    : 构造多轮消息的统一模块
"""

from typing import List, Dict, Optional, Any
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

    def build(self, query: str, context: str = "", system_prompt_template: Optional[str] = None,
              stm: List[Dict[str, str]] = None, ltm: List = None) -> List[Dict[str, str]]:
        """
        构建完整的对话消息列表
        """
        # 1. 系统提示词 (System Prompt)
        messages = [{
            "role": "system",
            "content": system_prompt_template
        }]
        # 2. 长期记忆 (Long-Term Memory)
        if ltm:
            memory_content = "关于用户的长期记忆信息：\n" + "\n".join([f"- {m}" for m in ltm])
            messages.append({"role": "system", "content": memory_content})
        # 3. 短期记忆: 当前会话的最近 N 轮对话。
        if stm:
            messages.extend(stm)
        # 4. 检索上下文与当前查询 (Context + Query),为了防止模型产生幻觉，通常将 Context 和 Query 组合在最后一条 User 消息中。
        if context:
            final_user_content = f"""基于以下参考信息回答问题（如果信息不足请说明）：
        ### 参考信息：
        {context}
        ### 用户问题：
        {query}"""
        else:
            final_user_content = query
        messages.append({
            "role": "user",
            "content": final_user_content
        })
        return messages
