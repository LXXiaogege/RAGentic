# -*- coding: utf-8 -*-
"""
@Time ： 2025/5/2 21:20
@Auth ： 吕鑫
@File ：memory.py
@IDE ：PyCharm
"""
from collections import deque
from typing import List, Dict
from src.config.logger_config import setup_logger

# 配置日志记录器
logger = setup_logger(__name__)


class ConversationMemory:
    def __init__(self, window_size: int = 5):
        self.logger = logger
        self.window_size = window_size
        self.memory = deque(maxlen=window_size)
        self.logger.info(f"初始化对话记忆，窗口大小: {window_size}")

    def add(self, question: str, answer: str):
        """添加一轮对话"""
        self.memory.append({
            "user": question,
            "assistant": answer
        })
        self.logger.info(f"添加新对话 - 问题: {question[:50]}...")

    def get_history(self) -> str:
        """格式化返回文本历史"""
        if not self.memory:
            self.logger.debug("对话历史为空")
            return ""
        history = []
        for i, conv in enumerate(self.memory):
            history.append(f"Q{i + 1}: {conv['user']}")
            history.append(f"A{i + 1}: {conv['assistant']}")
        self.logger.debug(f"获取对话历史，共 {len(self.memory)} 轮对话")
        return "\n".join(history)

    def get_messages(self) -> List[Dict[str, str]]:
        """导出为 message 列表，兼容 OpenAI 格式"""
        messages = []
        for conv in self.memory:
            messages.append({"role": "user", "content": conv["user"]})
            messages.append({"role": "assistant", "content": conv["assistant"]})
        self.logger.debug(f"导出消息列表，共 {len(messages)} 条消息")
        return messages

    def clear(self):
        """清除对话历史"""
        self.memory.clear()
        self.logger.info("清除所有对话历史")

    def save(self, path: str):
        """保存到本地 JSON 文件"""
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(list(self.memory), f, ensure_ascii=False, indent=2)
        self.logger.info(f"保存对话历史到文件: {path}")

    def load(self, path: str):
        """从 JSON 文件加载历史"""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.memory = deque(data, maxlen=self.window_size)
        self.logger.info(f"从文件加载对话历史: {path}，加载了 {len(data)} 轮对话")
