# -*- coding: utf-8 -*-
"""MessageBuilder 单元测试"""

import pytest
from unittest.mock import Mock, patch

from src.configs.retrieve_config import MessageBuilderConfig
from src.cores.message_builder import MessageBuilder


@pytest.fixture
def config():
    """测试配置"""
    config = MessageBuilderConfig(
        message_builder_model="gpt-3.5-turbo",
        message_max_tokens=3500,
    )
    return config


class TestMessageBuilder:
    """MessageBuilder 测试类"""

    def test_init(self, config):
        """测试初始化"""
        builder = MessageBuilder(config)

        assert builder.model_name == "gpt-3.5-turbo"
        assert builder.tokenizer is not None

    def test_build_with_no_context(self, config):
        """测试无上下文构建"""
        builder = MessageBuilder(config)

        result = builder.build(query="测试问题")

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert "测试问题" in result[1]["content"]

    def test_build_with_context(self, config):
        """测试带上下文构建"""
        builder = MessageBuilder(config)

        result = builder.build(query="测试问题", context="相关文档内容")

        assert len(result) == 2
        assert "相关文档内容" in result[1]["content"]
        assert "测试问题" in result[1]["content"]

    def test_build_with_stm(self, config):
        """测试带短期记忆构建"""
        builder = MessageBuilder(config)
        stm = [
            {"role": "user", "content": "之前的问题"},
            {"role": "assistant", "content": "之前的回答"},
        ]

        result = builder.build(query="新问题", stm=stm)

        assert len(result) == 4
        assert result[1]["content"] == "之前的问题"
        assert result[2]["content"] == "之前的回答"

    def test_build_with_ltm(self, config):
        """测试带长期记忆构建"""
        builder = MessageBuilder(config)
        ltm = ["用户喜欢咖啡", "用户住在上海"]

        result = builder.build(query="新问题", ltm=ltm)

        assert len(result) == 3
        assert "长期记忆" in result[1]["content"]
        assert "用户喜欢咖啡" in result[1]["content"]

    def test_num_tokens(self, config):
        """测试 token 计数"""
        builder = MessageBuilder(config)
        text = "这是一个测试文本"

        tokens = builder.num_tokens(text)

        assert tokens > 0
        assert isinstance(tokens, int)


@pytest.mark.skip(reason="build_context 方法不存在于 MessageBuilder")
class TestMessageBuilderBuildContext:
    """build_context 方法测试（已废弃）"""

    def test_build_context_empty(self, config):
        """测试空上下文构建"""
        builder = MessageBuilder(config)
        result = builder.build_context([])
        assert result == ""


@pytest.mark.skip(reason="get_system_prompt 方法不存在于 MessageBuilder")
class TestMessageBuilderSystemPrompt:
    """系统提示测试（已废弃）"""

    def test_get_system_prompt_rag_enabled(self, config):
        """测试 RAG 模式系统提示"""
        config.use_rag_system_prompt = True
        builder = MessageBuilder(config)
        prompt = builder.get_system_prompt(use_rag=True)
        assert prompt == config.rag_system_prompt
