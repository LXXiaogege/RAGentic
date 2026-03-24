# -*- coding: utf-8 -*-
"""MessageBuilder 单元测试"""

import pytest
from unittest.mock import Mock, patch

from src.configs.retrieve_config import MessageBuilderConfig
from src.cores.message_builder import MessageBuilder


@pytest.fixture
def config():
    """测试配置"""
    config = Mock(spec=MessageBuilderConfig)
    config.max_context_length = 4000
    config.system_prompt = "你是一个助手"
    config.use_rag_system_prompt = True
    config.rag_system_prompt = "你是一个 RAG 助手"
    return config


class TestMessageBuilder:
    """MessageBuilder 测试类"""

    def test_init(self, config):
        """测试初始化"""
        builder = MessageBuilder(config)

        assert builder.max_context_length == 4000
        assert builder.tokenizer is not None

    def test_build_context_empty(self, config):
        """测试空上下文构建"""
        builder = MessageBuilder(config)

        result = builder.build_context([])

        assert result == ""

    def test_build_context_single_doc(self, config):
        """测试单个文档构建"""
        builder = MessageBuilder(config)
        docs = [{"text": "文档内容", "score": 0.9}]

        result = builder.build_context(docs)

        assert "文档内容" in result
        assert "0.9" in result or "相关性" in result

    def test_build_context_multiple_docs(self, config):
        """测试多个文档构建"""
        builder = MessageBuilder(config)
        docs = [
            {"text": "文档 1", "score": 0.9},
            {"text": "文档 2", "score": 0.8},
            {"text": "文档 3", "score": 0.7},
        ]

        result = builder.build_context(docs)

        assert "文档 1" in result
        assert "文档 2" in result
        assert "文档 3" in result

    def test_build_context_truncation(self, config):
        """测试上下文截断"""
        builder = MessageBuilder(config)
        long_text = "这是很长的文本 " * 1000
        docs = [{"text": long_text, "score": 0.9}]

        result = builder.build_context(docs)

        assert len(result) <= config.max_context_length

    def test_format_doc_with_metadata(self, config):
        """测试带元数据的文档格式化"""
        builder = MessageBuilder(config)
        doc = {
            "text": "测试内容",
            "score": 0.95,
            "source": "test.pdf",
            "page": 10,
        }

        result = builder._format_doc(doc)

        assert "测试内容" in result
        assert "test.pdf" in result or "10" in result

    def test_format_doc_minimal(self, config):
        """测试最小文档格式化"""
        builder = MessageBuilder(config)
        doc = {"text": "简单内容"}

        result = builder._format_doc(doc)

        assert "简单内容" in result

    def test_estimate_tokens(self, config):
        """测试 token 估算"""
        builder = MessageBuilder(config)
        text = "这是一个测试文本"

        tokens = builder._estimate_tokens(text)

        assert tokens > 0
        assert isinstance(tokens, int)

    def test_truncate_to_token_limit(self, config):
        """测试 token 限制截断"""
        builder = MessageBuilder(config)
        long_text = "测试 " * 2000

        truncated = builder._truncate_to_token_limit(long_text, max_tokens=100)

        assert len(truncated) < len(long_text)
        assert builder._estimate_tokens(truncated) <= 100


class TestMessageBuilderSystemPrompt:
    """系统提示测试"""

    def test_get_system_prompt_rag_enabled(self, config):
        """测试 RAG 模式系统提示"""
        config.use_rag_system_prompt = True
        builder = MessageBuilder(config)

        prompt = builder.get_system_prompt(use_rag=True)

        assert prompt == config.rag_system_prompt

    def test_get_system_prompt_rag_disabled(self, config):
        """测试非 RAG 模式系统提示"""
        builder = MessageBuilder(config)

        prompt = builder.get_system_prompt(use_rag=False)

        assert prompt == config.system_prompt

    def test_get_system_prompt_custom(self, config):
        """测试自定义系统提示"""
        builder = MessageBuilder(config)
        custom_prompt = "自定义提示"

        prompt = builder.get_system_prompt(use_rag=False, custom_prompt=custom_prompt)

        assert prompt == custom_prompt
