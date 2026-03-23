# -*- coding: utf-8 -*-
"""QueryTransformer 单元测试"""

import pytest
from unittest.mock import Mock, MagicMock

from src.configs.config import AppConfig
from src.cores.query_transformer import QueryTransformer
from src.cores.message_builder import MessageBuilder
from src.models.llm import LLMWrapper
from src.models.embedding import TextEmbedding


@pytest.fixture
def config():
    """测试配置"""
    config = AppConfig()
    config.retrieve.use_kb = False
    config.retrieve.use_tool = False
    config.retrieve.use_memory = False
    return config


@pytest.fixture
def mock_llm(config):
    """模拟 LLM"""
    llm = Mock(spec=LLMWrapper)
    llm.config = config.llm
    llm.chat.return_value = "mocked response"
    return llm


@pytest.fixture
def mock_embeddings(config):
    """模拟 Embedding"""
    embeddings = Mock(spec=TextEmbedding)
    embeddings.config = config.embedding
    return embeddings


@pytest.fixture
def mock_message_builder(config):
    """模拟 MessageBuilder"""
    builder = Mock(spec=MessageBuilder)
    builder.build.return_value = [{"role": "user", "content": "test"}]
    return builder


@pytest.fixture
def mock_db_manager():
    """模拟数据库管理器"""
    db_manager = Mock()
    db_manager.asearch = MagicMock(return_value=[{"text": "mock result", "score": 0.9}])
    return db_manager


@pytest.fixture
def transformer(mock_llm, mock_message_builder, mock_embeddings, mock_db_manager):
    """创建 QueryTransformer 实例"""
    return QueryTransformer(
        llm_wrapper=mock_llm,
        message_builder=mock_message_builder,
        embeddings=mock_embeddings,
        db_connection_manager=mock_db_manager,
    )


class TestQueryTransformer:
    """QueryTransformer 测试类"""

    def test_transform_query_rewrite(self, transformer, mock_llm):
        """测试查询改写"""
        query = "什么是 RAG？"
        mock_llm.chat.return_value = "检索增强生成技术"

        result = transformer.transform_query(query, mode="rewrite")

        assert result == "检索增强生成技术"
        mock_llm.chat.assert_called_once()

    def test_transform_query_step_back(self, transformer, mock_llm):
        """测试 step-back 查询扩展"""
        query = "RAG 的优势"
        mock_llm.chat.return_value = "检索增强生成相比传统方法的优势"

        result = transformer.transform_query(query, mode="step-back")

        assert result == "检索增强生成相比传统方法的优势"

    def test_transform_query_sub_query(self, transformer, mock_llm):
        """测试子查询分解"""
        query = "RAG 和微调的区别"
        mock_llm.chat.return_value = "1. RAG 是什么 2. 微调是什么"

        result = transformer.transform_query(query, mode="sub-query")

        assert "RAG" in result or "微调" in result

    def test_transform_query_hyde(self, transformer, mock_llm):
        """测试 HyDE 假设文档生成"""
        query = "如何学习机器学习"
        mock_llm.chat.return_value = "机器学习学习指南：首先学习基础概念..."

        result = transformer.transform_query(query, mode="hyde")

        assert "机器学习" in result
        assert len(result) > len(query)

    def test_transform_query_unknown_mode(self, transformer):
        """测试未知模式返回原查询"""
        query = "测试问题"

        result = transformer.transform_query(query, mode="unknown")

        assert result == query

    def test_hyde_search(self, transformer, mock_db_manager):
        """测试 HyDE 检索"""
        query = "测试查询"

        import asyncio

        results = asyncio.run(transformer.hyde_search(query, Mock()))

        assert len(results) > 0
        assert results[0]["text"] == "mock result"
        mock_db_manager.asearch.assert_called_once()


class TestQueryTransformerIntegration:
    """QueryTransformer 集成测试（需要真实配置）"""

    @pytest.mark.skip(reason="需要真实 LLM 配置")
    def test_real_transform_query(self):
        """真实环境测试查询转换"""
        config = AppConfig()
        config.retrieve.use_rewrite = True

        llm = LLMWrapper(config.llm)
        message_builder = MessageBuilder(config.message_builder)
        embeddings = TextEmbedding(config)

        transformer = QueryTransformer(llm, message_builder, embeddings, None)

        result = transformer.transform_query("什么是人工智能？", mode="rewrite")
        assert result != "什么是人工智能？"
        assert len(result) > 0
