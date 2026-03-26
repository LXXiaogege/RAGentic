# -*- coding: utf-8 -*-
"""Pipeline 节点单元测试"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pydantic import BaseModel

from src.configs.config import AppConfig
from src.cores.pipeline_langgraph import LangGraphQAPipeline, QAState


@pytest.fixture
def config():
    """测试配置"""
    config = AppConfig()
    config.retrieve.use_kb = False
    config.retrieve.use_tool = False
    config.retrieve.use_memory = False
    config.retrieve.use_rewrite = False
    config.retrieve.memory_window_size = 5
    return config


@pytest.fixture
def mock_components(config):
    """模拟所有组件"""
    components = {}

    # 模拟 Embeddings
    components["embeddings"] = Mock()
    components["embeddings"].config = config.embedding

    # 模拟 TextSplitter
    components["text_splitter"] = Mock()
    components["text_splitter"].chunk_size = config.splitter.chunk_size

    # 模拟 MessageBuilder
    components["message_builder"] = Mock()
    components["message_builder"].build.return_value = [
        {"role": "user", "content": "test"}
    ]

    # 模拟 DB Connection Manager
    components["db_manager"] = Mock()
    components["db_manager"].asearch = MagicMock(return_value=[{"text": "mock result"}])

    # 模拟 LLM
    components["llm"] = Mock()
    components["llm"].config = config.llm
    components["llm"].chat.return_value = "这是模拟答案"

    # 模拟 QueryTransformer
    components["query_transformer"] = Mock()
    components["query_transformer"].transform_query.return_value = "转换后的查询"
    components["query_transformer"].hyde_search = MagicMock(
        return_value=[{"text": "hyde result"}]
    )

    # 模拟 MCP Client
    components["mcp_client"] = Mock()

    return components


class TestQAPipelineNodes:
    """Pipeline 节点测试"""

    def test_parse_query_node(self, config, mock_components):
        """测试查询解析节点"""
        with patch.object(LangGraphQAPipeline, "_init_components", return_value=None):
            with patch.object(LangGraphQAPipeline, "_build_graph", return_value=None):
                pipeline = LangGraphQAPipeline.__new__(LangGraphQAPipeline)
                pipeline.config = config
                pipeline.logger = Mock()

                state = QAState(original_query="测试问题")
                result = pipeline._parse_query(state)

                assert result.original_query == "测试问题"
                assert result.error is None

    def test_transform_query_node_rewrite_enabled(self, config, mock_components):
        """测试查询转换节点（启用改写）"""
        config.retrieve.use_rewrite = True

        with patch.object(LangGraphQAPipeline, "_init_components", return_value=None):
            with patch.object(LangGraphQAPipeline, "_build_graph", return_value=None):
                pipeline = LangGraphQAPipeline.__new__(LangGraphQAPipeline)
                pipeline.config = config
                pipeline.logger = Mock()
                pipeline.query_transformer = mock_components["query_transformer"]

                state = QAState(original_query="如何学习 Python？")
                result = pipeline._transform_query(state)

                assert result.transformed_query == "转换后的查询"
                mock_components[
                    "query_transformer"
                ].transform_query.assert_called_once()

    def test_transform_query_node_rewrite_disabled(self, config, mock_components):
        """测试查询转换节点（禁用改写）"""
        config.retrieve.use_rewrite = False

        with patch.object(LangGraphQAPipeline, "_init_components", return_value=None):
            with patch.object(LangGraphQAPipeline, "_build_graph", return_value=None):
                pipeline = LangGraphQAPipeline.__new__(LangGraphQAPipeline)
                pipeline.config = config
                pipeline.logger = Mock()

                state = QAState(original_query="测试问题")
                result = pipeline._transform_query(state)

                assert result.transformed_query == ""

    def test_retrieve_knowledge_node_disabled(self, config, mock_components):
        """测试知识检索节点（禁用 KB）"""
        config.retrieve.use_kb = False

        with patch.object(LangGraphQAPipeline, "_init_components", return_value=None):
            with patch.object(LangGraphQAPipeline, "_build_graph", return_value=None):
                pipeline = LangGraphQAPipeline.__new__(LangGraphQAPipeline)
                pipeline.config = config
                pipeline.logger = Mock()

                state = QAState(original_query="测试")
                result = pipeline._retrieve_knowledge(state)

                assert result.kb_context == ""

    def test_retrieve_knowledge_node_enabled(self, config, mock_components):
        """测试知识检索节点（启用 KB）"""
        config.retrieve.use_kb = True

        with patch.object(LangGraphQAPipeline, "_init_components", return_value=None):
            with patch.object(LangGraphQAPipeline, "_build_graph", return_value=None):
                pipeline = LangGraphQAPipeline.__new__(LangGraphQAPipeline)
                pipeline.config = config
                pipeline.logger = Mock()
                pipeline.db_connection_manager = mock_components["db_manager"]
                pipeline.query_transformer = mock_components["query_transformer"]

                state = QAState(
                    original_query="RAG 是什么", transformed_query="RAG 技术"
                )
                result = pipeline._retrieve_knowledge(state)

                assert (
                    "知识库检索内容" in result.kb_context
                    or "未检索到" in result.kb_context
                )

    def test_agent_node_no_tools_disabled(self, config, mock_components):
        """测试工具禁用时 agent_node 不会被调用（_should_call_tools 路由到 build_context）"""
        config.retrieve.use_tool = False

        with patch.object(LangGraphQAPipeline, "_init_components", return_value=None):
            with patch.object(LangGraphQAPipeline, "_build_graph", return_value=None):
                pipeline = LangGraphQAPipeline.__new__(LangGraphQAPipeline)
                pipeline.config = config
                pipeline.logger = Mock()

                state = QAState(original_query="测试")
                assert pipeline._should_call_tools(state) == "skip_tools"

    def test_build_context_node(self, config, mock_components):
        """测试上下文构建节点"""
        with patch.object(LangGraphQAPipeline, "_init_components", return_value=None):
            with patch.object(LangGraphQAPipeline, "_build_graph", return_value=None):
                pipeline = LangGraphQAPipeline.__new__(LangGraphQAPipeline)
                pipeline.config = config
                pipeline.logger = Mock()

                state = QAState(
                    original_query="测试",
                    kb_context="【知识库检索内容】\n文档 1",
                    tool_context="【工具返回内容】\n工具结果",
                )
                result = pipeline._build_context(state)

                assert result.final_context is not None
                assert (
                    "工具结果" in result.final_context
                    or "文档 1" in result.final_context
                )

    def test_agent_node_generates_answer(self, config, mock_components):
        """测试 agent_node 无 tool_calls 时直接生成答案"""
        from unittest.mock import MagicMock
        from langchain_core.messages import AIMessage

        with patch.object(LangGraphQAPipeline, "_init_components", return_value=None):
            with patch.object(LangGraphQAPipeline, "_build_graph", return_value=None):
                pipeline = LangGraphQAPipeline.__new__(LangGraphQAPipeline)
                pipeline.config = config
                pipeline.logger = Mock()
                pipeline.llm_caller = mock_components["llm"]

                # mock LLM 返回无 tool_calls 的响应
                mock_response = MagicMock()
                mock_response.choices[0].message.tool_calls = None
                mock_response.choices[0].message.content = "这是模拟答案"
                mock_components["llm"].chat.return_value = mock_response

                # mock MCP client
                pipeline.mcp_client = mock_components["mcp_client"]
                pipeline.mcp_client._convert_mcp_tools_to_openai_format.return_value = []

                state = QAState(original_query="测试问题", final_context="上下文信息")
                result = pipeline._agent_node(state)

                assert len(result.messages) > 0
                assert result.error is None
                ai_msgs = [m for m in result.messages if isinstance(m, AIMessage)]
                assert len(ai_msgs) > 0

    def test_update_memory_node_enabled(self, config, mock_components):
        """测试记忆更新节点（启用）"""
        config.retrieve.use_memory = True

        from langchain_core.messages import HumanMessage, AIMessage

        with patch.object(LangGraphQAPipeline, "_init_components", return_value=None):
            with patch.object(LangGraphQAPipeline, "_build_graph", return_value=None):
                pipeline = LangGraphQAPipeline.__new__(LangGraphQAPipeline)
                pipeline.config = config
                pipeline.logger = Mock()

                # 创建超过限制的消息历史
                state = QAState(
                    original_query="测试",
                    messages=[HumanMessage(content=f"消息{i}") for i in range(15)],
                )
                result = pipeline._update_memory(state)

                # 验证消息被截断到 memory_window_size
                assert len(result.messages) <= config.retrieve.memory_window_size

    def test_update_memory_node_disabled(self, config, mock_components):
        """测试记忆更新节点（禁用）"""
        config.retrieve.use_memory = False

        with patch.object(LangGraphQAPipeline, "_init_components", return_value=None):
            with patch.object(LangGraphQAPipeline, "_build_graph", return_value=None):
                pipeline = LangGraphQAPipeline.__new__(LangGraphQAPipeline)
                pipeline.config = config
                pipeline.logger = Mock()

                state = QAState(original_query="测试", messages=[])
                result = pipeline._update_memory(state)

                assert result.error is None

    def test_handle_error_node(self, config, mock_components):
        """测试错误处理节点"""
        with patch.object(LangGraphQAPipeline, "_init_components", return_value=None):
            with patch.object(LangGraphQAPipeline, "_build_graph", return_value=None):
                pipeline = LangGraphQAPipeline.__new__(LangGraphQAPipeline)
                pipeline.config = config
                pipeline.logger = Mock()

                state = QAState(original_query="测试", error="测试错误信息")
                result = pipeline._handle_error(state)

                assert len(result.messages) > 0
                assert "错误" in result.messages[-1].content


class TestQAPipelineConditionalEdges:
    """条件边测试"""

    def test_should_transform_query(self, config):
        """测试查询转换条件判断"""
        with patch.object(LangGraphQAPipeline, "_init_components", return_value=None):
            with patch.object(LangGraphQAPipeline, "_build_graph", return_value=None):
                pipeline = LangGraphQAPipeline.__new__(LangGraphQAPipeline)
                pipeline.config = config

                # 启用改写
                config.retrieve.use_rewrite = True
                state = QAState(original_query="测试")
                assert pipeline._should_transform_query(state) == "transform"

                # 禁用改写
                config.retrieve.use_rewrite = False
                assert pipeline._should_transform_query(state) == "skip_transform"

                # 有错误
                state.error = "错误"
                assert pipeline._should_transform_query(state) == "error"

    def test_should_retrieve_knowledge(self, config):
        """测试知识检索条件判断"""
        with patch.object(LangGraphQAPipeline, "_init_components", return_value=None):
            with patch.object(LangGraphQAPipeline, "_build_graph", return_value=None):
                pipeline = LangGraphQAPipeline.__new__(LangGraphQAPipeline)
                pipeline.config = config

                # 启用 KB
                config.retrieve.use_kb = True
                state = QAState(original_query="测试")
                assert pipeline._should_retrieve_knowledge(state) == "do_retrieve"

                # 禁用 KB
                config.retrieve.use_kb = False
                assert pipeline._should_retrieve_knowledge(state) == "skip_retrieve"

    def test_should_call_tools(self, config):
        """测试工具调用条件判断"""
        with patch.object(LangGraphQAPipeline, "_init_components", return_value=None):
            with patch.object(LangGraphQAPipeline, "_build_graph", return_value=None):
                pipeline = LangGraphQAPipeline.__new__(LangGraphQAPipeline)
                pipeline.config = config

                # 启用工具 → 路由到 agent_node
                config.retrieve.use_tool = True
                state = QAState(original_query="测试")
                assert pipeline._should_call_tools(state) == "call_agent"

                # 禁用工具
                config.retrieve.use_tool = False
                assert pipeline._should_call_tools(state) == "skip_tools"
