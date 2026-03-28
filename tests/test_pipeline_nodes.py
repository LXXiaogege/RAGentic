# -*- coding: utf-8 -*-
"""Pipeline 节点单元测试 - Async 版本"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock

from src.configs.config import AppConfig
from src.cores.pipeline_langgraph import LangGraphQAPipeline, QAState


@pytest.fixture
def config():
    """测试配置"""
    config = AppConfig()
    config.retrieve.use_kb = False
    config.retrieve.use_tool = False
    config.retrieve.use_memory = False
    config.retrieve.memory_window_size = 5
    config.retrieve.max_agent_iterations = 5
    return config


@pytest.fixture
def mock_components(config):
    """模拟所有组件"""
    components = {}

    components["embeddings"] = Mock()
    components["embeddings"].config = config.embedding

    components["text_splitter"] = Mock()
    components["text_splitter"].chunk_size = config.splitter.chunk_size

    components["message_builder"] = Mock()
    components["message_builder"].build.return_value = [
        {"role": "user", "content": "test"}
    ]

    components["db_manager"] = Mock()
    components["db_manager"].asearch = MagicMock(return_value=[{"text": "mock result"}])

    components["llm"] = Mock()
    components["llm"].config = config.llm
    components["llm"].chat.return_value = "这是模拟答案"

    components["query_transformer"] = Mock()
    components["query_transformer"].transform_query.return_value = "转换后的查询"
    components["query_transformer"].hyde_search = MagicMock(
        return_value=[{"text": "hyde result"}]
    )

    components["mcp_client"] = Mock()

    return components


def create_pipeline_with_mocks(config):
    """创建带完整 mock 的 pipeline 对象"""
    with patch.object(LangGraphQAPipeline, "_init_components", return_value=None):
        with patch.object(LangGraphQAPipeline, "_build_graph", return_value=None):
            pipeline = LangGraphQAPipeline.__new__(LangGraphQAPipeline)
            pipeline.config = config
            pipeline.logger = Mock()
            pipeline._memory_init_lock = asyncio.Lock()
            pipeline._memory_settings = Mock()
            pipeline._memory_settings.enable_ltm = False
            return pipeline


class TestQAPipelineNodes:
    """Pipeline 节点测试 - Async 版本"""

    @pytest.mark.asyncio
    async def test_parse_query_node(self, config, mock_components):
        """测试查询解析节点"""
        pipeline = create_pipeline_with_mocks(config)
        state = QAState(original_query="测试问题")
        result = await pipeline._parse_query(state)

        assert result.original_query == "测试问题"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_agent_node_no_tools_disabled(self, config, mock_components):
        """测试工具禁用时 agent_node 不会被调用"""
        config.retrieve.use_tool = False
        pipeline = create_pipeline_with_mocks(config)

        state = QAState(original_query="测试")
        assert pipeline._should_call_tools(state) == "skip_tools"

    @pytest.mark.asyncio
    async def test_build_context_node(self, config, mock_components):
        """测试上下文构建节点"""
        pipeline = create_pipeline_with_mocks(config)
        pipeline._memory_service = None
        pipeline._memory_initialized = True

        state = QAState(
            original_query="测试",
            kb_context="【知识库检索内容】\n文档 1",
            tool_context="【工具返回内容】\n工具结果",
        )
        result = await pipeline._build_context(state)

        assert result.final_context is not None
        assert "工具结果" in result.final_context or "文档 1" in result.final_context

    @pytest.mark.asyncio
    async def test_update_memory_node_disabled(self, config, mock_components):
        """测试记忆更新节点（禁用）"""
        config.retrieve.use_memory = False
        pipeline = create_pipeline_with_mocks(config)

        state = QAState(original_query="测试", messages=[])
        result = await pipeline._update_memory(state)

        assert result.error is None

    @pytest.mark.asyncio
    async def test_handle_error_node(self, config, mock_components):
        """测试错误处理节点"""
        pipeline = create_pipeline_with_mocks(config)

        state = QAState(original_query="测试", error="测试错误信息")
        result = await pipeline._handle_error(state)

        assert len(result.messages) > 0
        assert "错误" in result.messages[-1].content


class TestQAPipelineConditionalEdges:
    """条件边测试"""

    def test_should_call_tools(self, config):
        """测试工具调用条件判断"""
        pipeline = create_pipeline_with_mocks(config)

        config.retrieve.use_tool = True
        state = QAState(original_query="测试")
        assert pipeline._should_call_tools(state) == "call_agent"

        config.retrieve.use_tool = False
        assert pipeline._should_call_tools(state) == "skip_tools"

    def test_should_continue_agent_loop_with_tool_calls(self, config):
        """测试 agent 循环继续条件 - 有 tool_calls"""
        pipeline = create_pipeline_with_mocks(config)
        config.retrieve.max_agent_iterations = 5

        from langchain_core.messages import AIMessage

        state_with_tool_calls = QAState(
            original_query="测试",
            agent_iteration=3,
            messages=[
                AIMessage(
                    content="test",
                    tool_calls=[
                        {"id": "1", "name": "test", "args": {}, "type": "tool_call"}
                    ],
                )
            ],
        )

        result = pipeline._should_continue_agent_loop(state_with_tool_calls)
        assert result == "has_tool_calls"

    def test_should_continue_agent_loop_without_tool_calls(self, config):
        """测试 agent 循环继续条件 - 无 tool_calls"""
        pipeline = create_pipeline_with_mocks(config)
        config.retrieve.max_agent_iterations = 5

        from langchain_core.messages import AIMessage

        state_without_tool_calls = QAState(
            original_query="测试",
            agent_iteration=3,
            messages=[AIMessage(content="test")],
        )
        result = pipeline._should_continue_agent_loop(state_without_tool_calls)
        assert result == "done"

    def test_should_continue_agent_loop_max_iterations(self, config):
        """测试 agent 循环继续条件 - 达到最大迭代次数"""
        pipeline = create_pipeline_with_mocks(config)
        config.retrieve.max_agent_iterations = 5

        state_max_iterations = QAState(original_query="测试", agent_iteration=5)
        result = pipeline._should_continue_agent_loop(state_max_iterations)
        assert result == "done"

    def test_should_generate_answer_agent_completed(self, config):
        """测试生成答案判断 - agent 已完成"""
        pipeline = create_pipeline_with_mocks(config)
        config.retrieve.use_memory = False

        state = QAState(original_query="测试", agent_iteration=1)
        result = pipeline._should_generate_answer(state)
        assert result == "finish"

    def test_should_generate_answer_no_agent(self, config):
        """测试生成答案判断 - 无 agent 循环"""
        pipeline = create_pipeline_with_mocks(config)

        state = QAState(original_query="测试", agent_iteration=0)
        result = pipeline._should_generate_answer(state)
        assert result == "generate_answer"


class TestQAPipelineIntegration:
    """集成测试"""

    @pytest.mark.asyncio
    async def test_build_graph(self, config):
        """测试图构建"""
        with patch.object(LangGraphQAPipeline, "_init_components", return_value=None):
            pipeline = LangGraphQAPipeline.__new__(LangGraphQAPipeline)
            pipeline.config = config
            pipeline.logger = Mock()
            pipeline._memory_init_lock = asyncio.Lock()
            pipeline._memory_settings = Mock()
            pipeline._memory_settings.enable_ltm = False
            pipeline._define_workflow = Mock()
            pipeline.graph = Mock()
            pipeline.checkpointer = Mock()

            pipeline._build_graph()

            pipeline._define_workflow.assert_called_once()

    def test_export_graph(self, config):
        """测试图导出"""
        pipeline = create_pipeline_with_mocks(config)
        pipeline.graph = Mock()
        pipeline.graph.get_graph.return_value.draw_mermaid.return_value = "mermaid code"

        result = pipeline.export_graph("test.mmd")

        assert result == "mermaid code"


class TestQAState:
    """QAState 模型测试"""

    def test_qa_state_creation(self):
        """测试 QAState 创建"""
        state = QAState(original_query="测试问题")
        assert state.original_query == "测试问题"
        assert state.messages == []
        assert state.error is None
        assert state.agent_iteration == 0

    def test_qa_state_with_messages(self):
        """测试带消息的 QAState"""
        from langchain_core.messages import HumanMessage, AIMessage

        messages = [
            HumanMessage(content="你好"),
            AIMessage(content="你好，我是AI"),
        ]
        state = QAState(original_query="测试", messages=messages)
        assert len(state.messages) == 2
        assert state.original_query == "测试"

    def test_qa_state_with_context(self):
        """测试带上下文的 QAState"""
        state = QAState(
            original_query="测试",
            kb_context="知识库内容",
            tool_context="工具结果",
        )
        assert state.kb_context == "知识库内容"
        assert state.tool_context == "工具结果"
