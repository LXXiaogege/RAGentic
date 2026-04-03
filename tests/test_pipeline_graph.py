# -*- coding: utf-8 -*-
"""
验证 pipeline graph 结构
"""

from src.cores.pipeline_langgraph import LangGraphQAPipeline
from src.configs.config import AppConfig


def test_pipeline_graph_structure():
    """验证 pipeline graph 可以正确构建"""
    config = AppConfig()
    pipeline = LangGraphQAPipeline(config)

    # 验证 graph 存在
    assert pipeline.graph is not None

    # 验证可以导出 mermaid
    mermaid = pipeline.export_graph()
    assert mermaid is not None
    assert "graph TD" in mermaid

    # 验证关键节点存在
    assert "agent_node" in mermaid
    assert "tools_node" in mermaid
    assert "build_context" in mermaid
    assert "generate_answer" in mermaid
    assert "update_memory" in mermaid
    assert "handle_error" in mermaid

    # 验证 parse_query 已移除
    assert "parse_query" not in mermaid

    print("✓ Graph structure validated")
    print(f"  Nodes: route, agent_node, tools_node, build_context, generate_answer, update_memory, handle_error")


if __name__ == "__main__":
    test_pipeline_graph_structure()
