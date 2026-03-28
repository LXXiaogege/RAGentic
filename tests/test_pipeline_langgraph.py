# -*- coding: utf-8 -*-
"""
@Time ： 2025/6/8 10:27
@Auth ： 吕鑫
@File：test_pipeline_langgraph.py
@IDE：PyCharm

LangGraph QA Pipeline 异步测试用例
"""

import asyncio
from src.configs.config import AppConfig
from src.cores.pipeline_langgraph import LangGraphQAPipeline


async def test_pipeline_basic():
    """测试 Pipeline 基本功能"""
    config = AppConfig()
    pipeline = LangGraphQAPipeline(config)

    result = await pipeline.ask("你好，请介绍一下你自己")
    print(f"Basic ask result: {result.get('answer', '')[:100]}...")
    assert "error" not in result or not result["error"], (
        f"Unexpected error: {result.get('error')}"
    )
    print("✓ Basic ask test passed")


async def test_pipeline_sync_wrapper():
    """测试同步包装器"""
    config = AppConfig()
    pipeline = LangGraphQAPipeline(config)

    result = pipeline.ask_sync("你好")
    print(f"ask_sync result: {result.get('answer', '')[:100]}...")
    assert "error" not in result or not result["error"], (
        f"Unexpected error: {result.get('error')}"
    )
    print("✓ ask_sync test passed")


async def test_pipeline_with_kb():
    """测试带知识库的 Pipeline"""
    config = AppConfig()
    req_config = config.create_request_config(use_kb=True, top_k=3)
    pipeline = LangGraphQAPipeline(req_config)

    result = await pipeline.ask("请介绍一下这个系统", use_knowledge_base=True)
    print(f"With KB result: {result.get('answer', '')[:100]}...")
    print(f"Context: {result.get('context', '')[:100]}...")
    print("✓ With KB test passed")


async def test_pipeline_batch():
    """测试批量问答"""
    config = AppConfig()
    pipeline = LangGraphQAPipeline(config)

    questions = ["你好", "今天天气怎么样？", "再见"]

    results = await pipeline.batch_ask(questions)
    assert len(results) == 3
    for i, r in enumerate(results):
        print(f"Q{i + 1}: {questions[i]} -> A: {r.get('answer', '')[:50]}...")
    print("✓ Batch ask test passed")


async def test_pipeline_stream():
    """测试流式输出"""
    config = AppConfig()
    pipeline = LangGraphQAPipeline(config)

    events = []
    async for event in pipeline.ask_stream("解释一下什么是 RAG"):
        events.append(event)
        if event.get("status") == "complete":
            print(f"Stream complete: {event.get('answer', '')[:50]}...")

    assert len(events) > 0
    print("✓ Stream test passed")


async def test_export_graph():
    """测试图结构导出"""
    config = AppConfig()
    pipeline = LangGraphQAPipeline(config)

    mermaid_code = pipeline.export_graph("test_pipeline_graph.mmd")
    assert mermaid_code is not None
    print(f"Graph exported: {mermaid_code[:100]}...")
    print("✓ Export graph test passed")


async def main():
    print("=" * 60)
    print("Running Pipeline Tests")
    print("=" * 60)

    try:
        await test_pipeline_basic()
    except Exception as e:
        print(f"✗ Basic test failed: {e}")

    try:
        await test_pipeline_sync_wrapper()
    except Exception as e:
        print(f"✗ Sync wrapper test failed: {e}")

    try:
        await test_pipeline_with_kb()
    except Exception as e:
        print(f"✗ With KB test failed: {e}")

    try:
        await test_pipeline_batch()
    except Exception as e:
        print(f"✗ Batch test failed: {e}")

    try:
        await test_pipeline_stream()
    except Exception as e:
        print(f"✗ Stream test failed: {e}")

    try:
        await test_export_graph()
    except Exception as e:
        print(f"✗ Export graph test failed: {e}")

    print("=" * 60)
    print("All tests completed")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
