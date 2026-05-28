# -*- coding: utf-8 -*-
"""API Server cache behavior tests."""

import pytest
from unittest.mock import AsyncMock

import api_server


class FakePipeline:
    cleanup_calls = 0

    def __init__(self, config):
        self.config = config

    async def cleanup(self):
        FakePipeline.cleanup_calls += 1


class FakeAskPipeline(FakePipeline):
    async def ask(self, **kwargs):
        return {
            "answer": "可信答案",
            "context": "上下文",
            "kb_context": "知识库上下文",
            "tool_context": "",
            "answer_basis": "kb",
            "sources": [{"text": "知识片段", "title": "文档 A", "score": 0.9}],
            "tool_traces": [],
            "memory_hits": [],
        }


@pytest.fixture(autouse=True)
def clear_pipeline_cache(monkeypatch):
    api_server._pipeline_cache.clear()
    FakePipeline.cleanup_calls = 0
    monkeypatch.setattr(api_server, "LangGraphQAPipeline", FakePipeline)
    yield
    api_server._pipeline_cache.clear()


@pytest.mark.asyncio
async def test_pipeline_cache_uses_request_config_key():
    first = await api_server.get_pipeline(
        session_id="s1",
        use_memory=False,
        top_k=3,
        use_sparse=False,
        use_reranker=False,
        enable_think=False,
    )
    second = await api_server.get_pipeline(
        session_id="s1",
        use_memory=True,
        top_k=3,
        use_sparse=False,
        use_reranker=False,
        enable_think=False,
    )

    assert first is not second
    assert len(api_server._pipeline_cache) == 2


@pytest.mark.asyncio
async def test_clear_session_cleans_all_config_variants():
    await api_server.get_pipeline("s1", False, 3, False, False, False)
    await api_server.get_pipeline("s1", True, 3, False, False, False)

    response = await api_server.clear_session("s1")

    assert response["status"] == "ok"
    assert len(api_server._pipeline_cache) == 0
    assert FakePipeline.cleanup_calls == 2


@pytest.mark.asyncio
async def test_pipeline_cache_evicts_overflow(monkeypatch):
    monkeypatch.setattr(api_server, "PIPELINE_CACHE_MAX_SIZE", 1)

    first = await api_server.get_pipeline("s1", False, 3, False, False, False)
    second = await api_server.get_pipeline("s2", False, 3, False, False, False)

    assert first is not second
    assert len(api_server._pipeline_cache) == 1
    assert FakePipeline.cleanup_calls == 1


@pytest.mark.asyncio
async def test_ask_response_includes_trust_fields(monkeypatch):
    monkeypatch.setattr(
        api_server,
        "get_pipeline",
        AsyncMock(return_value=FakeAskPipeline(None)),
    )

    response = await api_server.ask(
        api_server.AskRequest(
            query="测试",
            session_id="s1",
            use_memory=False,
            allow_model_fallback=False,
        )
    )

    assert response.answer_basis == "kb"
    assert response.sources[0]["text"] == "知识片段"
    assert response.tool_traces == []
    assert response.memory_hits == []
