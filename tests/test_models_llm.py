# -*- coding: utf-8 -*-
"""LLM model wrapper tests."""

import pytest
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.configs.model_config import LLMConfig
from src.models.llm import LLMWrapper, OpenAICompatibleLLM
from src.models.llm_litellm import LiteLLMClient
from src.models.message_adapter import MessageAdapter
from src.models.response_utils import extract_text_response


def test_message_adapter_converts_tool_messages():
    messages = [
        HumanMessage(content="你好"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "kb_search",
                    "args": {"query": "RAG"},
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content="结果", tool_call_id="call_1"),
    ]

    converted = MessageAdapter.to_openai_messages(messages)

    assert converted[0] == {"role": "user", "content": "你好"}
    assert converted[1]["role"] == "assistant"
    assert converted[1]["tool_calls"][0]["function"]["name"] == "kb_search"
    assert converted[2] == {
        "role": "tool",
        "content": "结果",
        "tool_call_id": "call_1",
    }


def test_llm_wrapper_supports_openai_compatible_provider():
    config = LLMConfig(provider="openai", model="gpt-4o", api_key="key")
    wrapper = LLMWrapper(config)

    assert isinstance(wrapper.client, OpenAICompatibleLLM)
    assert wrapper.client.provider == "openai"


def test_llm_wrapper_supports_minimax_provider():
    config = LLMConfig(
        provider="minimax",
        model="minimax/MiniMax-Text-01",
        api_key="key",
        base_url="https://api.minimax.chat/v1",
    )
    wrapper = LLMWrapper(config)

    assert isinstance(wrapper.client, OpenAICompatibleLLM)
    assert wrapper.client.provider == "minimax"
    assert wrapper.client._normalize_model("minimax/MiniMax-Text-01") == "MiniMax-Text-01"


def test_minimax_openai_compatible_keeps_thinking_in_extra_body():
    client = OpenAICompatibleLLM(api_key="key", base_url="https://example.com", provider="minimax")

    kwargs = client._prepare_kwargs(
        {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}
    )

    assert kwargs["extra_body"] == {"thinking": True}


def test_litellm_client_uses_per_request_auth_without_global_state():
    client = LiteLLMClient(api_key="key", base_url="https://example.com", provider="minimax")
    model, kwargs = client._prepare_request(
        "minimax/MiniMax-Text-01",
        {"extra_body": {"foo": "bar"}},
    )

    assert model == "MiniMax-Text-01"
    assert kwargs["foo"] == "bar"
    assert kwargs["custom_llm_provider"] == "openai"
    assert kwargs["drop_params"] is True
    assert kwargs["api_key"] == "key"
    assert kwargs["api_base"] == "https://example.com"
    assert client.api_key == "key"
    assert client.base_url == "https://example.com"


def test_extract_text_response_raises_on_empty_content():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
    )

    with pytest.raises(ValueError, match="LLM 返回了空的 choices"):
        extract_text_response(response)


def test_cache_key_changes_with_model_and_context():
    wrapper = LLMWrapper(LLMConfig(provider="openai", model="gpt-4o", api_key="key"))
    base_messages = [{"role": "user", "content": "什么是 RAG？"}]
    context_messages = [
        {"role": "system", "content": "只基于知识库回答"},
        {"role": "user", "content": "什么是 RAG？"},
    ]

    base_key = wrapper._build_cache_key("gpt-4o", base_messages, {})
    model_key = wrapper._build_cache_key("gpt-4.1", base_messages, {})
    context_key = wrapper._build_cache_key("gpt-4o", context_messages, {})

    assert base_key != model_key
    assert base_key != context_key
