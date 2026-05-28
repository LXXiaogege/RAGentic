# -*- coding: utf-8 -*-
"""LLM model wrapper tests."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.configs.model_config import LLMConfig
from src.models.llm import LLMWrapper, OpenAICompatibleLLM
from src.models.llm_litellm import LiteLLMClient
from src.models.message_adapter import MessageAdapter


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
    kwargs = client._prepare_kwargs({"extra_body": {"foo": "bar"}})

    assert kwargs == {"foo": "bar"}
    assert client.api_key == "key"
    assert client.base_url == "https://example.com"

