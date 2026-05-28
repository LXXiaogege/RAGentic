# -*- coding: utf-8 -*-
"""Shared response helpers for model wrappers."""

from __future__ import annotations

from typing import Any


def extract_text_response(response: Any) -> str:
    """Extract assistant text from an OpenAI-compatible response."""
    if not getattr(response, "choices", None):
        raise ValueError("LLM 返回了空的 choices 或 content，请检查模型服务")

    message = getattr(response.choices[0], "message", None)
    content = getattr(message, "content", None)
    if content is None:
        raise ValueError("LLM 返回了空的 choices 或 content，请检查模型服务")
    return content

