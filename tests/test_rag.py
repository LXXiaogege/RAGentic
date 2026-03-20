# -*- coding: utf-8 -*-
"""
@Time ： 2025/4/25 22:54
@Auth ： 吕鑫
@File ：test_rag.py
@IDE ：PyCharm
"""

from src.configs.config import AppConfig
from langfuse import get_client
import os
from src.cores.pipeline import QAPipeline, _collect_generator
import asyncio

config = AppConfig()
os.environ["LANGFUSE_SECRET_KEY"] = config.langfuse.secret_key
os.environ["LANGFUSE_PUBLIC_KEY"] = config.langfuse.public_key
os.environ["LANGFUSE_HOST"] = config.langfuse.host
os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = "testing"
os.environ["LANGFUSE_TRACING_ENABLED"] = "false"

os.environ["OPENAI_API_KEY"] = config.llm.api_key
os.environ["OPENAI_API_BASE"] = config.llm.base_url

langfuse_client = get_client()
config.retrieve.use_kb = True
pipeline = QAPipeline(config)


async def test_ask():
    result = await _collect_generator(pipeline.ask("大模型是什么"))
    print(result)
    assert "answer" in result or "error" in result


if __name__ == '__main__':
    asyncio.run(test_ask())
