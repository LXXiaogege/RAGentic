# -*- coding: utf-8 -*-
"""
@Time ： 2025/9/30 09:31
@Auth ： 吕鑫
@File ：test_langfuse.py
@IDE ：PyCharm
"""
import os
from src.cores.pipeline import QAPipeline
from src.configs.config import AppConfig
from langfuse import get_client

config = AppConfig()
os.environ["LANGFUSE_SECRET_KEY"] = config.langfuse.secret_key
os.environ["LANGFUSE_PUBLIC_KEY"] = config.langfuse.public_key
os.environ["LANGFUSE_HOST"] = config.langfuse.host
os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = config.langfuse.LANGFUSE_TRACING_ENVIRONMENT


langfuse_client = get_client()


def test_langfuse():
    pipeline = QAPipeline(AppConfig(), langfuse_client)
    print(pipeline.ask("请问大模型是什么，简单概述下"))

def prompt_test():
    from src.utils.prompt import PromptManager
    prompt_manager = PromptManager(langfuse_client)
    # print(prompt_manager.get_prompt("system_prompt"))
    print(prompt_manager.get_prompt("kb_system_prompt",compile_vars={"prefix": "/no_think","context_hint":""}))


if __name__ == '__main__':
    # prompt_test()
    test_langfuse()