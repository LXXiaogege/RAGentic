# -*- coding: utf-8 -*-
"""
@Time ： 2025/4/25 22:54
@Auth ： 吕鑫
@File ：test_rag.py
@IDE ：PyCharm
"""
from src.config.config import QAPipelineConfig
from langfuse import get_client
import os
from src.cores.pipeline import QAPipeline

config = QAPipelineConfig()
os.environ["LANGFUSE_SECRET_KEY"] = config.langfuse_config.get("secret_key")
os.environ["LANGFUSE_PUBLIC_KEY"] = config.langfuse_config.get("public_key")
os.environ["LANGFUSE_HOST"] = config.langfuse_config.get("host")  # eu cloud
os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = "testing"  # 区分 生产、测试等环境
os.environ["LANGFUSE_TRACING_ENABLED"] = "false"  # 启用或禁用 Langfuse 客户端

langfuse_client = get_client()
pipeline = QAPipeline(config)
# pipeline.build_knowledge_base(config.knowledge_dir)

print(pipeline.ask("请问大模型是什么，简单概述下", use_knowledge_base=False, is_tool=False, no_think=True))
