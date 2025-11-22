# -*- coding: utf-8 -*-
"""
@Time ： 2025/11/21 15:57
@Auth ： 吕鑫
@File ：thirdpart_config.py
@IDE ：PyCharm
"""
from pydantic import BaseModel

class LangfuseConfig(BaseModel):
    host: str
    public_key: str
    secret_key: str
    LANGFUSE_TRACING_ENVIRONMENT: str = "testing"