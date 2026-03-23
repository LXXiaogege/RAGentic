# -*- coding: utf-8 -*-
"""
@Time：2025/11/21 15:57
@Auth：吕鑫
@File：thirdpart_config.py
@IDE：PyCharm
"""

from pydantic import BaseModel, Field


class LangfuseConfig(BaseModel):
    host: str = Field(default="", description="Langfuse host URL")
    public_key: str = Field(default="", description="Langfuse public key")
    secret_key: str = Field(default="", description="Langfuse secret key")
    LANGFUSE_TRACING_ENVIRONMENT: str = Field(
        default="testing", description="Langfuse 环境"
    )
