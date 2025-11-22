# -*- coding: utf-8 -*-
"""
@Time ： 2025/11/19 17:56
@Auth ： 吕鑫
@File ：tool_config.py
@IDE ：PyCharm
"""

from pydantic import BaseModel, Field
from typing import Dict, List


class ToolParameters(BaseModel):
    type: str
    properties: Dict[str, Dict[str, str]]
    required: List[str]


class ToolFunction(BaseModel):
    name: str
    description: str
    parameters: ToolParameters


class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: ToolParameters


class ToolsConfig(BaseModel):
    enable: bool = False
    tools: List[ToolSchema] = Field(default_factory=list)
