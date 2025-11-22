# -*- coding: utf-8 -*-
"""
@Time ： 2025/11/20 10:19
@Auth ： 吕鑫
@File ：config.py
@IDE ：PyCharm
"""

from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict
from pydantic import Field

from src.configs.evaluate_config import EvaluationConfig
from src.configs.model_config import LLMConfig, EmbeddingConfig
from src.configs.prompt_config import PromptConfig
from src.configs.retrieve_config import SplitterConfig, RewriteConfig, MilvusConfig, MessageBuilderConfig, SearchConfig
from src.configs.thirdpart_config import LangfuseConfig
from src.configs.tool_config import ToolsConfig
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = str(BASE_DIR / ".env")


class AppConfig(BaseSettings):
    """应用配置类"""
    # 基础配置
    debug: bool = Field(default=False, description="调试模式开关")

    # 模型配置
    llm: LLMConfig = Field(default_factory=LLMConfig, description="大语言模型配置")
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig, description="嵌入模型配置")

    # 检索配置
    retrieve: SearchConfig = Field(default_factory=SearchConfig, description="检索配置")
    splitter: SplitterConfig = Field(default_factory=SplitterConfig, description="文本分割配置")
    rewrite: RewriteConfig = Field(default_factory=RewriteConfig, description="查询重写配置")

    # 数据库配置
    milvus: MilvusConfig = Field(default_factory=MilvusConfig, description="Milvus向量数据库配置")

    # 工具和提示词配置
    prompt: PromptConfig = Field(default_factory=PromptConfig, description="提示词模板配置")
    message_builder: MessageBuilderConfig = Field(default_factory=MessageBuilderConfig, description="消息构建配置")
    tools: ToolsConfig = Field(default_factory=ToolsConfig, description="工具配置")

    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig, description="评估配置")

    langfuse: LangfuseConfig = Field(default_factory=LangfuseConfig, description="Langfuse配置")

    model_config = SettingsConfigDict(env_file=ENV_PATH, env_nested_delimiter="__", case_sensitive=False)
