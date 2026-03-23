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
from src.configs.memory_config import Mem0Config
from src.configs.model_config import (
    LLMConfig,
    EmbeddingConfig,
    RerankConfig,
    BM25Config,
)
from src.configs.prompt_config import PromptConfig
from src.configs.database_config import MilvusConfig, Neo4jConfig, RedisConfig
from src.configs.retrieve_config import (
    SplitterConfig,
    MessageBuilderConfig,
    SearchConfig,
)
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
    embedding: EmbeddingConfig = Field(
        default_factory=EmbeddingConfig, description="嵌入模型配置"
    )
    reranker: RerankConfig = Field(
        default_factory=RerankConfig, description="reranker 配置"
    )
    bm25: BM25Config = Field(default_factory=BM25Config, description="bm25 配置")

    # 检索配置
    retrieve: SearchConfig = Field(default_factory=SearchConfig, description="检索配置")
    splitter: SplitterConfig = Field(
        default_factory=SplitterConfig, description="文本分割配置"
    )

    # 数据库配置
    milvus: MilvusConfig = Field(
        default_factory=MilvusConfig, description="Milvus 向量数据库配置"
    )
    neo4j: Neo4jConfig = Field(
        default_factory=Neo4jConfig, description="Neo4j 图数据库配置"
    )
    redis: RedisConfig = Field(
        default_factory=RedisConfig, description="Redis 数据库配置"
    )

    memory: Mem0Config = Field(default_factory=Mem0Config, description="mem0 配置")
    # 工具和提示词配置
    prompt: PromptConfig = Field(
        default_factory=PromptConfig, description="提示词模板配置"
    )
    message_builder: MessageBuilderConfig = Field(
        default_factory=MessageBuilderConfig, description="消息构建配置"
    )
    tools: ToolsConfig = Field(default_factory=ToolsConfig, description="工具配置")

    evaluation: EvaluationConfig = Field(
        default_factory=EvaluationConfig, description="评估配置"
    )

    langfuse: LangfuseConfig = Field(
        default_factory=LangfuseConfig, description="Langfuse 配置"
    )

    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )

    def model_post_init(self, __context):
        """配置加载后处理：加载.env 文件和环境变量并转换路径"""
        self._load_dotenv()
        self._load_env_paths()
        self._resolve_paths()

    def _load_dotenv(self):
        """手动加载.env 文件（支持嵌套格式 XX__YY）"""
        from dotenv import dotenv_values

        if not Path(ENV_PATH).exists():
            return

        env_vars = dotenv_values(ENV_PATH)

        for key, value in env_vars.items():
            if value is None:
                continue

            # 处理嵌套环境变量 (XX__YY -> xx.yy)
            if "__" in key:
                parts = key.lower().split("__")
                if len(parts) == 2:
                    config_name, field_name = parts
                    try:
                        config_obj = getattr(self, config_name, None)
                        if config_obj and hasattr(config_obj, field_name):
                            setattr(config_obj, field_name, value)
                    except (AttributeError, ValueError):
                        pass
            else:
                # 普通环境变量
                try:
                    if hasattr(self, key.lower()):
                        setattr(self, key.lower(), value)
                except (AttributeError, ValueError):
                    pass

    def _load_env_paths(self):
        """从环境变量加载配置（覆盖默认值和.env 文件）"""
        import os

        # Langfuse 配置
        if env_val := os.getenv("LANGFUSE__HOST"):
            self.langfuse.host = env_val
        if env_val := os.getenv("LANGFUSE__PUBLIC_KEY"):
            self.langfuse.public_key = env_val
        if env_val := os.getenv("LANGFUSE__SECRET_KEY"):
            self.langfuse.secret_key = env_val

        # LLM 配置
        if env_val := os.getenv("LLM__API_KEY"):
            self.llm.api_key = env_val
        if env_val := os.getenv("LLM__BASE_URL"):
            self.llm.base_url = env_val

        # Embedding 配置
        if env_val := os.getenv("EMBEDDING__API_KEY"):
            self.embedding.api_key = env_val
        if env_val := os.getenv("EMBEDDING__BASE_URL"):
            self.embedding.base_url = env_val

        # Rerank 模型路径
        if env_val := os.getenv("RERANK__MODEL_PATH"):
            self.reranker.rerank_model_path = env_val

        # BM25 模型目录
        if env_val := os.getenv("BM25__MODEL_DIR"):
            self.bm25.bm25_model_dir = env_val

        # Milvus 数据库路径
        if env_val := os.getenv("MILVUS__VECTOR_DB_URI"):
            self.milvus.vector_db_uri = env_val

        if env_val := os.getenv("MILVUS__MEMORY_DB_URI"):
            self.milvus.memory_db_uri = env_val

        # Mem0 历史数据库路径
        if env_val := os.getenv("MEM0__HISTORY_DB_PATH"):
            self.memory.history_db_path = env_val

        # 知识库路径
        if env_val := os.getenv("RETRIEVE__KB_PATH"):
            self.retrieve.kb_path = env_val

    def _resolve_paths(self):
        """将所有相对路径转换为相对于 BASE_DIR 的绝对路径"""
        # Rerank 模型路径
        if (
            self.reranker.rerank_model_path
            and not Path(self.reranker.rerank_model_path).is_absolute()
        ):
            self.reranker.rerank_model_path = str(
                (BASE_DIR / self.reranker.rerank_model_path).resolve()
            )

        # BM25 模型目录
        if (
            self.bm25.bm25_model_dir
            and not Path(self.bm25.bm25_model_dir).is_absolute()
        ):
            self.bm25.bm25_model_dir = str(
                (BASE_DIR / self.bm25.bm25_model_dir).resolve()
            )

        # Milvus 数据库路径
        if (
            self.milvus.vector_db_uri
            and not Path(self.milvus.vector_db_uri).is_absolute()
        ):
            self.milvus.vector_db_uri = str(
                (BASE_DIR / self.milvus.vector_db_uri).resolve()
            )

        if (
            self.milvus.memory_db_uri
            and not Path(self.milvus.memory_db_uri).is_absolute()
        ):
            self.milvus.memory_db_uri = str(
                (BASE_DIR / self.milvus.memory_db_uri).resolve()
            )

        # Mem0 历史数据库路径
        if (
            self.memory.history_db_path
            and not Path(self.memory.history_db_path).is_absolute()
        ):
            self.memory.history_db_path = str(
                (BASE_DIR / self.memory.history_db_path).resolve()
            )

        # 知识库路径
        if self.retrieve.kb_path and not Path(self.retrieve.kb_path).is_absolute():
            self.retrieve.kb_path = str((BASE_DIR / self.retrieve.kb_path).resolve())
