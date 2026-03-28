# -*- coding: utf-8 -*-
"""
@Time ： 2025/11/20 10:19
@Auth ： 吕鑫
@File ：config.py
@IDE ：PyCharm
"""

from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict

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

    model_config = ConfigDict(
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )

    def model_post_init(self, __context):
        """配置加载后处理：加载.env 文件和环境变量并转换路径"""
        self._load_env_vars()
        self._resolve_paths()
        self._validate_paths()

    def _load_env_vars(self):
        """统一加载配置：先加载.env 文件，再加载系统环境变量（覆盖.env）"""
        import os
        from dotenv import dotenv_values

        env_mapping = {
            "LANGFUSE__HOST": ("langfuse", "host"),
            "LANGFUSE__PUBLIC_KEY": ("langfuse", "public_key"),
            "LANGFUSE__SECRET_KEY": ("langfuse", "secret_key"),
            "LLM__API_KEY": ("llm", "api_key"),
            "LLM__BASE_URL": ("llm", "base_url"),
            "EMBEDDING__API_KEY": ("embedding", "api_key"),
            "EMBEDDING__BASE_URL": ("embedding", "base_url"),
            "RERANK__MODEL_PATH": ("reranker", "rerank_model_path"),
            "BM25__MODEL_DIR": ("bm25", "bm25_model_dir"),
            "MILVUS__VECTOR_DB_URI": ("milvus", "vector_db_uri"),
            "MILVUS__MEMORY_DB_URI": ("milvus", "memory_db_uri"),
            "MEM0__HISTORY_DB_PATH": ("memory", "history_db_path"),
            "RETRIEVE__KB_PATH": ("retrieve", "kb_path"),
        }

        def apply_env(key: str, value: str):
            """应用单个环境变量到配置对象"""
            if value is None:
                return
            if key in env_mapping:
                config_name, field_name = env_mapping[key]
                try:
                    config_obj = getattr(self, config_name, None)
                    if config_obj and hasattr(config_obj, field_name):
                        setattr(config_obj, field_name, value)
                except (AttributeError, ValueError):
                    pass
            elif "__" in key:
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
                try:
                    if hasattr(self, key.lower()):
                        setattr(self, key.lower(), value)
                except (AttributeError, ValueError):
                    pass

        if Path(ENV_PATH).exists():
            for key, value in dotenv_values(ENV_PATH).items():
                apply_env(key, value)

        for key in env_mapping.keys():
            if env_val := os.getenv(key):
                apply_env(key, env_val)

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

    def _validate_paths(self):
        """验证所有路径配置是否存在"""
        from src.configs.logger_config import setup_logger

        logger = setup_logger(__name__)

        path_configs = [
            ("Rerank 模型路径", self.reranker.rerank_model_path),
            ("BM25 模型目录", self.bm25.bm25_model_dir),
            ("Milvus 数据库路径", self.milvus.vector_db_uri),
            ("知识库路径", self.retrieve.kb_path),
        ]

        for name, path in path_configs:
            if path and not Path(path).exists():
                logger.warning(f"{name} 不存在：{path}")

    def create_request_config(
        self,
        use_kb: bool = None,
        use_tool: bool = None,
        use_memory: bool = None,
        top_k: int = None,
        use_sparse: bool = None,
        use_reranker: bool = None,
        extra_body: dict = None,
    ) -> "AppConfig":
        """创建请求特定的配置（避免 deepcopy 整个 AppConfig）

        只复制检索相关的设置，共享其他expensive对象（LLM, Milvus连接等）
        """
        new_retrieve = self.retrieve.model_copy(
            update={
                k: v
                for k, v in {
                    "use_kb": use_kb,
                    "use_tool": use_tool,
                    "use_memory": use_memory,
                    "top_k": top_k,
                    "use_sparse": use_sparse,
                    "use_reranker": use_reranker,
                    "extra_body": extra_body,
                }.items()
                if v is not None
            }
        )

        new_config = AppConfig.__new__(AppConfig)
        new_config.debug = self.debug
        new_config.llm = self.llm
        new_config.embedding = self.embedding
        new_config.reranker = self.reranker
        new_config.bm25 = self.bm25
        new_config.retrieve = new_retrieve
        new_config.splitter = self.splitter
        new_config.milvus = self.milvus
        new_config.neo4j = self.neo4j
        new_config.redis = self.redis
        new_config.memory = self.memory
        new_config.prompt = self.prompt
        new_config.message_builder = self.message_builder
        new_config.tools = self.tools
        new_config.evaluation = self.evaluation
        new_config.langfuse = self.langfuse
        return new_config
