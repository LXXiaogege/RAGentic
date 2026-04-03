# -*- coding: utf-8 -*-
"""
@Time：2026/3/28
@Auth：RAGentic
@File：connection.py
@IDE：PyCharm

Milvus 连接管理
职责：MilvusClient 创建、连接生命周期管理、数据库切换
"""

from typing import Optional

from pymilvus import MilvusClient

from src.configs.database_config import MilvusConfig
from src.configs.logger_config import setup_logger

logger = setup_logger(__name__)


class MilvusConnection:
    """
    Milvus 连接管理器

    负责 MilvusClient 的创建、数据库切换和连接关闭
    """

    def __init__(self, config: MilvusConfig):
        """
        初始化 Milvus 连接

        Args:
            config: Milvus 配置
        """
        self.logger = logger
        self.config = config
        self._client: Optional[MilvusClient] = None
        self._connect()

    def _connect(self):
        """建立 Milvus 连接"""
        self.logger.info("初始化 Milvus 客户端...")
        try:
            if self.config.milvus_mode == "local":
                self._client = MilvusClient(self.config.vector_db_uri)
            else:
                self._client = MilvusClient(
                    uri=self.config.vector_db_uri,
                    user=self.config.db_user,
                    password=self.config.db_password,
                    db_name=self.config.db_name,
                )
            self.logger.info("Milvus 客户端初始化成功")
        except Exception as e:
            if "milvus-lite" in str(e):
                raise RuntimeError(
                    "Milvus 本地连接失败：缺少 `milvus_lite` 依赖。"
                    "请执行：`pip install 'pymilvus[milvus_lite]'`，"
                    "或将配置 `milvus_mode` 改为 `remote`（并确保远端 Milvus 可用）。"
                ) from e
            raise

    @property
    def client(self) -> MilvusClient:
        """获取 Milvus 客户端实例"""
        if self._client is None:
            self._connect()
        return self._client

    @property
    def db_name(self) -> str:
        """获取当前数据库名称"""
        return self.config.db_name

    def create_database(self, db_name: str):
        """创建数据库"""
        self.client.create_database(db_name)

    def drop_database(self, db_name: str):
        """删除数据库"""
        self.client.drop_database(db_name)

    def list_databases(self):
        """列出所有数据库"""
        return self.client.list_databases()

    def use_database(self, db_name: str):
        """切换数据库"""
        self.client.use_database(db_name)

    def check_connection(self) -> bool:
        """检查连接是否正常"""
        try:
            return self.client is not None and self.client.list_databases() is not None
        except Exception as e:
            self.logger.error(f"连接检查失败: {e}")
            return False

    def close(self):
        """关闭连接"""
        if self._client:
            self._client.close()
            self._client = None
            self.logger.info("Milvus 连接已关闭")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
