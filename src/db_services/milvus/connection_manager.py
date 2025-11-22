# -*- coding: utf-8 -*-
"""
@Time ： 2025/11/16 13:19
@Auth ： 吕鑫
@File ：connection_manager.py
@IDE ：PyCharm
"""
from pymilvus import MilvusClient
from src.configs.retrieve_config import MilvusConfig
from src.db_services.milvus.database_manager import MilvusDBManager
from src.db_services.milvus.collection_manager import MilvusCollectionManager
from src.db_services.milvus.data_service import MilvusDataService
from src.configs.logger_config import setup_logger
from typing import Union, List, Optional, Dict, Any
import numpy as np
from src.db_services.milvus.data_service import SearchSettings

logger = setup_logger(__name__)


class MilvusConnectionManager:
    """
    统一的 Milvus 数据库入口类
    """

    def __init__(self, embeddings, text_splitter, config: MilvusConfig):
        """
        初始化 Milvus 数据库

        Args:
            config: 配置对象
            embeddings: 嵌入模型
            text_splitter: 文本分割器
        """
        self.logger = logger
        self.config = config
        self.embeddings = embeddings
        self.text_splitter = text_splitter

        # 1. 创建 Milvus 客户端
        self.logger.info("初始化 Milvus 客户端...")
        if self.config.milvus_mode == "local":
            self.client = MilvusClient(self.config.vector_db_uri)
        else:
            self.client = MilvusClient(
                uri=self.config.vector_db_uri,
                user=self.config.db_user,
                password=self.config.db_password,
                db_name=self.config.db_name
            )

        # 2. 初始化数据库管理器
        self.logger.info("初始化数据库管理器...")
        self.db_manager = MilvusDBManager(self.client)

        # 如果配置了数据库名称，切换数据库
        if self.config.db_name and self.config.milvus_mode != "local":
            try:
                databases = self.db_manager.list_databases()
                if self.config.db_name not in (databases or []):
                    self.db_manager.create_database(self.config.db_name)
                self.db_manager.use_database(self.config.db_name)
            except Exception as e:
                self.logger.warning(f"数据库操作警告: {e}")

        # 3. 初始化集合管理器
        self.logger.info("初始化集合管理器...")
        self.collection_manager = MilvusCollectionManager(self.db_manager)

        # 4. 初始化数据服务
        self.logger.info("初始化数据服务...")
        self.data_service = MilvusDataService(
            collection_manager=self.collection_manager,
            embeddings=self.embeddings,
            text_splitter=self.text_splitter,
            config=self.config,
        )

        # 保持向后兼容的属性
        self.data_processor = self.data_service.data_processor
        self.bm25_func = self.data_service.bm25_func
        self.bm25_model_path = self.data_service.bm25_model_path
        self.rerank_function = self.data_service.rerank_function

        self.logger.info("Milvus 数据库初始化完成（统一入口）")

    # --- Database Level ---
    def create_database(self, name: str):
        self.db_manager.create_database(name)

    def drop_database(self, name: str):
        """删除数据库"""
        self.db_manager.drop_database(name)

    def list_databases(self):
        """列出所有数据库"""
        return self.db_manager.list_databases()

    def use_database(self, name: str):
        """使用数据库"""
        self.db_manager.use_database(name)

    def check_database(self):
        """检查数据库是否正常"""
        return self.db_manager.check_database()

    def close(self):
        """关闭数据库连接"""
        self.db_manager.close()

    # --- Collection Level ---

    def create_collection(self, name: str, schema: dict):
        """创建集合"""
        self.collection_manager.create_collection(name, schema)

    def has_collection(self, name: str) -> bool:
        """检查集合是否存在"""
        return self.collection_manager.has_collection(name)

    def drop_collection(self, name: str):
        """删除集合"""
        self.collection_manager.drop_collection(name)

    def load_collection(self, name: str):
        """加载集合"""
        self.collection_manager.load(name)

    def release_collection(self, name: str):
        """释放集合"""
        self.collection_manager.release(name)

    def build_collection(self):
        """构建集合（如果不存在则创建）"""
        if not self.has_collection(self.config.collection_name):
            self.logger.info(f"集合 {self.config.collection_name} 不存在，开始创建...")
            schema = self._create_default_schema()
            self.collection_manager.create_collection(self.config.collection_name, schema)
            self.collection_manager.load(self.config.collection_name)
            self.logger.info(f"集合 {self.config.collection_name} 创建并加载完成")
        else:
            self.logger.info(f"集合 {self.config.collection_name} 已存在，跳过创建")
            self.load_collection(self.config.collection_name)

    def _create_default_schema(self) -> dict:
        """创建默认的集合 schema"""
        return {
            "auto_id": False,
            "enable_dynamic_fields": False,
            "fields": [
                {
                    "name": "id",
                    "dtype": "VARCHAR",
                    "max_length": 64,
                    "is_primary": True
                },
                {
                    "name": "dense_vec",
                    "dtype": "FLOAT_VECTOR",
                    "dim": self.config.vector_dimension
                },
                {
                    "name": "bm25_vec",
                    "dtype": "SPARSE_FLOAT_VECTOR",
                    "drop_ratio_build": self.config.bm25_drop_ratio
                },
                {
                    "name": "text",
                    "dtype": "VARCHAR",
                    "max_length": self.config.max_text_length
                },
                {
                    "name": "metadata",
                    "dtype": "VARCHAR",
                    "max_length": self.config.max_metadata_length
                }
            ]
        }

    # --- Data Level ---
    def load_bm25_model(self, texts: List[str]):
        """加载/训练 BM25 模型"""
        self.data_service.load_bm25_model(texts)

    def prepare_records(self, documents):
        """将文档转换为结构化记录和 ID"""
        return self.data_service.prepare_records(documents)

    def insert_records(self, records: List[Dict]):
        """插入新向量记录"""
        self.data_service.insert_records(records)

    def update_records(self, records: List[Dict]):
        """更新已有向量记录"""
        self.data_service.update_records(records)

    def delete_by_ids(self, collection_name: str, ids: Optional[Union[list, str, int]]):
        """按ID删除"""
        self.data_service.delete_by_ids(collection_name, ids)

    def delete_by_filter(self, collection_name: str, filter_condition: str):
        """按条件删除"""
        self.data_service.delete_by_filter(collection_name, filter_condition)

    def add_documents_from_dir(self, data_dir: str):
        """从目录中批量添加或更新文档向量"""
        # 确保集合已创建
        self.build_collection()
        # 添加文档
        self.data_service.add_documents_from_dir(data_dir)

    def search(self, query: Union[str, List[float], np.array], k: Optional[int] = None,
               use_sparse: Optional[bool] = None, use_reranker: Optional[bool] = None,
               use_contextualize_embedding: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        搜索向量

        Args:
            query: 查询文本或向量
            k: 返回结果数量
            use_sparse: 是否使用稀疏搜索
            use_reranker: 是否使用重排序
            use_contextualize_embedding: 是否使用上下文嵌入

        Returns:
            搜索结果列表
        """

        # 如果传入了参数，创建 SearchSettings 对象
        settings = SearchSettings(
            k=k,
            use_sparse=use_sparse,
            use_reranker=use_reranker,
            use_contextualize_embedding=use_contextualize_embedding
        )
        return self.data_service.search(query, settings=settings)
