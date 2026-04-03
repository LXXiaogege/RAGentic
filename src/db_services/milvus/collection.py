# -*- coding: utf-8 -*-
"""
@Time：2026/3/28
@Auth：RAGentic
@File：collection.py
@IDE：PyCharm

Milvus 集合管理
职责：集合的创建、删除、索引管理、加载/释放
"""

from pymilvus import DataType

from src.configs.logger_config import setup_logger
from src.db_services.milvus.connection import MilvusConnection

logger = setup_logger(__name__)


def auto_index_params(field: dict):
    """
    根据字段类型自动生成索引参数

    Args:
        field: 字段定义字典

    Returns:
        索引参数字典，如果不需要索引则返回 None
    """
    dtype = field["dtype"]
    name = field["name"]

    if dtype == "FLOAT_VECTOR":
        return {
            "index_name": f"{name}_dense_idx",
            "index_type": "AUTOINDEX",
            "metric_type": "COSINE",
            "params": {},
        }

    if dtype == "SPARSE_FLOAT_VECTOR":
        bm25_drop_ratio = field.get("drop_ratio_build", 0.2)
        return {
            "index_name": f"{name}_sparse_idx",
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "IP",
            "params": {"drop_ratio_build": bm25_drop_ratio},
        }

    if field.get("is_primary"):
        return {
            "index_name": f"{name}_pk_idx",
            "index_type": "INVERTED_INDEX",
            "params": {},
        }

    return None


class MilvusCollectionManager:
    """
    Milvus 集合管理器

    负责集合的创建、删除、索引管理和加载/释放
    """

    def __init__(self, connection: MilvusConnection):
        """
        初始化集合管理器

        Args:
            connection: Milvus 连接实例
        """
        self.logger = logger
        self.connection = connection

    @property
    def client(self):
        """获取 Milvus 客户端"""
        return self.connection.client

    def create_collection(self, collection_name: str, schema: dict):
        """
        创建集合

        Args:
            collection_name: 集合名称
            schema: 集合 schema 定义

        Raises:
            ValueError: 集合已存在
        """
        self.logger.info(f"开始创建集合: {collection_name}")

        if self.has_collection(collection_name):
            raise ValueError(
                f"Collection '{collection_name}' 已存在，"
                f"禁止覆盖。请使用一个新的 collection_name 或先手动删除。"
            )

        schema_obj = self.client.create_schema(
            auto_id=schema.get("auto_id", True),
            enable_dynamic_fields=schema.get("enable_dynamic_fields", False),
        )

        index_fields = []
        for field in schema["fields"]:
            name = field["name"]
            dtype = field["dtype"]

            if dtype == "FLOAT_VECTOR":
                schema_obj.add_field(
                    field_name=name, datatype=DataType.FLOAT_VECTOR, dim=field["dim"]
                )
                index_fields.append(("dense", name))

            elif dtype == "SPARSE_FLOAT_VECTOR":
                schema_obj.add_field(
                    field_name=name, datatype=DataType.SPARSE_FLOAT_VECTOR
                )
                index_fields.append(("sparse", name))

            elif dtype == "VARCHAR":
                schema_obj.add_field(
                    field_name=name,
                    datatype=DataType.VARCHAR,
                    max_length=field["max_length"],
                    is_primary=field.get("is_primary", False),
                )
                if field.get("is_primary", False):
                    index_fields.append(("primary", name))

            else:
                raise ValueError(f"不支持的 dtype: {dtype}")

        self.logger.debug(f"已创建 schema: {schema_obj}")

        self.client.create_collection(
            collection_name=collection_name,
            schema=schema_obj,
        )

        for field in schema["fields"]:
            if field.get("is_primary", False):
                self.logger.debug(
                    f"字段 [{field['name']}] 是主键，跳过索引创建（主键会自动创建索引）"
                )
                continue

            idx_params = auto_index_params(field)
            if idx_params:
                self.create_index(collection_name, field["name"], idx_params)

        self.logger.info(f"集合创建成功: {collection_name}")

    def drop_collection(self, collection_name: str):
        """删除集合"""
        self.client.drop_collection(collection_name)
        self.logger.info(f"集合已删除: {collection_name}")

    def has_collection(self, collection_name: str) -> bool:
        """检查集合是否存在"""
        return self.client.has_collection(collection_name)

    def load(self, collection_name: str):
        """加载集合到内存"""
        self.client.load_collection(collection_name)
        self.logger.info(f"集合已加载: {collection_name}")

    def release(self, collection_name: str):
        """释放集合"""
        self.client.release_collection(collection_name)
        self.logger.info(f"集合已释放: {collection_name}")

    def create_index(
        self, collection_name: str, field_name: str, index_params_dict: dict
    ):
        """
        创建索引

        Args:
            collection_name: 集合名称
            field_name: 字段名称
            index_params_dict: 索引参数字典
        """
        self.logger.info(f"为字段 [{field_name}] 创建索引...")

        index_params_obj = self.client.prepare_index_params()

        add_index_kwargs = {
            "field_name": field_name,
            "index_type": index_params_dict.get("index_type"),
            "params": index_params_dict.get("params", {}),
        }

        if "index_name" in index_params_dict:
            add_index_kwargs["index_name"] = index_params_dict["index_name"]

        if "metric_type" in index_params_dict:
            add_index_kwargs["metric_type"] = index_params_dict["metric_type"]

        index_params_obj.add_index(**add_index_kwargs)

        self.client.create_index(
            collection_name=collection_name, index_params=index_params_obj
        )
        self.logger.info(f"索引创建成功: {collection_name}.{field_name}")
