# -*- coding: utf-8 -*-
"""
@Time ： 2025/11/15 12:51
@Auth ： 吕鑫
@File ：collection_manager.py
@IDE ：PyCharm
"""
from pymilvus import DataType

from src.configs.logger_config import setup_logger
from src.db_services.milvus.database_manager import MilvusDBManager
logger = setup_logger(__name__)


def auto_index_params(field: dict):
    dtype = field["dtype"]
    name = field["name"]

    # 稠密向量
    if dtype == "FLOAT_VECTOR":
        return {
            "index_name": f"{name}_dense_idx",
            "index_type": "AUTOINDEX",
            "metric_type": "COSINE",
            "params": {}
        }

    # 稀疏向量（BM25）
    if dtype == "SPARSE_FLOAT_VECTOR":
        bm25_drop_ratio = field.get("drop_ratio_build", 0.2)
        return {
            "index_name": f"{name}_sparse_idx",
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "IP",
            "params": {"drop_ratio_build": bm25_drop_ratio}
        }

    # 主键
    if field.get("is_primary"):
        return {
            "index_name": f"{name}_pk_idx",
            "index_type": "INVERTED_INDEX",
            "params": {}
        }

    # 其他字段不创建索引
    return None


class MilvusCollectionManager:
    def __init__(self, db_manager: MilvusDBManager):
        self.logger = logger
        self.client = db_manager.client

    def create_collection(self, collection_name, schema: dict):
        self.logger.info("开始构建集合...")
        # 1) 集合存在检查
        if self.has_collection(collection_name):
            raise ValueError(
                f"Collection '{collection_name}' 已存在，"
                f"禁止覆盖。请使用一个新的 collection_name 或先手动删除。"
            )
        # 2) 创建空 schema
        schema_obj = self.client.create_schema(
            auto_id=schema.get("auto_id", True),
            enable_dynamic_fields=schema.get("enable_dynamic_fields", False)
        )
        # 3) 添加字段到schema
        index_fields = []  # 用于后续自动创建索引
        for field in schema["fields"]:
            name = field["name"]
            dtype = field["dtype"]

            if dtype == "FLOAT_VECTOR":
                schema_obj.add_field(
                    field_name=name,
                    datatype=DataType.FLOAT_VECTOR,
                    dim=field["dim"]
                )
                index_fields.append(("dense", name))

            elif dtype == "SPARSE_FLOAT_VECTOR":
                schema_obj.add_field(
                    field_name=name,
                    datatype=DataType.SPARSE_FLOAT_VECTOR
                )
                index_fields.append(("sparse", name))

            elif dtype == "VARCHAR":
                schema_obj.add_field(
                    field_name=name,
                    datatype=DataType.VARCHAR,
                    max_length=field["max_length"],
                    is_primary=field.get("is_primary", False)
                )
                if field.get("is_primary", False):
                    index_fields.append(("primary", name))

            else:
                raise ValueError(f"不支持的 dtype: {dtype}")
        self.logger.debug(f"已创建 schema: {schema_obj}")

        # 4) 创建 collection
        self.client.create_collection(
            collection_name=collection_name,
            schema=schema_obj,
        )

        # 5) 自动构建索引参数
        # 注意：主键字段会自动创建索引，不需要手动创建
        for field in schema["fields"]:
            # 跳过主键字段（主键会自动创建索引）
            if field.get("is_primary", False):
                self.logger.debug(f"字段 [{field['name']}] 是主键，跳过索引创建（主键会自动创建索引）")
                continue

            idx_params = auto_index_params(field)
            if idx_params:
                self.create_index(collection_name, field["name"], idx_params)
        self.logger.info(f"所有索引已创建：{collection_name}")

        self.logger.info(f"成功创建集合: {collection_name}")

    def drop_collection(self, collection_name):
        self.client.drop_collection(collection_name)

    def has_collection(self, collection_name):
        return self.client.has_collection(collection_name)

    def load(self, collection_name):
        self.client.load_collection(collection_name)

    def release(self, collection_name):
        self.client.release_collection(collection_name)

    def create_index(self, collection_name, field_name, index_params_dict):
        """
        抽象的 index 创建方法

        index_params_dict 示例（字典格式）：
        {
            "index_name": "dense_auto_index",
            "index_type": "AUTOINDEX",
            "metric_type": "COSINE",  # 仅向量字段需要
            "params": {}
        }
        """
        self.logger.info(f"为字段 [{field_name}] 创建索引...")
        
        # 将字典转换为 IndexParams 对象
        index_params_obj = self.client.prepare_index_params()
        
        # 构建 add_index 参数
        add_index_kwargs = {
            "field_name": field_name,
            "index_type": index_params_dict.get("index_type"),
            "params": index_params_dict.get("params", {})
        }
        
        # 添加可选的 index_name
        if "index_name" in index_params_dict:
            add_index_kwargs["index_name"] = index_params_dict["index_name"]
        
        # 添加 metric_type（仅向量字段需要）
        if "metric_type" in index_params_dict:
            add_index_kwargs["metric_type"] = index_params_dict["metric_type"]
        
        index_params_obj.add_index(**add_index_kwargs)
        
        # 创建索引
        self.client.create_index(
            collection_name=collection_name,
            index_params=index_params_obj
        )
        self.logger.info(f"索引创建成功: {collection_name}.{field_name}")
