# -*- coding: utf-8 -*-
"""
@Time ： 2025/11/28 15:07
@Auth ： 吕鑫
@File ：database_config.py
@IDE ：PyCharm
"""
from typing import Optional

from pydantic import BaseModel, Field


class MilvusConfig(BaseModel):
    milvus_mode: str = Field("local", description="local / remote")
    vector_db_uri: str = Field("/Users/lvxin/PycharmProjects/RAGentic/data/knowledge_db/db/rag.db",
                               description="Milvus URI 或 SQLite 文件路径")
    db_user: str = "root"
    db_password: str = "Milvus"
    db_name: str = "test"
    token: Optional[str] = Field(None, description="Zilliz Cloud 的访问令牌；本地部署时可留为 None")

    collection_name: str = "pys"
    vector_dimension: int = Field(1024, gt=0)
    max_text_length: int = Field(10000, gt=0)
    max_metadata_length: int = Field(256, gt=0)

    # memory
    memory_collection_name: str = "mem0_with_milvus"
    memory_db_uri: str = Field("/Users/lvxin/PycharmProjects/RAGentic/data/memory.db",
                               description="memory 存储数据库文件路径")


class Neo4jConfig(BaseModel):
    url: str = Field(default="neo4j+s://your-instance", description="Neo4j URL")
    username: str = Field(default="neo4j", description="Neo4j 用户名")
    password: str = Field(default="password", description="Neo4j 密码")
    database: str = Field(default="neo4j", description="Neo4j 数据库名称")
    use_base_entity_label: bool = Field(default=False, description="是否为所有实体节点添加基础标签 __Entity__。")
