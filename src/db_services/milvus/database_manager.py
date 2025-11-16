# -*- coding: utf-8 -*-
"""
@Time ： 2025/11/15 12:50
@Auth ： 吕鑫
@File ：database_manager.py.py
@IDE ：PyCharm
"""
from src.config.logger_config import setup_logger

from pymilvus import MilvusClient

logger = setup_logger(__name__)


class MilvusDBManager:
    def __init__(self, client: MilvusClient):
        self.logger = logger
        self.client = client

    def create_database(self, db_name):
        """创建 Milvus 客户端"""
        self.client.create_database(db_name)

    def drop_database(self, db_name):
        self.client.drop_database(db_name)

    def list_databases(self):
        return self.client.list_databases()

    def use_database(self, db_name):
        self.client.use_database(db_name)

    def check_database(self):
        """健康检查"""
        try:
            if self.client and self.client.list_databases() is not None:
                return True
            return False
        except Exception as e:
            self.logger.error(f"当前无数据库: {e}")
            return False

    def close(self):
        """关闭连接"""
        if self.client:
            self.client.close()
            self.client = None
