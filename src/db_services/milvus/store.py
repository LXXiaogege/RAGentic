# -*- coding: utf-8 -*-
"""
@Time：2026/3/28
@Auth：RAGentic
@File：store.py
@IDE：PyCharm

Milvus 向量库实现
符合 VectorStore 接口，整合 connection + collection + data + retrieval
"""

import asyncio
from typing import Any, List, Optional, Union


from src.configs.database_config import MilvusConfig
from src.configs.logger_config import setup_logger
from src.configs.model_config import BM25Config, RerankConfig
from src.configs.retrieve_config import SearchConfig
from src.db_services.base import SearchResult, VectorStore
from src.db_services.milvus.collection import MilvusCollectionManager
from src.db_services.milvus.connection import MilvusConnection
from src.db_services.milvus.data import MilvusDataService
from src.db_services.milvus.retrieval import HybridRetrievalPipeline
from src.models.embedding import TextEmbedding

logger = setup_logger(__name__)


class MilvusVectorStore(VectorStore):
    """
    Milvus 向量库实现

    符合 VectorStore 接口的门面类，整合所有 Milvus 相关服务
    """

    def __init__(
        self,
        milvus_config: MilvusConfig,
        search_config: SearchConfig,
        rerank_config: RerankConfig,
        bm25_config: BM25Config,
        embeddings: TextEmbedding,
        text_splitter,
    ):
        self.logger = logger
        self.milvus_config = milvus_config
        self.search_config = search_config
        self.embeddings = embeddings
        self.text_splitter = text_splitter

        self._connection: Optional[MilvusConnection] = None
        self._collection_manager: Optional[MilvusCollectionManager] = None
        self._data_service: Optional[MilvusDataService] = None
        self._retrieval_pipeline: Optional[HybridRetrievalPipeline] = None

        self._init_components()

    def _init_components(self):
        """初始化所有组件"""
        self.logger.info("初始化 Milvus 向量库组件...")

        self._connection = MilvusConnection(self.milvus_config)

        self._collection_manager = MilvusCollectionManager(self._connection)

        if self.milvus_config.db_name and self.milvus_config.milvus_mode != "local":
            try:
                databases = self._connection.list_databases()
                if self.milvus_config.db_name not in (databases or []):
                    self._connection.create_database(self.milvus_config.db_name)
                self._connection.use_database(self.milvus_config.db_name)
            except Exception as e:
                self.logger.warning(f"数据库操作警告: {e}")

        self._data_service = MilvusDataService(
            collection_manager=self._collection_manager,
            embeddings=self.embeddings,
            text_splitter=self.text_splitter,
            config=self.milvus_config,
            rerank_config=self.rerank_config,
            bm25_config=self.bm25_config,
        )

        self._retrieval_pipeline = HybridRetrievalPipeline(
            collection_manager=self._collection_manager,
            data_service=self._data_service,
            embeddings=self.embeddings,
            search_config=self.search_config,
        )

        self.logger.info("Milvus 向量库组件初始化完成")

    @property
    def client(self):
        """获取 Milvus 客户端"""
        return self._connection.client if self._connection else None

    @property
    def data_service(self) -> MilvusDataService:
        """获取数据服务"""
        return self._data_service

    @property
    def retrieval_pipeline(self) -> HybridRetrievalPipeline:
        """获取检索流水线"""
        return self._retrieval_pipeline

    def build_collection(self):
        """构建集合（如果不存在则创建）"""
        if not self._collection_manager.has_collection(
            self.milvus_config.collection_name
        ):
            self.logger.info(
                f"集合 {self.milvus_config.collection_name} 不存在，开始创建..."
            )
            schema = self._create_default_schema()
            self._collection_manager.create_collection(
                self.milvus_config.collection_name, schema
            )
            self._collection_manager.load(self.milvus_config.collection_name)
            self.logger.info(
                f"集合 {self.milvus_config.collection_name} 创建并加载完成"
            )
        else:
            self.logger.info(
                f"集合 {self.milvus_config.collection_name} 已存在，跳过创建"
            )
            self._collection_manager.load(self.milvus_config.collection_name)

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
                    "is_primary": True,
                },
                {
                    "name": "dense_vec",
                    "dtype": "FLOAT_VECTOR",
                    "dim": self.milvus_config.vector_dimension,
                },
                {
                    "name": "bm25_vec",
                    "dtype": "SPARSE_FLOAT_VECTOR",
                    "drop_ratio_build": 0.2,
                },
                {
                    "name": "text",
                    "dtype": "VARCHAR",
                    "max_length": self.milvus_config.max_text_length,
                },
                {
                    "name": "metadata",
                    "dtype": "VARCHAR",
                    "max_length": self.milvus_config.max_metadata_length,
                },
            ],
        }

    async def asearch(
        self, query: Union[str, List[float]], config: Optional[SearchConfig] = None
    ) -> List[SearchResult]:
        """
        异步向量搜索

        Args:
            query: 查询字符串或向量
            config: 搜索配置，如果为 None 则使用默认配置

        Returns:
            SearchResult 列表
        """
        search_config = config if config is not None else self.search_config
        return await self._retrieval_pipeline.search(query, search_config)

    async def aadd_documents(self, documents: List[Any]) -> None:
        """
        异步批量添加文档

        Args:
            documents: Document 对象列表
        """
        self.build_collection()
        await self.aadd_documents_from_dir_or_list(documents)

    async def aadd_documents_from_dir(self, data_dir: str) -> None:
        """从目录批量添加文档（兼容旧接口）"""
        self.build_collection()

        def prepare_data():
            from src.data_process.processor import DataProcessor

            processor = DataProcessor(self.text_splitter)
            docs = processor.batch_process_files(data_dir)
            return self._data_service.prepare_records(docs)

        texts, meta_datas, ids = await asyncio.to_thread(prepare_data)

        if not texts:
            self.logger.warning("未找到任何文档进行处理")
            return

        self.logger.info(f"准备处理 {len(texts)} 条文档记录")

        existing_ids = await self._query_existing_ids(ids)

        dense_vectors = await self.embeddings.aget_embedding(texts)

        self._data_service.load_bm25_model(texts)
        bm25_vectors = await asyncio.to_thread(
            self._data_service._prepare_bm25_vectors, texts
        )

        insert_data, update_data = [], []
        for id_, text, meta_data, dense_vector, bm25_vector in zip(
            ids, texts, meta_datas, dense_vectors, bm25_vectors
        ):
            if dense_vector is None:
                self.logger.warning(f"跳过文档 {id_}，因为嵌入向量生成失败")
                continue
            record = {
                "id": id_,
                "text": self._truncate_by_bytes(
                    text, self.milvus_config.max_text_length
                ),
                "metadata": str(meta_data),
                "dense_vec": dense_vector,
                "bm25_vec": bm25_vector,
            }
            if id_ in existing_ids:
                update_data.append(record)
            else:
                insert_data.append(record)

        def run_insert_update():
            if insert_data:
                self._data_service.insert_records(insert_data)
                self.logger.info(f"插入 {len(insert_data)} 条新记录")
            if update_data:
                self._data_service.update_records(update_data)
                self.logger.info(f"更新 {len(update_data)} 条已有记录")

        await asyncio.to_thread(run_insert_update)

        self.logger.info(
            f"文档添加完成，共处理 {len(insert_data)} 条新文档，{len(update_data)} 条更新文档"
        )

    async def aadd_documents_from_dir_or_list(self, documents) -> None:
        """从文档列表添加文档"""
        texts, meta_datas, ids = self._data_service.prepare_records(documents)

        if not texts:
            self.logger.warning("未找到任何文档进行处理")
            return

        self.logger.info(f"准备处理 {len(texts)} 条文档记录")

        existing_ids = await self._query_existing_ids(ids)

        dense_vectors = await self.embeddings.aget_embedding(texts)

        self._data_service.load_bm25_model(texts)
        bm25_vectors = await asyncio.to_thread(
            self._data_service._prepare_bm25_vectors, texts
        )

        insert_data, update_data = [], []
        for id_, text, meta_data, dense_vector, bm25_vector in zip(
            ids, texts, meta_datas, dense_vectors, bm25_vectors
        ):
            if dense_vector is None:
                self.logger.warning(f"跳过文档 {id_}，因为嵌入向量生成失败")
                continue
            record = {
                "id": id_,
                "text": self._truncate_by_bytes(
                    text, self.milvus_config.max_text_length
                ),
                "metadata": str(meta_data),
                "dense_vec": dense_vector,
                "bm25_vec": bm25_vector,
            }
            if id_ in existing_ids:
                update_data.append(record)
            else:
                insert_data.append(record)

        def run_insert_update():
            if insert_data:
                self._data_service.insert_records(insert_data)
                self.logger.info(f"插入 {len(insert_data)} 条新记录")
            if update_data:
                self._data_service.update_records(update_data)
                self.logger.info(f"更新 {len(update_data)} 条已有记录")

        await asyncio.to_thread(run_insert_update)

        self.logger.info(
            f"文档添加完成，共处理 {len(insert_data)} 条新文档，{len(update_data)} 条更新文档"
        )

    async def _query_existing_ids(self, ids_list: List[str]) -> set:
        """查询已存在的文档 ID"""
        if not ids_list:
            return set()
        quoted_ids = [f'"{id_}"' for id_ in ids_list]
        filter_expr = f"id in [{', '.join(quoted_ids)}]"

        try:
            results = await asyncio.to_thread(
                self._collection_manager.client.query,
                collection_name=self.milvus_config.collection_name,
                filter=filter_expr,
                output_fields=["id"],
            )
            return {r["id"] for r in results}
        except Exception as e:
            self.logger.warning(f"查询已有文档ID失败: {e}，假设全部为新文档")
            return set()

    def _truncate_by_bytes(
        self, text: str, max_bytes: int = 1024, encoding: str = "utf-8"
    ) -> str:
        """按字节截断文本"""
        encoded = text.encode(encoding)
        if len(encoded) <= max_bytes:
            return text
        truncated = encoded[:max_bytes]
        return truncated.decode(encoding, errors="ignore")

    async def adelete(self, ids: List[str]) -> None:
        """
        异步删除文档

        Args:
            ids: 文档ID列表
        """
        if not ids:
            return
        self._data_service.delete_by_ids(self.milvus_config.collection_name, ids)
        self.logger.info(f"已删除 {len(ids)} 条文档")

    async def aclose(self) -> None:
        """关闭连接"""
        if self._connection:
            self._connection.close()
        self.logger.info("Milvus 向量库连接已关闭")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._connection:
            self._connection.close()
