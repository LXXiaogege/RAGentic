# -*- coding: utf-8 -*-
"""
@Time：2026/3/28
@Auth：RAGentic
@File：data.py
@IDE：PyCharm

Milvus 数据服务
职责：文档的插入、更新、删除、备份恢复（不含检索逻辑）
"""

import json
import os
from typing import Dict, List, Optional, Union

from src.configs.database_config import MilvusConfig
from src.configs.logger_config import setup_logger
from src.configs.model_config import BM25Config, RerankConfig
from src.data_process.processor import DataProcessor
from src.db_services.milvus.collection import MilvusCollectionManager
from src.models.embedding import TextEmbedding
from src.utils.utils import get_text_hash

logger = setup_logger(__name__)


def truncate_by_bytes(text: str, max_bytes: int = 1024, encoding: str = "utf-8") -> str:
    """按字节截断文本"""
    encoded = text.encode(encoding)
    if len(encoded) <= max_bytes:
        return text
    truncated = encoded[:max_bytes]
    return truncated.decode(encoding, errors="ignore")


class MilvusDataService:
    """
    Milvus 数据服务

    负责文档的插入、更新、删除、批量添加等数据操作
    不直接负责检索召回
    """

    def __init__(
        self,
        collection_manager: MilvusCollectionManager,
        embeddings: TextEmbedding,
        text_splitter,
        config: MilvusConfig,
        rerank_config: RerankConfig,
        bm25_config: BM25Config,
    ):
        self.logger = logger
        self.milvus_config = config
        self.rerank_config = rerank_config
        self.bm25_config = bm25_config
        self.collection_manager = collection_manager
        self.embeddings = embeddings
        self.data_processor = DataProcessor(text_splitter)

        self.bm25_func = None
        self.rerank_function = None
        self._init_models()

    def _init_models(self):
        """懒加载初始化 BM25 和 Rerank 模型"""
        from pymilvus.model.reranker import CrossEncoderRerankFunction
        from pymilvus.model.sparse import BM25EmbeddingFunction
        from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer

        if BM25EmbeddingFunction is not None and build_default_analyzer is not None:
            try:
                self.bm25_func = BM25EmbeddingFunction(
                    analyzer=build_default_analyzer(
                        language=self.bm25_config.bm25_language
                    )
                )
            except (ImportError, ValueError, OSError) as e:
                self.logger.warning(f"初始化 BM25 失败，将禁用 sparse：{e}")
        else:
            self.logger.warning("当前 pymilvus 不包含 BM25 模块，将禁用 sparse 检索")

        self.bm25_model_path = os.path.join(
            self.bm25_config.bm25_model_dir,
            self.milvus_config.collection_name + "_bm25.pkl",
        )
        if (
            self.bm25_func is not None
            and self.bm25_model_path
            and os.path.exists(self.bm25_model_path)
        ):
            self.bm25_func.load(self.bm25_model_path)
            self.logger.info(f"已加载 BM25 模型: {self.bm25_model_path}")

        if CrossEncoderRerankFunction is not None:
            try:
                self.rerank_function = CrossEncoderRerankFunction(
                    model_name=self.rerank_config.rerank_model_path,
                    device=self.rerank_config.rerank_device,
                )
            except (ImportError, ValueError, OSError, IOError) as e:
                self.logger.warning(f"初始化 reranker 失败，将禁用 reranker：{e}")
        else:
            self.logger.warning("当前 pymilvus 不包含 reranker 模块，将禁用 reranker")

    def load_bm25_model(self, texts: List[str]):
        """加载/训练 BM25 模型"""
        if self.bm25_func is None:
            self.logger.warning("BM25 不可用，跳过模型加载/训练")
            return

        self.logger.info("开始加载/训练 BM25 模型...")
        os.makedirs(self.bm25_config.bm25_model_dir, exist_ok=True)
        model_path = self.bm25_model_path
        meta_path = model_path + ".meta.json"
        texts_hash = get_text_hash("".join(texts[:10]))
        self.logger.debug(f"文本哈希值: {texts_hash}")

        if os.path.exists(model_path) and os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("texts_hash") == texts_hash:
                self.bm25_func.load(model_path)
                self.logger.info("已加载 BM25 模型，无需重新训练")
                return

        self.logger.info("开始训练新的 BM25 模型...")
        self.bm25_func.fit(texts)
        self.bm25_func.save(model_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"texts_hash": texts_hash}, f)
        self.logger.info("BM25 模型已重新训练并保存")

    def prepare_records(self, documents) -> tuple:
        """
        将文档转换为结构化记录

        Returns:
            (texts, meta_datas, ids) 元组
        """
        self.logger.info(f"准备 {len(documents)} 条文档记录...")
        texts, meta_datas, ids = [], [], []
        for doc in documents:
            text = doc.page_content
            text_hash = get_text_hash(text)
            texts.append(text)
            meta_datas.append(doc.metadata)
            ids.append(text_hash)
        self.logger.debug(f"文档记录准备完成，生成 {len(ids)} 条记录")
        return texts, meta_datas, ids

    def insert_records(self, records: List[Dict]):
        """插入新向量记录"""
        if not records:
            return
        self.collection_manager.client.insert(
            collection_name=self.milvus_config.collection_name, data=records
        )

    def delete_by_ids(self, collection_name: str, ids: Optional[Union[list, str, int]]):
        """按ID删除"""
        if not ids:
            return
        self.collection_manager.client.delete(collection_name=collection_name, ids=ids)

    def delete_by_filter(self, collection_name: str, filter_condition: str):
        """按条件删除"""
        if not filter_condition:
            raise ValueError("过滤条件不能为空")
        self.collection_manager.client.delete(
            collection_name=collection_name, filter=filter_condition
        )

    def update_records(self, records: List[Dict]):
        """
        更新已有向量记录：先删后插，有失败保护

        Args:
            records: 要更新的记录列表
        """
        if not records:
            return
        update_ids = [r["id"] for r in records]
        backup_data = self._backup_records(update_ids)
        if not backup_data:
            raise RuntimeError(
                f"备份记录失败（返回空），中止更新操作以防止数据丢失。涉及 ID：{update_ids}"
            )
        try:
            self.collection_manager.client.delete(
                collection_name=self.milvus_config.collection_name, ids=update_ids
            )
            self.collection_manager.client.insert(
                collection_name=self.milvus_config.collection_name, data=records
            )
        except (IOError, OSError, RuntimeError) as update_error:
            self.logger.error(f"更新文档失败，尝试恢复旧数据: {update_error}")
            if backup_data:
                try:
                    self.collection_manager.client.insert(
                        collection_name=self.milvus_config.collection_name,
                        data=backup_data,
                    )
                    self.logger.info("回滚成功")
                except (IOError, OSError) as rollback_error:
                    self.logger.critical(f"回滚失败，需要人工检查: {rollback_error}")
            raise

    def _prepare_bm25_vector(self, text: str):
        """为单个文本生成 BM25 稀疏向量"""
        if self.bm25_func is None:
            return {"indices": [], "values": []}
        bm25_embeddings = self.bm25_func.encode_documents([text])
        return bm25_embeddings[[0]]

    def _prepare_bm25_vectors(self, texts: List[str]) -> List:
        """为多个文本生成 BM25 稀疏向量"""
        if self.bm25_func is None:
            return [{"indices": [], "values": []} for _ in range(len(texts))]
        bm25_docs_embeddings = self.bm25_func.encode_documents(texts)
        return [bm25_docs_embeddings[[i]] for i in range(bm25_docs_embeddings.shape[0])]

    def _backup_records(self, ids: List[str]) -> List[Dict]:
        """备份要更新的记录"""
        if not ids:
            return []
        try:
            quoted_ids = [f'"{id_}"' for id_ in ids]
            filter_expr = f"id in [{', '.join(quoted_ids)}]"
            results = self.collection_manager.client.query(
                collection_name=self.milvus_config.collection_name,
                filter=filter_expr,
                output_fields=["id", "text", "metadata", "dense_vec", "bm25_vec"],
            )
            return [r for r in results]
        except Exception as e:
            self.logger.warning(f"备份记录失败: {e}")
            return []
