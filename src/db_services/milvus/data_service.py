# -*- coding: utf-8 -*-
"""
@Time ： 2025/11/15 12:53
@Auth ： 吕鑫
@File ：data_service.py.py
@IDE ：PyCharm
"""
import asyncio

from pymilvus.model.reranker import CrossEncoderRerankFunction

from src.configs.retrieve_config import SearchConfig
from src.configs.database_config import MilvusConfig
from src.configs.model_config import BM25Config, RerankConfig
from src.configs.logger_config import setup_logger
from src.data_process.processor import DataProcessor
from src.utils.utils import get_text_hash
from src.db_services.milvus.collection_manager import MilvusCollectionManager
from src.models.embedding import TextEmbedding
from typing import Union, List, Optional, Dict, Any
import numpy as np
from langfuse import observe
import os
import json

from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
from pymilvus import AnnSearchRequest, RRFRanker
from pymilvus.model.sparse import BM25EmbeddingFunction

logger = setup_logger(__name__)


def truncate_by_bytes(text: str, max_bytes: int = 1024, encoding: str = 'utf-8') -> str:
    encoded = text.encode(encoding)
    if len(encoded) <= max_bytes:
        return text
    # 找到截断位置
    truncated = encoded[:max_bytes]
    # 防止截断到一半的字符，decode 时加 errors='ignore'
    return truncated.decode(encoding, errors='ignore')


class MilvusDataService:
    def __init__(self, collection_manager: MilvusCollectionManager, embeddings: TextEmbedding, text_splitter,
                 config: MilvusConfig, rerank_config: RerankConfig, bm25_config: BM25Config):
        self.logger = logger
        self.milvus_config = config
        self.rerank_config = rerank_config
        self.bm25_config = bm25_config
        self.client = collection_manager.client
        self.embeddings = embeddings
        self.data_processor = DataProcessor(text_splitter)
        self.bm25_func = BM25EmbeddingFunction(
            analyzer=build_default_analyzer(language=self.bm25_config.bm25_language)
        )
        self.bm25_model_path = os.path.join(self.bm25_config.bm25_model_dir,
                                            self.milvus_config.collection_name + "_bm25.pkl")
        if self.bm25_model_path and os.path.exists(self.bm25_model_path):
            self.bm25_func.load(self.bm25_model_path)
            self.logger.info(f"已加载 BM25 模型: {self.bm25_model_path}")

        self.rerank_function = CrossEncoderRerankFunction(
            model_name=self.rerank_config.rerank_model_path,
            device=self.rerank_config.rerank_device
        )

    def load_bm25_model(self, texts: List[str]):
        self.logger.info("开始加载/训练 BM25 模型...")
        os.makedirs(self.bm25_config.bm25_model_dir, exist_ok=True)
        model_path = self.bm25_model_path
        meta_path = model_path + ".meta.json"
        texts_hash = get_text_hash(''.join(texts[:10]))
        self.logger.debug(f"文本哈希值: {texts_hash}")

        # 2. 加载已有模型
        if os.path.exists(model_path) and os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("texts_hash") == texts_hash:
                self.bm25_func.load(model_path)
                self.logger.info("✅ 已加载 BM25 模型，无需重新训练")
                return

        # 3. 否则重新训练并保存
        self.logger.info("开始训练新的 BM25 模型...")
        self.bm25_func.fit(texts)
        self.bm25_func.save(model_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"texts_hash": texts_hash}, f)
        self.logger.info("✅ BM25 模型已重新训练并保存")

    def _prepare_bm25_vector(self, text: str):
        """
        为单个文本生成 BM25 稀疏向量，返回 Milvus 兼容的格式（形状为 (1, n) 的稀疏矩阵）

        Args:
            text: 输入文本

        Returns:
            形状为 (1, n) 的稀疏矩阵，可以直接用于 Milvus 的 bm25_vec 字段
        """
        bm25_embeddings = self.bm25_func.encode_documents([text])
        return bm25_embeddings[[0]]  # 返回形状为 (1, n) 的矩阵

    def _prepare_bm25_vectors(self, texts: List[str]):
        """
        为多个文本生成 BM25 稀疏向量列表，返回 Milvus 兼容的格式

        Args:
            texts: 输入文本列表

        Returns:
            稀疏向量列表，每个元素是形状为 (1, n) 的稀疏矩阵
        """
        bm25_docs_embeddings = self.bm25_func.encode_documents(texts)
        return [bm25_docs_embeddings[[i]] for i in range(bm25_docs_embeddings.shape[0])]

    def prepare_records(self, documents):
        """将文档转换为结构化记录和 ID, 从Langchain documents 改为milvus原生支持格式"""
        self.logger.info(f"开始准备 {len(documents)} 条文档记录...")
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
        self.client.insert(collection_name=self.milvus_config.collection_name, data=records)

    def delete_by_ids(self, collection_name: str, ids: Optional[Union[list, str, int]]):
        """按ID删除"""
        if not ids:
            return
        self.client.delete(collection_name=collection_name, ids=ids)

    def delete_by_filter(self, collection_name: str, filter_condition: str):
        """按条件删除"""
        if not filter_condition:
            raise ValueError("过滤条件不能为空")
        self.client.delete(collection_name=collection_name, filter=filter_condition)

    def update_records(self, records: List[Dict]):
        """更新已有向量记录：先删后插，有失败保护"""
        if not records:
            return
        update_ids = [r["id"] for r in records]
        # 备份原始数据
        backup_data = self._backup_records(update_ids)
        try:
            self.client.delete(collection_name=self.milvus_config.collection_name, ids=update_ids)
            self.client.insert(collection_name=self.milvus_config.collection_name, data=records)
        except Exception as update_error:
            self.logger.error(f"更新文档失败，尝试恢复旧数据: {update_error}")
            if backup_data:
                try:
                    self.client.insert(collection_name=self.milvus_config.collection_name, data=backup_data)
                    self.logger.info("回滚成功")
                except Exception as rollback_error:
                    self.logger.critical(f"回滚失败，需要人工检查: {rollback_error}")
            raise

    async def aretrieve(self, query: Union[str, List[float], np.ndarray], config: SearchConfig):
        """文档召回"""
        # 获取稠密向量
        if isinstance(query, str):
            dense_vec = (await self.embeddings.aget_embedding(query))[0]
        elif isinstance(query, (list, np.ndarray)):
            dense_vec = np.array(query).flatten()
        else:
            self.logger.error(f"不支持的查询类型: {type(query)}")
            raise TypeError("query must be a string, list of floats, or numpy array")

        # 获取稀疏向量（若启用）
        sparse_vec = None
        if config.use_sparse:
            self.logger.debug("生成查询的稀疏向量...")
            sparse_vec = self.bm25_func.encode_queries([query])

        def run_milvus_search():
            req_list = []
            dense_search_param = AnnSearchRequest(
                data=[dense_vec],
                anns_field="dense_vec",
                param={"metric_type": "COSINE"},
                limit=config.top_k * config.search_multiplier,
            )
            req_list.append(dense_search_param)
            if config.use_sparse and sparse_vec is not None:
                sparse_search_param = AnnSearchRequest(
                    data=[sparse_vec],
                    anns_field="bm25_vec",
                    param={"metric_type": "IP"},
                    limit=config.top_k * config.search_multiplier,
                )
                req_list.append(sparse_search_param)
            # 执行混合搜索
            if len(req_list) > 1:
                return self.client.hybrid_search(
                    self.milvus_config.collection_name,
                    req_list,
                    RRFRanker(),
                    config.top_k,
                    output_fields=["id", "text", "metadata"],
                )
            else:
                return self.client.search(
                    self.milvus_config.collection_name,
                    data=[dense_vec],
                    anns_field="dense_vec",
                    limit=config.top_k,
                    output_fields=["id", "text", "metadata"],
                )

        docs = await asyncio.to_thread(run_milvus_search)
        return docs

    def rerank(self, query: str, docs, config: SearchConfig):
        """重排序"""
        rerank_texts = []
        for hit in docs[0]:
            if config.use_contextualize_embedding:
                rerank_texts.append(
                    f"{hit['entity']['text']}\n\n{hit['entity']['context']}"
                )
            else:
                rerank_texts.append(hit['entity']['text'])

        scores = self.rerank_function(query, rerank_texts)
        reranked = sorted(
            zip(docs[0], scores),
            key=lambda x: x[1].score,
            reverse=True
        )
        docs[0] = [item[0] for item in reranked[:config.top_k]]
        return docs

    async def arerank(self, query: str, docs, config: SearchConfig):
        """异步重排序 (通过线程池)"""
        self.logger.info("开始异步重排序...")
        reranked_docs = await asyncio.to_thread(self.rerank, query, docs, config)
        self.logger.info("异步重排序完成")
        return reranked_docs

    @observe(name="MilvusDB.asearch", as_type="retriever")
    async def asearch(self, query: Union[str, List[float], np.array], config: SearchConfig) -> List[Dict[str, Any]]:
        """
        多路召回 + 多策略重排序：dense + sparse + RRFRanker + CrossEncoder rerank
        """
        self.logger.info("开始搜索...")

        docs = await self.aretrieve(query, config)

        if config.use_reranker:
            docs = await self.arerank(query, docs, config)
        else:
            docs = docs[:config.top_k]

        # 统一输出格式
        results = []
        for hit in docs[0][:config.top_k]:
            entity = hit["entity"]
            results.append({
                "score": hit["score"] if "score" in hit else None,
                "id": entity["id"],
                "text": entity.get("text", ""),
                "metadata": entity.get("metadata"),
            })

        self.logger.info(f"搜索完成，返回 {len(results)} 条结果")
        return results

    def _backup_records(self, ids: List[str]) -> List[Dict]:
        """备份要更新的记录"""
        if not ids:
            return []
        try:
            quoted_ids = [f'"{id_}"' for id_ in ids]
            filter_expr = f"id in [{', '.join(quoted_ids)}]"
            results = self.client.query(
                collection_name=self.milvus_config.collection_name,
                filter=filter_expr,
                output_fields=["id", "text", "metadata", "dense_vec", "bm25_vec"]
            )
            return [r for r in results]
        except Exception as e:
            self.logger.warning(f"备份记录失败: {e}")
            return []

    async def aadd_documents_from_dir(self, data_dir: str):
        """从目录中批量添加或更新文档向量"""

        # 1. 文件处理和记录准备 (CPU密集型，但涉及文件I/O，最好线程化)
        def prepare_data():
            documents = self.data_processor.batch_process_files(data_dir)
            texts, meta_datas, ids = self.prepare_records(documents)
            return texts, meta_datas, ids

        texts, meta_datas, ids = await asyncio.to_thread(prepare_data)

        # 2. 批量查询已有 ID (Milvus I/O 阻塞)
        async def query_existing_ids(ids_list):
            if not ids_list:
                return set()
            quoted_ids = [f'"{id_}"' for id_ in ids_list]
            filter_expr = f"id in [{', '.join(quoted_ids)}]"

            # 使用 asyncio.to_thread 封装同步 Milvus 查询
            results = await asyncio.to_thread(
                self.client.query,
                collection_name=self.milvus_config.collection_name,
                filter=filter_expr,
                output_fields=["id"]
            )
            return {r['id'] for r in results}

        existing_ids = await query_existing_ids(ids)

        # 3. 异步生成稠密向量
        dense_vectors = await self.embeddings.aget_embedding(texts)
        # 4. 稀疏向量生成 (CPU密集型，可同步/线程化)
        self.load_bm25_model(texts)
        bm25_vectors = await asyncio.to_thread(self._prepare_bm25_vectors, texts)

        # 5. 准备插入/更新数据 (CPU密集型，可同步)
        insert_data, update_data = [], []
        for id_, text, meta_data, dense_vector, bm25_vector in zip(ids, texts, meta_datas, dense_vectors, bm25_vectors):
            record = {
                "id": id_,
                "text": truncate_by_bytes(text, self.milvus_config.max_text_length),
                "metadata": str(meta_data),
                "dense_vec": dense_vector,
                "bm25_vec": bm25_vector
            }
            if id_ in existing_ids:
                update_data.append(record)
            else:
                insert_data.append(record)

        # 6. 异步插入和更新 (Milvus I/O 阻塞)
        def run_insert_update():
            self.insert_records(insert_data)
            self.update_records(update_data)

        await asyncio.to_thread(run_insert_update)

        self.logger.info("文档添加完成 (异步)")
