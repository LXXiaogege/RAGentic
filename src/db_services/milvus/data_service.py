# -*- coding: utf-8 -*-
"""
@Time ： 2025/11/15 12:53
@Auth ： 吕鑫
@File ：data_service.py.py
@IDE ：PyCharm
"""
from pymilvus.model.reranker import CrossEncoderRerankFunction

from src.config.config import QAPipelineConfig
from src.config.logger_config import setup_logger
from src.data_process.processor import DataProcessor
from src.utils.utils import get_text_hash
from src.db_services.milvus.collection_manager import MilvusCollectionManager

from typing import Union, List, Optional, Dict, Any
import numpy as np
from langfuse import observe
import os
import json

from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
from pymilvus import AnnSearchRequest, RRFRanker
from pymilvus.model.sparse import BM25EmbeddingFunction
from pydantic import BaseModel

logger = setup_logger(__name__)


def truncate_by_bytes(text: str, max_bytes: int = 1024, encoding: str = 'utf-8') -> str:
    encoded = text.encode(encoding)
    if len(encoded) <= max_bytes:
        return text
    # 找到截断位置
    truncated = encoded[:max_bytes]
    # 防止截断到一半的字符，decode 时加 errors='ignore'
    return truncated.decode(encoding, errors='ignore')


class SearchSettings(BaseModel):
    """
    定义了可以在运行时（从前端）覆盖的搜索设置
    """
    k: Optional[int] = None
    use_sparse: Optional[bool] = None
    use_reranker: Optional[bool] = None
    use_contextualize_embedding: Optional[bool] = None


class MilvusDataService:
    def __init__(self, collection_manager: MilvusCollectionManager, embeddings, collection_name: str,
                 config: QAPipelineConfig, text_splitter):
        self.logger = logger
        self.config = config
        self.client = collection_manager.client
        self.embeddings = embeddings
        self.data_processor = DataProcessor(text_splitter)
        self.bm25_func = BM25EmbeddingFunction(analyzer=build_default_analyzer(language=self.config.bm25_language))
        self.bm25_model_path = os.path.join(self.config.bm25_model_dir, collection_name + "_bm25.pkl")
        if self.bm25_model_path and os.path.exists(self.bm25_model_path):
            self.bm25_func.load(self.bm25_model_path)
            self.logger.info(f"已加载 BM25 模型: {self.bm25_model_path}")

        self.rerank_function = CrossEncoderRerankFunction(
            model_name=self.config.cross_encoder_model_path,
            device=self.config.cross_encoder_device
        )

    def load_bm25_model(self, texts: List[str]):
        self.logger.info("开始加载/训练 BM25 模型...")
        os.makedirs(self.config.bm25_model_dir, exist_ok=True)
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
        self.client.insert(collection_name=self.config.collection_name, data=records)

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
            self.logger.info(f"开始更新 {len(records)} 条文档，再删除旧数据")
            self.client.delete(collection_name=self.config.collection_name, ids=update_ids)
            self.client.insert(collection_name=self.config.collection_name, data=records)
            self.logger.info(f"成功更新 {len(records)} 条已有文档")
        except Exception as update_error:
            self.logger.error(f"更新文档失败，尝试恢复旧数据: {update_error}")
            if backup_data:
                try:
                    self.client.insert(collection_name=self.config.collection_name, data=backup_data)
                    self.logger.info("回滚成功")
                except Exception as rollback_error:
                    self.logger.critical(f"回滚失败，需要人工检查: {rollback_error}")
            raise

    def retrieve(self, query: Union[str, List[float], np.ndarray], k: int = 10, use_sparse: bool = True):
        """文档召回"""
        # 获取稠密向量
        if isinstance(query, str):
            dense_vec = self.embeddings.get_embedding(query)[0]
        elif isinstance(query, (list, np.ndarray)):
            dense_vec = np.array(query).flatten()
        else:
            self.logger.error(f"不支持的查询类型: {type(query)}")
            raise TypeError("query must be a string, list of floats, or numpy array")

        # 获取稀疏向量（若启用）
        sparse_vec = None
        if use_sparse:
            self.logger.debug("生成查询的稀疏向量...")
            sparse_vec = self.bm25_func.encode_queries([query])

        # 稠密召回
        req_list = []
        dense_search_param = AnnSearchRequest(
            data=[dense_vec],
            anns_field="dense_vec",
            param={"metric_type": "COSINE"},
            limit=k * self.config.search_multiplier,
        )
        req_list.append(dense_search_param)

        # 稀疏召回（可选）
        if use_sparse and sparse_vec is not None:
            sparse_search_param = AnnSearchRequest(
                data=[sparse_vec],
                anns_field="bm25_vec",
                param={"metric_type": "IP"},
                limit=k * self.config.search_multiplier,
            )
            req_list.append(sparse_search_param)

        # 执行混合搜索（多路 + RRF 排序）
        if len(req_list) > 1:
            self.logger.debug("使用混合搜索（稠密+稀疏）")
            docs = self.client.hybrid_search(
                self.config.collection_name,
                req_list,
                RRFRanker(),  # Reciprocal Rank Fusion，轻量级排序器
                k,
                output_fields=[
                    "id",
                    "text",
                    "metadata"
                ],
            )
        else:
            self.logger.debug("使用单一稠密搜索")
            docs = self.client.search(
                self.config.collection_name,
                data=[dense_vec],
                anns_field="dense_vec",
                limit=k,
                output_fields=[
                    "id",
                    "text",
                    "metadata"
                ],
            )
        return docs

    def rerank(self, query: str, docs, k: int, use_contextualize=False):
        """重排序"""
        self.logger.info("开始重排序...")
        rerank_texts = []
        for hit in docs[0]:
            if use_contextualize:
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
        docs[0] = [item[0] for item in reranked[:k]]
        self.logger.info("重排序完成")
        return docs

    @observe(name="MilvusDB.search", as_type="retriever")
    def search(self, query: Union[str, List[float], np.array],
               settings: Optional[SearchSettings] = None) -> List[Dict[str, Any]]:
        """
        多路召回 + 多策略重排序：dense + sparse + RRFRanker + CrossEncoder rerank
        """
        self.logger.info("开始搜索...")
        # 使用配置中的默认值
        # 1. 首先，使用类中存储的“默认值”
        k = self.config.search_top_k
        use_sparse = self.config.use_sparse_search
        use_reranker = self.config.use_reranker
        use_contextualize = self.config.use_contextualize_embedding

        # 2. 然后，如果传入了 settings，就用它来“覆盖”默认值（前台等）
        if settings:
            k = settings.k if settings.k is not None else k
            use_sparse = settings.use_sparse if settings.use_sparse is not None else use_sparse
            use_reranker = settings.use_reranker if settings.use_reranker is not None else use_reranker
            use_contextualize = settings.use_contextualize_embedding if settings.use_contextualize_embedding is not None else use_contextualize

        docs = self.retrieve(query, k, use_sparse)

        if use_reranker:
            docs = self.rerank(query, docs, k, use_contextualize)
        else:
            docs = docs[:k]

        # 统一输出格式
        results = []
        for hit in docs[0][:k]:
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
                collection_name=self.config.collection_name,
                filter=filter_expr,
                output_fields=["id", "text", "metadata", "dense_vec", "bm25_vec"]
            )
            return [r for r in results]
        except Exception as e:
            self.logger.warning(f"备份记录失败: {e}")
            return []

    def add_documents_from_dir(self, data_dir: str):
        """从目录中批量添加或更新文档向量"""
        self.logger.info(f"开始从目录 {data_dir} 添加文档...")
        try:
            self.logger.info("开始处理文件...")
            documents = self.data_processor.batch_process_files(data_dir)
            texts, meta_datas, ids = self.prepare_records(documents)

            # 批量查询已有 ID
            existing_ids = set()
            if ids:
                self.logger.info("查询已存在的文档 ID...")
                quoted_ids = [f'"{id_}"' for id_ in ids]
                filter_expr = f"id in [{', '.join(quoted_ids)}]"
                results = self.client.query(
                    collection_name=self.config.collection_name,
                    filter=filter_expr,
                    output_fields=["id"]
                )
                existing_ids = {r['id'] for r in results}
                self.logger.info(f"找到 {len(existing_ids)} 条已存在的文档")

            # 生成向量
            self.logger.info("开始生成文档向量...")
            dense_vectors = self.embeddings.get_embedding(texts)
            insert_data, update_data = [], []
            self.load_bm25_model(texts)
            # 使用辅助方法生成稀疏向量，确保格式一致
            bm25_vectors = self._prepare_bm25_vectors(texts)
            self.logger.info("向量生成完成")

            self.logger.info("准备插入和更新数据...")
            for id_, text, meta_data, dense_vector, bm25_vector in zip(ids, texts, meta_datas, dense_vectors,
                                                                       bm25_vectors):
                record = {
                    "id": id_,
                    "text": truncate_by_bytes(text, self.config.max_text_length),
                    "metadata": str(meta_data),
                    "dense_vec": dense_vector,
                    "bm25_vec": bm25_vector
                }
                if id_ in existing_ids:
                    update_data.append(record)
                else:
                    insert_data.append(record)

            # 插入和更新
            self.insert_records(insert_data)
            self.update_records(update_data)
            self.logger.info("文档添加完成")

        except Exception as e:
            self.logger.exception(f"文档向量插入/更新失败：{e}")
            raise
