# -*- coding: utf-8 -*-
"""
@Time    : 2025/4/23
@Author  : 吕鑫
@File    : milvus_db.py
@Desc    : Milvus向量数据库管理模块
"""

from pymilvus import MilvusClient, DataType, AnnSearchRequest, RRFRanker
from pymilvus.model.sparse import BM25EmbeddingFunction
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
from pymilvus.model.reranker import CrossEncoderRerankFunction
from typing import List, Dict, Union, Optional, Any
from src.data_process.processor import DataProcessor
from src.utils.utils import get_text_hash
from src.config.config import QAPipelineConfig
from sentence_transformers import CrossEncoder
import os
import json
import numpy as np
from src.config.logger_config import setup_logger
from langfuse import observe

logger = setup_logger(__name__)


def truncate_by_bytes(text: str, max_bytes: int = 1024, encoding: str = 'utf-8') -> str:
    encoded = text.encode(encoding)
    if len(encoded) <= max_bytes:
        return text
    # 找到截断位置
    truncated = encoded[:max_bytes]
    # 防止截断到一半的字符，decode 时加 errors='ignore'
    return truncated.decode(encoding, errors='ignore')


class MilvusDB:
    def __init__(self, config: QAPipelineConfig, embeddings, text_splitter):
        """初始化 Milvus 客户端"""
        self.logger = logger
        self.logger.info("初始化 Milvus 数据库连接...")
        self.config = config

        if self.config.milvus_mode == "local":
            self.db_client = MilvusClient(self.config.vector_db_uri)
        else:
            conn_args = {"uri": self.config.vector_db_uri,
                         "user": self.config.db_user,
                         "password": self.config.db_password,
                         "db_name": self.config.db_name,
                         "collection_name": self.config.collection_name}
            self.logger.debug(f"连接参数: {conn_args}")
            self.db_client = MilvusClient(**conn_args)

        self.data_processor = DataProcessor(text_splitter)
        self.embeddings = embeddings

        self.logger.info("初始化 BM25 模型...")
        self.bm25_func = BM25EmbeddingFunction(analyzer=build_default_analyzer(language=self.config.bm25_language))

        self.bm25_model_path = os.path.join(self.config.bm25_model_dir, self.config.collection_name + "_bm25.pkl")
        if self.bm25_model_path and os.path.exists(self.bm25_model_path):
            self.bm25_func.load(self.bm25_model_path)
            self.logger.info(f"已加载 BM25 模型: {self.bm25_model_path}")

        self.logger.info("初始化重排序模型...")
        self.rerank_function = CrossEncoderRerankFunction(
            model_name=self.config.cross_encoder_model_path,
            device=self.config.cross_encoder_device
        )
        self.cross_encoder = CrossEncoder(self.config.cross_encoder_model_path)
        self.logger.info("Milvus 数据库初始化完成")

    def build_collection(self):
        self.logger.info("开始构建集合...")
        try:
            self.db_client.drop_collection(collection_name=self.config.collection_name)
            self.logger.info(f"已删除旧集合: {self.config.collection_name}")

            schema = self.db_client.create_schema(auto_id=False, enable_dynamic_fields=True)
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=64)
            schema.add_field(field_name="dense_vec", datatype=DataType.FLOAT_VECTOR, dim=self.config.vector_dimension)
            schema.add_field(field_name="bm25_vec", datatype=DataType.SPARSE_FLOAT_VECTOR)
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=self.config.max_text_length)
            schema.add_field(field_name="metadata", datatype=DataType.VARCHAR,
                             max_length=self.config.max_metadata_length)
            self.logger.debug("已创建集合 schema")

            index_params = self.db_client.prepare_index_params()
            index_params.add_index(
                field_name="id"
            )
            index_params.add_index(
                field_name="bm25_vec",
                index_name="sparse_inverted_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": self.config.bm25_drop_ratio},
            )
            index_params.add_index(
                field_name="dense_vec",
                index_type="AUTOINDEX",
                metric_type="COSINE"
            )
            self.logger.debug("已配置索引参数")

            self.db_client.create_collection(collection_name=self.config.collection_name, schema=schema,
                                             index_params=index_params)
            self.logger.info(f"成功创建集合: {self.config.collection_name}")
        except Exception as e:
            self.logger.exception(f"构建集合时发生错误:{e}")
            raise

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
        if records:
            self.logger.info(f"开始插入 {len(records)} 条新文档...")
            try:
                self.db_client.insert(collection_name=self.config.collection_name, data=records)
                self.logger.info(f"成功插入 {len(records)} 条新文档")
            except Exception as e:
                self.logger.exception(f"插入文档时发生错误:{e}")
                raise

    def _backup_records(self, ids: List[str]) -> List[Dict]:
        """备份要更新的记录"""
        if not ids:
            return []

        try:
            quoted_ids = [f'"{id_}"' for id_ in ids]
            filter_expr = f"id in [{', '.join(quoted_ids)}]"

            results = self.db_client.query(
                collection_name=self.config.collection_name,
                filter=filter_expr,
                output_fields=["id", "text", "metadata", "dense_vec", "bm25_vec"]
            )

            return [r for r in results]

        except Exception as e:
            self.logger.warning(f"备份记录失败: {e}")
            return []

    def update_records(self, records: List[Dict]):
        """更新已有向量记录：先删后插，有失败保护"""
        if not records:
            return
        update_ids = [r["id"] for r in records]
        # 备份原始数据
        backup_data = self._backup_records(update_ids)
        try:
            self.logger.info(f"开始更新 {len(records)} 条文档，再删除旧数据")
            self.db_client.delete(collection_name=self.config.collection_name, ids=update_ids)
            self.db_client.insert(collection_name=self.config.collection_name, data=records)
            self.logger.info(f"成功更新 {len(records)} 条已有文档")
        except Exception as update_error:
            self.logger.error(f"更新文档失败，尝试恢复旧数据: {update_error}")
            if backup_data:
                try:
                    self.db_client.insert(collection_name=self.config.collection_name, data=backup_data)
                    self.logger.info("回滚成功")
                except Exception as rollback_error:
                    self.logger.critical(f"回滚失败，需要人工检查: {rollback_error}")
            raise

    def add_documents_from_dir(self, data_dir: str):
        """从目录中批量添加或更新文档向量"""
        self.logger.info(f"开始从目录 {data_dir} 添加文档...")
        try:
            self.build_collection()

            self.logger.info("开始处理文件...")
            documents = self.data_processor.batch_process_files(data_dir)
            texts, meta_datas, ids = self.prepare_records(documents)

            # 批量查询已有 ID
            existing_ids = set()
            if ids:
                self.logger.info("查询已存在的文档 ID...")
                quoted_ids = [f'"{id_}"' for id_ in ids]
                filter_expr = f"id in [{', '.join(quoted_ids)}]"
                results = self.db_client.query(
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
            bm25_docs_embeddings = self.bm25_func.encode_documents(texts)
            bm25_vectors = [{"bm25_vec": bm25_docs_embeddings[[i]]} for i in range(bm25_docs_embeddings.shape[0])]
            self.logger.info("向量生成完成")

            self.logger.info("准备插入和更新数据...")
            for id_, text, meta_data, dense_vector, bm25_vector in zip(ids, texts, meta_datas, dense_vectors,
                                                                       bm25_vectors):
                record = {
                    "id": id_,
                    "text": truncate_by_bytes(text, self.config.max_text_length),
                    "metadata": str(meta_data),
                    "dense_vec": dense_vector,
                    "bm25_vec": bm25_vector['bm25_vec']
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

    def delete_data(self, collection_name: str, ids: List[int], filter_condition: str):
        """删除数据"""
        self.logger.info(f"开始删除数据，集合: {collection_name}")
        if ids:
            self.logger.info(f"按 ID 删除: {ids}")
            return self.db_client.delete(collection_name=collection_name, ids=ids)
        elif filter_condition:
            self.logger.info(f"按条件删除: {filter_condition}")
            return self.db_client.delete(collection_name=collection_name, filter=filter_condition)
        else:
            self.logger.error("ids 和 filter_condition 不能同时为空")
            raise ValueError("ids 和 filter_condition 不能同时为空")

    @observe(name="MilvusDB.search", as_type="retriever")
    def search(self, query: Union[str, List[float], np.array], k: Optional[int] = None,
               use_sparse: Optional[bool] = None, use_reranker: Optional[bool] = None,
               use_contextualize_embedding: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        多路召回 + 多策略重排序：dense + sparse + RRFRanker + CrossEncoder rerank
        """
        self.logger.info("开始搜索...")
        # 使用配置中的默认值
        k = k or self.config.search_top_k
        use_sparse = use_sparse if use_sparse is not None else self.config.use_sparse_search
        use_reranker = use_reranker if use_reranker is not None else self.config.use_reranker
        use_contextualize_embedding = use_contextualize_embedding if use_contextualize_embedding is not None else self.config.use_contextualize_embedding
        self.logger.debug(f"搜索参数: k={k}, use_sparse={use_sparse}, use_reranker={use_reranker}")

        # 获取稠密向量
        if isinstance(query, str):
            self.logger.debug("生成查询的稠密向量...")
            dense_vec = self.embeddings.get_embedding(query)[0]
        elif isinstance(query, (list, np.ndarray)):
            self.logger.debug("使用提供的稠密向量...")
            dense_vec = np.array(query).flatten()
        else:
            self.logger.error(f"不支持的查询类型: {type(query)}")
            raise TypeError("query must be a string, list of floats, or numpy array")

        # 获取稀疏向量（若启用）
        sparse_vec = None
        if use_sparse:
            self.logger.debug("生成查询的稀疏向量...")
            sparse_vec = self.bm25_func.encode_queries([query])

        req_list = []

        # 稠密召回
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
        self.logger.info("执行混合搜索...")
        if len(req_list) > 1:
            self.logger.debug("使用混合搜索（稠密+稀疏）")
            docs = self.db_client.hybrid_search(
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
            docs = self.db_client.search(
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

        # Rerank with CrossEncoder
        if use_reranker:
            self.logger.info("开始重排序...")
            rerank_texts = []
            for hit in docs[0]:
                if use_contextualize_embedding:
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
