# -*- coding: utf-8 -*-
"""
@Time ： 2025/11/28 11:08
@Auth ： 吕鑫
@File ：memory_adapter.py
@IDE ：PyCharm
"""
from mem0.configs.base import VectorStoreConfig, EmbedderConfig, RerankerConfig, GraphStoreConfig
from mem0.graphs.configs import Neo4jConfig, LlmConfig
from src.configs.config import AppConfig
from typing import Dict, Any


def app_llm_to_mem0_dict(llm_config) -> Dict[str, Any]:
    return {
        "model": llm_config.model,
        "openai_base_url": llm_config.base_url,
        "api_key": llm_config.api_key,
        "temperature": llm_config.temperature,
        "max_tokens": llm_config.max_tokens,
    }


def app_embed_to_mem0_dict(embed_config) -> Dict[str, Any]:
    return {
        "model": embed_config.model,
        "openai_base_url": embed_config.base_url,
        "api_key": embed_config.api_key,
        "embedding_dims": 1024
    }


def app_vector_store_to_mem0_dict(milvus):
    return {
        "collection_name": milvus.memory_collection_name,
        "embedding_model_dims": milvus.vector_dimension,
        "url": milvus.memory_db_uri,
        "token": ""
    }


def app_rerank_to_mem0_dict(reranker):
    return {
        "model": reranker.rerank_model_path,
        "device": reranker.rerank_device,
        "max_length": reranker.max_length
    }


def app_graph_store_to_mem0_dict(neo4j):
    return {
        "url": neo4j.neo4j_uri,
        "username": neo4j.neo4j_user,
        "password": neo4j.neo4j_password
    }


def adapt_appconfig_to_mem0(app_config: AppConfig) -> Dict[str, Any]:
    """
    接收 AppConfig，将其数据结构映射为 mem0 框架所需的 MemoryConfig 结构。
    """

    # 1. 特有参数获取
    mem_specific = app_config.memory

    # 2. 核心组件适配
    vector_store_cfg = VectorStoreConfig(
        provider="milvus",
        config=app_vector_store_to_mem0_dict(app_config.milvus)
    )
    llm_cfg = LlmConfig(
        provider=app_config.llm.provider,
        config=app_llm_to_mem0_dict(app_config.llm)
    )
    embedder_cfg = EmbedderConfig(
        provider="openai",
        config=app_embed_to_mem0_dict(app_config.embedding)
    )
    reranker_cfg = RerankerConfig(
        provider="sentence_transformer",
        config=app_rerank_to_mem0_dict(app_config.reranker) if mem_specific.enable_rerank else None
    )

    neo4j_cfg = Neo4jConfig(url=app_config.neo4j.url, username=app_config.neo4j.username,
                            password=app_config.neo4j.password, database=app_config.neo4j.database,
                            base_label=app_config.neo4j.use_base_entity_label)
    graph_store = GraphStoreConfig(provider="neo4j", config=neo4j_cfg, llm=llm_cfg)

    # # 3. 组装 MemoryConfig
    memory_config_dict = {
        "vector_store": vector_store_cfg.model_dump(),
        "llm": llm_cfg.model_dump(),
        "embedder": embedder_cfg.model_dump(),
        "history_db_path": mem_specific.history_db_path,
        "version": mem_specific.version,
    }

    if mem_specific.enable_rerank:
        memory_config_dict["reranker"] = reranker_cfg.model_dump()
    if mem_specific.enable_graph:
        memory_config_dict["graph_store"] = graph_store.model_dump()
    return memory_config_dict
