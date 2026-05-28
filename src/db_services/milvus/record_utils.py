# -*- coding: utf-8 -*-
"""Shared helpers for Milvus schemas, filters, and records."""

from __future__ import annotations

import json
from typing import Any, Iterable

from src.configs.database_config import MilvusConfig
from src.configs.model_config import BM25Config
from src.utils.utils import get_text_hash


def truncate_by_bytes(text: str, max_bytes: int = 1024, encoding: str = "utf-8") -> str:
    encoded = text.encode(encoding)
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode(encoding, errors="ignore")


def serialize_metadata(metadata: Any, max_bytes: int) -> str:
    """Serialize metadata deterministically and keep it within Milvus VARCHAR limit."""
    if metadata is None:
        raw = "{}"
    elif isinstance(metadata, str):
        raw = metadata
    else:
        try:
            raw = json.dumps(metadata, ensure_ascii=False, sort_keys=True, default=str)
        except TypeError:
            raw = json.dumps({"raw": str(metadata)}, ensure_ascii=False)
    return truncate_by_bytes(raw, max_bytes)


def quote_milvus_string(value: str) -> str:
    """Quote a string for a simple Milvus filter expression."""
    return json.dumps(str(value), ensure_ascii=False)


def build_id_filter(ids: Iterable[str]) -> str:
    quoted_ids = [quote_milvus_string(id_) for id_ in ids]
    if not quoted_ids:
        raise ValueError("ids 不能为空")
    return f"id in [{', '.join(quoted_ids)}]"


def prepare_document_parts(documents: Iterable[Any]) -> tuple[list[str], list[Any], list[str]]:
    texts: list[str] = []
    metadatas: list[Any] = []
    ids: list[str] = []
    for doc in documents:
        text = getattr(doc, "page_content", "")
        metadata = getattr(doc, "metadata", {}) or {}
        text_hash = get_text_hash(text)
        texts.append(text)
        metadatas.append(metadata)
        ids.append(text_hash)
    return texts, metadatas, ids


def build_milvus_records(
    ids: list[str],
    texts: list[str],
    metadatas: list[Any],
    dense_vectors: list[Any],
    sparse_vectors: list[Any],
    config: MilvusConfig,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for id_, text, metadata, dense_vector, sparse_vector in zip(
        ids, texts, metadatas, dense_vectors, sparse_vectors
    ):
        if dense_vector is None:
            continue
        records.append(
            {
                "id": id_,
                "text": truncate_by_bytes(text, config.max_text_length),
                "metadata": serialize_metadata(metadata, config.max_metadata_length),
                "dense_vec": dense_vector,
                "bm25_vec": sparse_vector,
            }
        )
    return records


def split_records_by_existing_ids(
    records: list[dict[str, Any]], existing_ids: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    insert_data: list[dict[str, Any]] = []
    update_data: list[dict[str, Any]] = []
    for record in records:
        if record["id"] in existing_ids:
            update_data.append(record)
        else:
            insert_data.append(record)
    return insert_data, update_data


def create_default_schema(config: MilvusConfig, bm25_config: BM25Config) -> dict[str, Any]:
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
                "dim": config.vector_dimension,
            },
            {
                "name": "bm25_vec",
                "dtype": "SPARSE_FLOAT_VECTOR",
                "drop_ratio_build": bm25_config.bm25_drop_ratio,
            },
            {
                "name": "text",
                "dtype": "VARCHAR",
                "max_length": config.max_text_length,
            },
            {
                "name": "metadata",
                "dtype": "VARCHAR",
                "max_length": config.max_metadata_length,
            },
        ],
    }

