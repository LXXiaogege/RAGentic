# -*- coding: utf-8 -*-
"""Milvus record utility tests."""

from types import SimpleNamespace

from src.configs.database_config import MilvusConfig
from src.configs.model_config import BM25Config
from src.db_services.milvus.record_utils import (
    build_id_filter,
    build_milvus_records,
    create_default_schema,
    prepare_document_parts,
    serialize_metadata,
    split_records_by_existing_ids,
    truncate_by_bytes,
)


def test_truncate_by_bytes_keeps_valid_utf8():
    text = "你好世界"

    truncated = truncate_by_bytes(text, max_bytes=5)

    assert truncated == "你"
    assert truncated.encode("utf-8").decode("utf-8") == truncated


def test_serialize_metadata_uses_json_and_byte_limit():
    metadata = {"source": "文档.md", "page": 1}

    serialized = serialize_metadata(metadata, max_bytes=128)

    assert '"source": "文档.md"' in serialized
    assert '"page": 1' in serialized


def test_build_id_filter_quotes_ids_safely():
    filter_expr = build_id_filter(["abc", 'a"b'])

    assert filter_expr == 'id in ["abc", "a\\"b"]'


def test_prepare_document_parts_generates_stable_ids():
    docs = [
        SimpleNamespace(page_content="hello", metadata={"source": "a.md"}),
        SimpleNamespace(page_content="hello", metadata={"source": "b.md"}),
    ]

    texts, metadatas, ids = prepare_document_parts(docs)

    assert texts == ["hello", "hello"]
    assert metadatas == [{"source": "a.md"}, {"source": "b.md"}]
    assert ids[0] == ids[1]


def test_build_and_split_milvus_records():
    config = MilvusConfig(max_text_length=16, max_metadata_length=64)

    records = build_milvus_records(
        ids=["1", "2"],
        texts=["长文本" * 10, "短文本"],
        metadatas=[{"source": "a.md"}, {"source": "b.md"}],
        dense_vectors=[[0.1, 0.2], None],
        sparse_vectors=[{"indices": [], "values": []}, {"indices": [], "values": []}],
        config=config,
    )
    insert_data, update_data = split_records_by_existing_ids(records, {"1"})

    assert len(records) == 1
    assert records[0]["metadata"] == '{"source": "a.md"}'
    assert insert_data == []
    assert update_data == records


def test_create_default_schema_uses_configured_sparse_drop_ratio():
    config = MilvusConfig(vector_dimension=3, max_text_length=100, max_metadata_length=80)
    bm25_config = BM25Config(bm25_drop_ratio=0.4)

    schema = create_default_schema(config, bm25_config)
    fields = {field["name"]: field for field in schema["fields"]}

    assert fields["dense_vec"]["dim"] == 3
    assert fields["bm25_vec"]["drop_ratio_build"] == 0.4
    assert fields["text"]["max_length"] == 100
    assert fields["metadata"]["max_length"] == 80

