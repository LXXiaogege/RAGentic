# -*- coding: utf-8 -*-
"""
@Time ： 2025/11/15 15:42
@Auth ： 吕鑫
@File ：main.py
@IDE ：PyCharm
"""
from pymilvus import MilvusClient
from src.db_services.milvus.database_manager import MilvusDBManager
from src.db_services.milvus.collection_manager import MilvusCollectionManager
from src.db_services.milvus.data_service import MilvusDataService, SearchSettings
from src.models.embedding import TextEmbedding
from src.config.config import QAPipelineConfig
from langchain_core.documents import Document
import os


def test_milvus_pipeline():
    """
    完整的 Milvus 测试流程方法
    测试从数据库初始化到文档增删改查的完整流程
    """
    print("=" * 80)
    print("开始 Milvus 测试流程")
    print("=" * 80)

    # ===============================
    # 1. 初始化 Milvus Client
    # ===============================
    print("\n[步骤 1] 初始化 Milvus Client...")
    client = MilvusClient(
        uri="/Users/lvxin/PycharmProjects/RAGentic/data/knowledge_db/db/test.db"
    )
    print("✓ Milvus Client 初始化成功")

    # ===============================
    # 2. 初始化 DBManager
    # ===============================
    print("\n[步骤 2] 初始化 DBManager...")
    db_manager = MilvusDBManager(client)

    # 创建并切换到测试数据库  lite模式不能测试，lite只支持本地一个数据库
    db_name = "test_db"
    try:
        databases = db_manager.list_databases()
        if databases and db_name in databases:
            print(f"数据库 {db_name} 已存在，跳过创建")
        else:
            db_manager.create_database(db_name)
            print(f"✓ 数据库 {db_name} 创建成功")
    except Exception as e:
        print(f"数据库操作警告: {e}")
        # 尝试直接创建
        try:
            db_manager.create_database(db_name)
            print(f"✓ 数据库 {db_name} 创建成功")
        except:
            pass

    db_manager.use_database(db_name)
    print(f"✓ 已切换到数据库: {db_name}")

    # ===============================
    # 3. 初始化 CollectionManager
    # ===============================
    print("\n[步骤 3] 初始化 CollectionManager...")
    collection_manager = MilvusCollectionManager(db_manager)

    collection_name = "test_collection"

    # 如果集合已存在，先删除（用于测试）
    try:
        has_collection = collection_manager.has_collection(collection_name)
        if has_collection:
            print(f"集合 {collection_name} 已存在，删除旧集合...")
            collection_manager.drop_collection(collection_name)
            print("✓ 旧集合已删除")
    except Exception as e:
        # 尝试直接删除（忽略错误）
        try:
            collection_manager.drop_collection(collection_name)
            print("✓ 尝试删除旧集合")
        except:
            print(f"集合检查警告（可忽略）: {e}")

    # 创建集合 schema（符合 MilvusDocumentStore 要求的格式）
    print("创建集合 schema...")
    schema = {
        "auto_id": False,  # 手动指定 ID
        "enable_dynamic_fields": False,
        "fields": [
            {
                "name": "id",
                "dtype": "VARCHAR",
                "max_length": 64,
                "is_primary": True
            },
            {
                "name": "dense_vec",
                "dtype": "FLOAT_VECTOR",
                "dim": 1024  # 根据配置的 vector_dimension
            },
            {
                "name": "bm25_vec",
                "dtype": "SPARSE_FLOAT_VECTOR",
                "drop_ratio_build": 0.2
            },
            {
                "name": "text",
                "dtype": "VARCHAR",
                "max_length": 10000
            },
            {
                "name": "metadata",
                "dtype": "VARCHAR",
                "max_length": 256
            }
        ]
    }

    collection_manager.create_collection(collection_name, schema)
    collection_manager.load(collection_name)
    print(f"✓ 集合 {collection_name} 创建并加载成功")

    # ===============================
    # 4. 初始化配置和 Embedding
    # ===============================
    print("\n[步骤 4] 初始化配置和 Embedding...")
    config = QAPipelineConfig()
    config.collection_name = collection_name
    config.vector_dimension = 1024

    embeddings = TextEmbedding(
        api_key=config.embedding_api_key,
        base_url=config.embedding_base_url,
        model=config.embedding_model,
        cache_path=config.embedding_cache_path
    )
    print("✓ Embedding 模型初始化成功")

    # ===============================
    # 5. 初始化 DocumentStore
    # ===============================
    print("\n[步骤 5] 初始化 DocumentStore...")
    store = MilvusDataService(
        collection_manager=collection_manager,
        embeddings=embeddings,
        collection_name=collection_name,
        config=config
    )
    print("✓ DocumentStore 初始化成功")

    # ===============================
    # 6. 准备测试文档
    # ===============================
    print("\n[步骤 6] 准备测试文档...")
    test_texts = [
        "人工智能是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。",
        "机器学习是人工智能的一个子领域，它使计算机能够从数据中学习，而无需明确编程。",
        "深度学习是机器学习的一个子集，它使用神经网络来模拟人脑的工作方式。",
        "自然语言处理（NLP）是人工智能的一个分支，专注于计算机与人类语言之间的交互。",
        "向量数据库是专门用于存储和检索高维向量数据的数据库系统，常用于推荐系统和搜索引擎。"
    ]

    # 转换为 Langchain Document 格式
    documents = [
        Document(
            page_content=text,
            metadata={"source": f"test_{i + 1}", "page": i + 1}
        )
        for i, text in enumerate(test_texts)
    ]
    print(f"✓ 准备了 {len(documents)} 条测试文档")

    # ===============================
    # 7. 准备并插入文档记录
    # ===============================
    print("\n[步骤 7] 准备并插入文档记录...")
    texts, meta_datas, ids = store.prepare_records(documents)

    # 生成向量
    print("生成稠密向量...")
    dense_vectors = embeddings.get_embedding(texts)

    # 加载/训练 BM25 模型
    print("加载/训练 BM25 模型...")
    store.load_bm25_model(texts)

    # 生成稀疏向量（使用辅助方法确保格式正确）
    print("生成稀疏向量...")
    bm25_vectors = store._prepare_bm25_vectors(texts)

    # 构建记录
    records = []
    for id_, text, meta_data, dense_vector, bm25_vector in zip(
            ids, texts, meta_datas, dense_vectors, bm25_vectors
    ):
        record = {
            "id": id_,
            "text": text[:config.max_text_length],  # 截断超长文本
            "metadata": str(meta_data),
            "dense_vec": dense_vector,
            "bm25_vec": bm25_vector  # 已经是正确的格式（形状为 (1, n) 的稀疏矩阵）
        }
        records.append(record)

    # 插入记录
    store.insert_records(records)
    print(f"✓ 成功插入 {len(records)} 条文档记录")

    # ===============================
    # 8. 测试搜索功能
    # ===============================
    print("\n[步骤 8] 测试搜索功能...")
    query = "什么是人工智能？"

    print(f"查询: {query}")
    print("-" * 80)

    # 测试稠密搜索（不使用稀疏搜索和重排序）
    config.use_sparse_search = False
    config.use_reranker = False
    config.search_top_k = 3
    results = store.search(query)

    print(f"返回 {len(results)} 条结果:")
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Score: {result.get('score', 'N/A')}")
        print(f"ID: {result['id']}")
        print(f"Text: {result['text'][:100]}...")
        print(f"Metadata: {result.get('metadata', 'N/A')}")

    # 测试混合搜索（稠密 + 稀疏）
    print("\n" + "-" * 80)
    print("测试混合搜索（稠密 + 稀疏）...")
    config.use_sparse_search = True
    config.use_reranker = False
    results_hybrid = store.search(query, settings=SearchSettings(k=3, use_sparse=True))

    print(f"返回 {len(results_hybrid)} 条结果:")
    for i, result in enumerate(results_hybrid, 1):
        print(f"\n[{i}] Score: {result.get('score', 'N/A')}")
        print(f"Text: {result['text'][:100]}...")

    # 测试带重排序的搜索
    print("\n" + "-" * 80)
    print("测试带重排序的搜索...")
    config.use_sparse_search = True
    config.use_reranker = True
    results_reranked = store.search(query, settings=SearchSettings(k=3, use_sparse=True, use_reranker=True))

    print(f"返回 {len(results_reranked)} 条结果:")
    for i, result in enumerate(results_reranked, 1):
        print(f"\n[{i}] Score: {result.get('score', 'N/A')}")
        print(f"Text: {result['text'][:100]}...")

    # ===============================
    # 9. 测试更新文档
    # ===============================
    print("\n[步骤 9] 测试更新文档...")
    if results:
        update_id = results[0]['id']
        updated_text = "人工智能（AI）是计算机科学的一个重要分支，致力于创建能够模拟人类智能的系统。"
        updated_meta = {"source": "test_updated", "page": 1, "updated": True}

        # 生成更新后的向量
        updated_dense_vec = embeddings.get_embedding([updated_text])[0]

        # 使用辅助方法生成稀疏向量（确保格式正确：形状为 (1, n) 的稀疏矩阵）
        updated_bm25_vec = store._prepare_bm25_vector(updated_text)

        update_record = {
            "id": update_id,
            "text": updated_text,
            "metadata": str(updated_meta),
            "dense_vec": updated_dense_vec,
            "bm25_vec": updated_bm25_vec
        }

        store.update_records([update_record])
        print(f"✓ 成功更新文档 ID: {update_id}")

        # 验证更新
        config.search_top_k = 1
        updated_results = store.search(updated_text)
        if updated_results and updated_results[0]['id'] == update_id:
            print("✓ 更新验证成功")

    # ===============================
    # 10. 测试删除文档
    # ===============================
    print("\n[步骤 10] 测试删除文档...")
    if len(results) > 1:
        delete_id = results[1]['id']
        store.delete_by_ids(collection_name, [delete_id])
        print(f"✓ 成功删除文档 ID: {delete_id}")

        # 验证删除（尝试查询）
        config.search_top_k = 10
        verify_results = store.search(query)
        remaining_ids = [r['id'] for r in verify_results]
        if delete_id not in remaining_ids:
            print("✓ 删除验证成功")

    # ===============================
    # 11. 测试从目录批量添加文档
    # ===============================
    print("\n[步骤 11] 测试从目录批量添加文档...")
    test_data_dir = "../data/knowledge_db/1. 简介.md"
    if os.path.exists(test_data_dir):
        try:
            # 注意：这里需要目录路径，如果是文件需要调整
            print(f"从 {test_data_dir} 添加文档...")
            # store.add_documents_from_dir(test_data_dir)
            print("✓ 批量添加文档功能测试（跳过，需要正确的目录路径）")
        except Exception as e:
            print(f"批量添加文档测试跳过: {e}")
    else:
        print("测试数据目录不存在，跳过批量添加测试")

    # ===============================
    # 12. 清理资源（可选）
    # ===============================
    print("\n[步骤 12] 测试完成，保留数据用于验证")
    print("如需清理，请手动执行:")
    print(f"  - 删除集合: collection_manager.drop_collection('{collection_name}')")
    print(f"  - 删除数据库: db_manager.drop_database('{db_name}')")

    print("\n" + "=" * 80)
    print("Milvus 测试流程完成！")
    print("=" * 80)

    return {
        "db_manager": db_manager,
        "collection_manager": collection_manager,
        "store": store,
        "collection_name": collection_name
    }


if __name__ == "__main__":
    try:
        test_milvus_pipeline()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
