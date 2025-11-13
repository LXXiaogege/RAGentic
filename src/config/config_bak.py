# -*- coding: utf-8 -*-
"""
@Time ： 2025/5/9 00:36
@Auth ： 吕鑫
@File ：config.py
@IDE ：PyCharm
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class QAPipelineConfig:
    # --- Embedding ---
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    embedding_model: str = "text2vec"
    embedding_cache_path: str = "../data/embeddings/embedding_cache.json"



    llm_provider = "openai"
    llm_api_key: str = "sk-"
    llm_base_url: str = ""
    llm_model: str = ""

    # --- Text Splitter ---
    chunk_size: int = 500
    chunk_overlap: int = 50

    # --- BM25 ---
    bm25_model_path: str = "../data/knowledge_db/bm25_model.pkl"
    bm25_autofit: bool = True
    bm25_language: str = "zh"
    bm25_drop_ratio: float = 0.2

    # --- Memory ---
    use_memory = False
    memory_window_size: int = 5

    # --- Milvus DB ---
    ## 本地
    milvus_mode = 'local'
    ## uri
    vector_db_uri: str = ""  #
    db_user: str = ""
    db_password: str = ""
    db_name: str = ""
    collection_name: str = ""  # docs
    vector_dimension: int = 1024
    max_text_length: int = 10000
    max_metadata_length: int = 256
    search_top_k: int = 20
    search_multiplier: int = 2  # 用于计算初始召回数量 (k * multiplier)
    use_sparse_search: bool = False
    use_reranker: bool = False
    use_contextualize_embedding: bool = False

    # --- Knowledge Base ---
    knowledge_base_config = {
        'enable': False,
        'path': '../data/knowledge_db/psychology',
        'use_json': False
    }
    use_knowledge_base = False
    knowledge_dir: str = "../data/knowledge_db/psychology"
    use_json: bool = False

    # --- CrossEncoder ---
    cross_encoder_model_path: str = ""
    cross_encoder_device: str = "cpu"

    # --- Retriever ---
    MAX_CONTEXT_LENGTH: int = 3000
    retriever_type: str = "dense"
    retriever_weights: List[float] = field(default_factory=lambda: [0.6, 0.4])
    default_top_k: int = 3
    use_sparse: bool = False

    # --- Prompt ---
    kb_system_prompt_name: str = "kb_system_prompt"
    kb_system_prompt: str = """{prefix}你是一个资深知识问答助手，回答时需参考给定的上下文资料。
    请用中文作答，若无法从资料中获取信息，请如实说明。
    最终回答使用 JSON 格式，如：{{"answer": "回答内容"}}。
    {context_hint}"""
    system_prompt_name: str = "system_prompt"
    system_prompt: str = """你是一个知识问答小助手，请帮助用回答用户想知道的问题。
    """

    # --- Tools ---
    use_tools = False
    tools: Optional[List[Dict]] = field(default_factory=lambda: [
        {
            "type": "function",
            "function": {
                "name": "web_crawl_web_crawl",
                "description": "爬取网页内容",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "网页链接"
                        }
                    },
                    "required": ["url"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "weather_get_forecast",
                "description": "获取美国各州的天气预警信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "latitude": {
                            "type": "string",
                            "description": "经度"
                        },
                        "longitude": {
                            "type": "string",
                            "description": "纬度"
                        },
                    },
                    "required": ["state"]
                }
            }
        }
    ])

    # --- Evaluation ---
    default_eval_method: str = "rouge"  # 可选: rouge/bert/gpt/ragas
    default_eval_limit: int = 3
    bert_score_lang: str = "zh"
    gpt_judge_system_prompt: str = "你是问答评估专家"
    gpt_judge_prompt_template: str = """
    你是一个问答评估专家，请判断模型的回答是否与参考答案在语义上表达的是同一个意思。
    请严格根据参考答案来判断，不要自行拓展或联想。如果不一致，请输出"否"。
    问题：{question}
    模型回答：{model_answer}
    参考答案：{reference}
    请你判断模型回答是否和参考答案表达的是同一个意思，只回答"是"或"否"：
    """.strip()

    # --- RAGAS Evaluation ---
    ragas_eval_model: str = "gpt-4"  # RAGAS 评估使用的模型
    ragas_metrics: List[str] = field(default_factory=lambda: [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_similarity"
    ])
    ragas_save_path: Optional[str] = None  # RAGAS 评估结果保存路径
    ragas_file_format: str = "csv"  # RAGAS 评估结果保存格式
    # --- Message Builder ---
    message_builder_model: str = "gpt-3.5-turbo"
    message_max_tokens: int = 3500
    message_system_prompt_template: str = "{prefix}{context_hint}"
    message_context_hint_template: str = "以下是外部知识库资料：\n{context}" if "{context}" else "当前无外部知识库上下文。"
    message_no_think_prefix: str = "/no_think "

    # pre-retrievers
    rewrite_config = {
        "enable": True,  # 是否启用查询重写
        "mode": "rewrite",  # 可选: rewrite / step_back / sub_query / hyde('只有开启知识库才可有效使用'')
    }

    # langfuse
    langfuse_config = {
        "secret_key": "sk-",
        "public_key": "",
        "host": "https://cloud.langfuse.com",
    }
