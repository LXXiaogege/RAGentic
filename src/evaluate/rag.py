# -*- coding: utf-8 -*-
"""
@Time ： 2025/6/4 01:02
@Auth ： 吕鑫
@File ：rag.py
@IDE ：PyCharm
"""
from typing import List, Dict, Optional
import pandas as pd
from datasets import Dataset
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_similarity
)
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from src.config.config import QAPipelineConfig
from src.config.logger_config import setup_logger

# 设置日志记录器
logger = setup_logger(__name__)


class RAGASEvaluator:
    def __init__(self, config: Optional[QAPipelineConfig] = None):
        """
        初始化评估器，可自定义评估指标
        
        Args:
            config: 配置对象，如果为None则使用默认配置
        """
        self.logger = logger
        self.logger.info("Initializing RAGAS Evaluator")
        self.config = config or QAPipelineConfig()

        # 设置评估指标
        metric_map = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "answer_similarity": answer_similarity
        }

        self.metrics = [metric_map[m] for m in self.config.ragas_metrics if m in metric_map]
        self.logger.debug(f"Using metrics: {[metric.name for metric in self.metrics]}")

        self.evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
        self.logger.info("Initialized evaluator LLM with GPT-4")
        # self.config = QAPipelineConfig()
        # self.evaluator_llm = LangchainLLMWrapper(
        #     ChatOpenAI(model=self.config.llm_model, api_key=self.config.llm_api_key,
        #                base_url=self.config.llm_base_url))

    def prepare_dataset(self, qa_data: List[Dict]) -> Dataset:
        """
        将问答数据转换为 ragas 所需的 Dataset，并补充 reference 字段
        """
        self.logger.info(f"Preparing dataset with {len(qa_data)} QA pairs")
        records = []
        for item in qa_data:
            ground_truth = item["ground_truths"][0] if item["ground_truths"] else ""
            records.append({
                "question": item["query"],
                "answer": item["prediction"],
                "contexts": item["contexts"],
                "ground_truths": item["ground_truths"],
                "reference": ground_truth  # 添加这一字段
            })
        self.logger.debug(f"Dataset preparation completed with {len(records)} records")
        return Dataset.from_list(records)

    def evaluate(
            self,
            qa_data: List[Dict],
            save_path: Optional[str] = None,
            file_format: Optional[str] = None
    ) -> pd.DataFrame:
        """
        评估RAG系统性能
        
        Args:
            qa_data: 问答数据列表
            save_path: 结果保存路径，如果为None则使用配置中的路径
            file_format: 保存文件格式（csv或json），如果为None则使用配置中的格式
            
        Returns:
            pd.DataFrame: 评估结果
        """
        self.logger.info("Starting RAG evaluation")
        self.logger.debug(f"Input QA data size: {len(qa_data)}")

        # 使用配置中的保存路径和格式（如果未指定）
        save_path = save_path or self.config.ragas_save_path
        file_format = file_format or self.config.ragas_file_format

        try:
            ragas_dataset = self.prepare_dataset(qa_data)
            self.logger.info("Dataset prepared successfully")

            self.logger.info("Running RAGAS evaluation")
            results = evaluate(ragas_dataset, metrics=self.metrics, llm=self.evaluator_llm)
            self.logger.info("Evaluation completed successfully")

            df = pd.DataFrame([results])
            self.logger.debug(f"Evaluation results: {results}")

            if save_path:
                self.logger.info(f"Saving results to {save_path} in {file_format} format")
                if file_format == "csv":
                    df.to_csv(save_path, index=False)
                elif file_format == "json":
                    df.to_json(save_path, orient="records", indent=2)
                self.logger.info("Results saved successfully")

            return df

        except Exception as e:
            self.logger.error(f"Error during evaluation: {str(e)}")
            raise
