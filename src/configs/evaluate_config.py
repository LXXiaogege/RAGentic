# -*- coding: utf-8 -*-
"""
@Time ： 2025/11/19 18:20
@Auth ： 吕鑫
@File ：evaluate_config.py
@IDE ：PyCharm
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class EvaluationConfig(BaseModel):
    """
    问答评估配置（EvaluationConfig）

    - 支持 Rouge / BERTScore / GPT-Judge / RAGAS 多种评估方式
    - 所有带默认值的简单字段不使用 Field，保持简洁
    - 有列表默认值的字段使用 default_factory 避免可变对象陷阱
    - 自动校验赋值 & 禁止未声明字段（建议在全局 Config 中加）
    """

    # --- 基础评估方式 ---
    default_eval_method: str = "rouge"  # 可选: rouge | bert | gpt | ragas
    default_eval_limit: int = 3  # 每条 QA 最大参考答案数量限制
    bert_score_lang: str = "zh"  # BERTScore 使用的语言

    # --- GPT Judge 配置 ---
    gpt_judge_system_prompt: str = "你是问答评估专家"
    gpt_judge_prompt_template: str = (
        "你是一个问答评估专家，请判断模型的回答是否与参考答案在语义上表达的是同一个意思。\n"
        "请严格根据参考答案来判断，不要自行拓展或联想。如果不一致，请输出\"否\"。\n"
        "问题：{question}\n"
        "模型回答：{model_answer}\n"
        "参考答案：{reference}\n"
        "请你判断模型回答是否和参考答案表达的是同一个意思，只回答\"是\"或\"否\"："
    )

    # --- RAGAS 评估 ---
    ragas_eval_model: str = "gpt-4"  # RAGAS 使用的 LLM 模型
    ragas_metrics: List[str] = Field(
        default_factory=lambda: [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "answer_similarity",
        ],
        description="RAGAS 评估的指标列表"
    )
    ragas_save_path: Optional[str] = None  # RAGAS 评估结果存储路径
    ragas_file_format: str = "csv"  # 评估结果导出格式: csv、json 等
