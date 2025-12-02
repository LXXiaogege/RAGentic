# -*- coding: utf-8 -*-
"""
@Time    : 2025/4/23
@Author  : 吕鑫
@File    : evaluate.py
@Desc    : 支持多种指标（ROUGE/BERTScore/GPT-judge）问答评估模块
"""

from rouge import Rouge
from bert_score import score as bert_score_func
from typing import List, Dict, Optional
import re
import json
from src.configs.evaluate_config import EvaluationConfig
from src.configs.logger_config import setup_logger
from langfuse import observe

from src.configs.retrieve_config import SearchConfig

logger = setup_logger(__name__)


class QAEvaluator:
    def __init__(self, qa_pipeline, eval_config: EvaluationConfig, search_config: SearchConfig):
        self.logger = logger
        self.logger.info("初始化问答评估器...")
        self.eval_config = eval_config
        self.search_config = search_config
        self.qa_pipeline = qa_pipeline  # QAPipeline 实例
        self.llm = qa_pipeline.llm_caller
        self.logger.info("问答评估器初始化完成")

    def _extract_answer(self, raw_answer: str) -> str:
        """提取模型回答中的核心内容"""
        self.logger.debug("开始提取答案核心内容...")
        try:
            match = re.search(r'{.*}', raw_answer, re.DOTALL)
            if match:
                answer = json.loads(match.group())["answer"]
                self.logger.debug("成功从JSON格式中提取答案")
                return answer
        except Exception as e:
            self.logger.warning(f"JSON解析失败，使用备用提取方法: {str(e)}")
            pass
        answer = raw_answer.strip().splitlines()[-1]
        self.logger.debug(f"使用最后一行作为答案，长度: {len(answer)}")
        return answer

    @observe(name="QAEvaluator.evaluate", as_type="evaluator")
    def evaluate(self, qa_pairs: List[Dict[str, str]]) -> List[
        Dict]:
        """
        评估问答对
        
        Args:
            qa_pairs: 问答对列表
        """
        self.logger.info(f"开始评估，使用 {self.eval_config.eval_method} 方法，检索数量 :{self.search_config.top_k}")
        self.logger.info(f"待评估问答对数量: {len(qa_pairs)}")

        try:
            if self.eval_config.eval_method == "rouge":
                results = self.evaluate_with_rouge(qa_pairs)
            elif self.eval_config.eval_method == "bert":
                results = self.evaluate_with_bert_score(qa_pairs)
            elif self.eval_config.eval_method == "gpt":
                results = self.evaluate_with_gpt_judge(qa_pairs)
            else:
                self.logger.error(f"未知评估方法: {self.eval_config.eval_method}")
                raise ValueError(f"未知评估方法: {self.eval_config.eval_method}")

            self.logger.info(f"评估完成，共处理 {len(results)} 个问答对")
            return results

        except Exception as e:
            self.logger.exception(f"评估过程发生错误:{e}")
            raise

    @observe(name="QAEvaluator.evaluate_with_rouge", as_type="evaluator")
    def evaluate_with_rouge(self, qa_pairs: List[Dict[str, str]]) -> List[Dict]:
        """使用 ROUGE 指标评估"""
        self.logger.info("开始 ROUGE 评估...")
        rouge = Rouge()
        results = []

        for i, pair in enumerate(qa_pairs, 1):
            self.logger.info(f"处理第 {i}/{len(qa_pairs)} 个问答对")
            try:
                result = self.qa_pipeline.ask(pair["question"])
                model_answer = self._extract_answer(result["answer"])
                scores = rouge.get_scores(model_answer, pair["answer"])[0]
                self.logger.debug(f"ROUGE 分数: {scores}")

                result.update({
                    "reference": pair["answer"],
                    "rouge": scores
                })
                results.append(result)

            except Exception as e:
                self.logger.error(f"处理问答对时出错: {str(e)}")
                continue

        self.logger.info(f"ROUGE 评估完成，成功评估 {len(results)} 个问答对")
        return results

    @observe(name="QAEvaluator.evaluate_with_bert_score", as_type="evaluator")
    def evaluate_with_bert_score(self, qa_pairs: List[Dict[str, str]]) -> List[Dict]:
        """使用 BERTScore 评估"""
        self.logger.info("开始 BERTScore 评估...")
        candidates = []
        references = []
        raw_results = []

        # 收集所有问答对
        for i, pair in enumerate(qa_pairs, 1):
            self.logger.info(f"处理第 {i}/{len(qa_pairs)} 个问答对")
            try:
                qa_result = self.qa_pipeline.ask(pair["question"])
                model_answer = self._extract_answer(qa_result["answer"])
                reference = pair["answer"]
                candidates.append(model_answer)
                references.append(reference)
                raw_results.append(qa_result)
                self.logger.debug(f"答案长度: {len(model_answer)}, 参考长度: {len(reference)}")
            except Exception as e:
                self.logger.error(f"处理问答对时出错: {str(e)}")
                continue

        if not candidates:
            self.logger.error("没有有效的问答对可供评估")
            return []

        # 计算 BERTScore
        self.logger.info("计算 BERTScore...")
        try:
            P, R, F1 = bert_score_func(
                candidates,
                references,
                lang=self.eval_config.bert_score_lang,
            )
            self.logger.debug(f"BERTScore 计算完成，样本数: {len(P)}")
        except Exception as e:
            self.logger.exception(f"BERTScore 计算失败:{e}")
            raise

        # 更新结果
        for result, p, r, f, ref in zip(raw_results, P, R, F1, references):
            result.update({
                "reference": ref,
                "bert_score": {
                    "precision": round(p.item(), 4),
                    "recall": round(r.item(), 4),
                    "f1": round(f.item(), 4)
                }
            })
            self.logger.debug(f"BERTScore - P: {p.item():.4f}, R: {r.item():.4f}, F1: {f.item():.4f}")

        self.logger.info(f"BERTScore 评估完成，成功评估 {len(raw_results)} 个问答对")
        return raw_results

    @observe(name="QAEvaluator.evaluate_with_gpt_judge", as_type="evaluator")
    def evaluate_with_gpt_judge(self, qa_pairs: List[Dict[str, str]]) -> List[Dict]:
        """使用 GPT 评估"""
        self.logger.info("开始 GPT 评估...")
        results = []

        for i, pair in enumerate(qa_pairs, 1):
            self.logger.info(f"处理第 {i}/{len(qa_pairs)} 个问答对")
            try:
                # 获取模型回答
                qa_result = self.qa_pipeline.ask(pair["question"])
                model_answer = self._extract_answer(qa_result["answer"])
                reference = pair["answer"]
                self.logger.debug(f"答案长度: {len(model_answer)}, 参考长度: {len(reference)}")

                # 构建评估提示
                prompt = self.eval_config.gpt_judge_prompt_template.format(
                    question=pair["question"],
                    model_answer=model_answer,
                    reference=reference
                )
                self.logger.debug("已构建评估提示")

                # 获取 GPT 评估结果
                try:
                    judge = self.llm.chat(
                        messages=[
                            {"role": "system", "content": self.eval_config.gpt_judge_system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        stream=False,
                        return_raw=False,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
                    ).strip()
                    self.logger.debug(f"GPT 评估结果: {judge[:100]}...")
                except Exception as e:
                    self.logger.error(f"GPT 评估失败: {str(e)}")
                    judge = f"ERROR: {str(e)}"

                qa_result.update({
                    "reference": reference,
                    "gpt_judge": judge
                })
                results.append(qa_result)

            except Exception as e:
                self.logger.error(f"处理问答对时出错: {str(e)}")
                continue

        self.logger.info(f"GPT 评估完成，成功评估 {len(results)} 个问答对")
        return results
