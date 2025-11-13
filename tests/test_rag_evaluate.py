# -*- coding: utf-8 -*-
"""
@Time ： 2025/6/4 01:11
@Auth ： 吕鑫
@File ：test_rag_evaluate.py
@IDE ：PyCharm
"""
from src.evaluate.rag import RAGASEvaluator


def test_ragas_basic():

    qa_data = [
        {
            "query": "中国的首都是哪？",
            "prediction": "中国的首都是北京。",
            "contexts": [
                "北京是中国的首都，也是政治中心。",
                "上海是中国的经济中心。"
            ],
            "ground_truths": ["中国的首都是北京。"]
        },
        {
            "query": "苹果公司是谁创立的？",
            "prediction": "乔布斯创立了苹果。",
            "contexts": [
                "史蒂夫·乔布斯和沃兹尼亚克在1976年共同创立了苹果公司。"
            ],
            "ground_truths": ["史蒂夫·乔布斯和沃兹尼亚克共同创立了苹果。"]
        }
    ]

    evaluator = RAGASEvaluator()
    results = evaluator.evaluate(qa_data)
    print("RAGAS 评估结果：\n", results)


if __name__ == "__main__":
    test_ragas_basic()
