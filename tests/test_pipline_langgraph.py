# -*- coding: utf-8 -*-
"""
@Time ： 2025/6/8 10:27
@Auth ： 吕鑫
@File ：test_pipline_langgraph.py
@IDE ：PyCharm
"""

from src.config.config import QAPipelineConfig
from src.cores.pipline_langgraph import LangGraphQAPipeline

"""使用示例"""
# 假设有配置对象
config = QAPipelineConfig()

# 初始化流水线
pipeline = LangGraphQAPipeline(config)

# 导出图结构
# pipeline.export_graph()

# 使用tool
result = pipeline.ask("美国纽约今天天气如何，同时帮我看看百度首页有什么新闻？", use_tools=True)
print(result)

# result = pipeline.ask("你还记得我叫什么嘛？", use_memory=True)
# print(result)

# 流式问答
# for event in pipeline.ask_stream("解释深度学习的原理", use_knowledge_base=True):
#     print(event)
