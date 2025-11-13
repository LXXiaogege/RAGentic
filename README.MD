### 项目简介
RAG学习项目，随缘更新，包含问答流水线，集成检索、工具调用与对话记忆，支持同步与流式输出，并可选接入 Langfuse 进行观测。


### 环境要求
- Python 3.10+
- Milvus
- 可访问的 LLM 与 Embedding 服务

### 安装与运行
1) 安装依赖
```
uv sync
```


2) 配置
- 编辑 `src/config/config.py`，填入 LLM、Embedding、Milvus、Langfuse 等参数。

3) 运行示例
- 同步问答：参考 `tests/test_rag.py`
- LangGraph 流式：参考 `tests/test_pipline_langgraph.py`

### 快速开始（代码片段）
```python
from src.config.config import QAPipelineConfig
from src.cores.pipline_langgraph import LangGraphQAPipeline

config = QAPipelineConfig()
pipeline = LangGraphQAPipeline(config)
res = pipeline.ask("什么是RAG？")
print(res["answer"]) 
```

### 主要能力
- 查询重写（支持 HYDE）
- 稀疏/稠密检索与重排
- 可选工具调用（MCP）
- 上下文构建与记忆
- 同步/流式输出
- Langfuse 观测（可选）

### 目录结构（简略）
- `src/models/` LLM 与 Embedding 封装
- `src/cores/` 流水线、DB 管理、查询转换
- `src/data/` 加载与处理
- `src/mcp/` 工具调用
- `tests/` 用例与示例

