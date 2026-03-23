# RAGentic - 基于 LangGraph 的智能问答系统

> 🚀 基于 LangGraph 的 RAG 学习项目，集成检索、工具调用与对话记忆

### 项目简介

RAGentic 是一个企业级 RAG（检索增强生成）智能问答系统，基于 LangGraph 构建复杂工作流。支持知识库检索、工具调用、多轮对话记忆，并提供同步/流式输出，可选接入 Langfuse 进行全链路观测。

### ✨ 主要特性

- 🔄 **查询改写** - 支持 rewrite/step-back/sub-query/HyDE 多种模式
- 🔍 **混合检索** - 稠密 + 稀疏检索，支持 Rerank 重排序
- 🛠️ **工具调用** - 基于 MCP 协议，支持天气、搜索等外部工具
- 💬 **对话记忆** - 短期记忆（多轮对话）+ 长期记忆（Mem0）
- ⚡ **流式输出** - 支持节点级流式响应
- 📊 **可观测性** - Langfuse 深度集成，完整追踪链路

### 📋 环境要求

- Python 3.10+
- Milvus（本地模式或远程服务）
- 可访问的 LLM 与 Embedding 服务（支持 OpenAI 兼容接口）

### 🚀 快速开始

#### 1. 安装依赖

```bash
uv sync
```

#### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并填写配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```ini
# LLM 配置
LLM__API_KEY=your_api_key
LLM__BASE_URL=https://api.example.com/v1

# Embedding 配置
EMBEDDING__BASE_URL=https://api.example.com/v1
EMBEDDING__API_KEY=your_api_key

# Langfuse 配置（可选）
LANGFUSE__HOST=https://cloud.langfuse.com
LANGFUSE__PUBLIC_KEY=your_public_key
LANGFUSE__SECRET_KEY=your_secret_key
```

#### 3. 运行示例

**同步问答：**

```python
from src.configs.config import AppConfig
from src.cores.pipeline_langgraph import LangGraphQAPipeline

config = AppConfig()
config.retrieve.use_kb = True  # 启用知识库检索

pipeline = LangGraphQAPipeline(config)
result = pipeline.ask("什么是 RAG？")

print(result["answer"])
```

**流式问答：**

```python
from src.configs.config import AppConfig
from src.cores.pipeline_langgraph import LangGraphQAPipeline

config = AppConfig()
pipeline = LangGraphQAPipeline(config)

for event in pipeline.ask_stream("什么是 RAG？"):
    if event.get("node") == "generate_answer" and event.get("status") == "complete":
        print(event["answer"])
```

**启用工具调用：**

```python
config.retrieve.use_tool = True  # 启用工具调用
config.retrieve.use_kb = True    # 启用知识库
config.retrieve.use_memory = True  # 启用对话记忆

pipeline = LangGraphQAPipeline(config)
result = pipeline.ask("洛杉矶今天天气怎么样？")
```

### 🎯 使用 LangGraph 工作流

LangGraph 提供了清晰的工作流可视化和状态管理：

```python
# 导出工作流图
pipeline.export_graph("workflow.png")

# 查看 Mermaid 格式的工作流
mermaid_code = pipeline.export_graph()
print(mermaid_code)
```

工作流包含以下节点：
1. `parse_query` - 解析查询参数
2. `transform_query` - 查询改写（可选）
3. `retrieve_knowledge` - 知识库检索
4. `call_tools` - 工具调用
5. `build_context` - 构建上下文
6. `generate_answer` - 生成答案
7. `update_memory` - 更新记忆

### 📁 项目结构

```
RAGentic/
├── src/
│   ├── configs/          # 配置管理
│   │   ├── config.py     # 主配置类
│   │   ├── model_config.py
│   │   └── database_config.py
│   ├── cores/            # 核心流水线
│   │   ├── pipeline_langgraph.py  # LangGraph 流水线
│   │   ├── query_transformer.py   # 查询转换
│   │   └── message_builder.py     # 消息构建
│   ├── models/           # 模型封装
│   │   ├── llm.py        # LLM 包装器
│   │   └── embedding.py  # Embedding 封装
│   ├── db_services/      # 数据库服务
│   │   └── milvus/       # Milvus 向量数据库
│   ├── mcp/              # MCP 工具调用
│   ├── memory/           # 对话记忆
│   ├── data_process/     # 数据处理
│   └── evaluate/         # 评估模块
├── tests/                # 测试用例
├── data/                 # 数据目录
│   ├── knowledge_db/     # 知识库文档
│   └── models/           # 模型文件
├── .env.example          # 环境变量示例
└── pyproject.toml        # 项目依赖
```

### 🔧 配置说明

所有路径配置支持相对路径（相对于项目根目录）：

```ini
# 路径配置（可选）
RERANK__MODEL_PATH=data/models/bge-reranker-base
BM25__MODEL_DIR=data/knowledge_db/bm25_model
MILVUS__VECTOR_DB_URI=data/knowledge_db/db/rag.db
RETRIEVE__KB_PATH=data/knowledge_db/psychology
```

支持环境变量覆盖：

```bash
export RERANK__MODEL_PATH=/custom/path/to/rerank
export MILVUS__VECTOR_DB_URI=/custom/path/to/db
```

### 📊 测试

运行 LangGraph 流水线测试：

```bash
PYTHONPATH=. uv run python tests/test_pipeline_langgraph.py
```

### 🌐 Web 界面

启动 Gradio Web 应用：

```bash
PYTHONPATH=. uv run python web_app.py
```

访问 http://127.0.0.1:7860

### 📝 技术栈

- **核心框架**: LangGraph, LangChain
- **向量数据库**: Milvus
- **模型服务**: OpenAI 兼容接口
- **可观测性**: Langfuse
- **工具协议**: MCP (Model Context Protocol)
- **Web 界面**: Gradio

### 🤝 贡献

欢迎提交 Issue 和 Pull Request！

### 📄 许可证

MIT License
