# RAGentic - 基于 LangGraph 的智能问答系统

> 🚀 基于 LangGraph 的 RAG 学习项目，集成检索、工具调用与对话记忆

### 项目简介

RAGentic 是一个企业级 RAG（检索增强生成）智能问答系统，基于 LangGraph 构建复杂工作流。支持知识库检索、工具调用、多轮对话记忆，并提供同步/流式输出，可选接入 Langfuse 进行全链路观测。

### ✨ 主要特性

- 🔄 **查询改写** - 支持 rewrite/step-back/sub-query/HyDE 多种模式
- 🔍 **混合检索** - 稠密 + 稀疏检索，支持 Rerank 重排序
- 🛠️ **工具调用** - 基于 MCP 协议，支持天气、搜索等外部工具
- 💬 **对话记忆** - 分层记忆（短期滑动窗口 + 长期 Mem0 向量存储）
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
3. `check_retrieve_knowledge` - 决定是否进入知识库检索
4. `retrieve_knowledge` - 知识库检索
5. `check_call_tools` - 决定是否进入工具/Agent 循环
6. `agent_node` - Agent 推理并生成 tool 调用请求（use_tool=True 时）
7. `tools_node` - 执行 MCP 工具调用并回填 tool 结果
8. `build_context` - 构建最终上下文
9. `generate_answer` - 生成答案
10. `update_memory` - 更新记忆
11. `handle_error` - 异常处理分支（错误时）

### 📁 项目结构

```
RAGentic/
├── src/
│   ├── configs/          # 配置管理
│   │   ├── config.py     # 主配置类
│   │   ├── model_config.py
│   │   ├── database_config.py
│   │   └── memory_settings.py  # 记忆服务配置
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
│   ├── memory/           # 分层记忆模块
│   │   ├── base.py                    # 抽象基类
│   │   ├── short_term_memory.py       # 短期记忆（滑动窗口）
│   │   ├── long_term_memory.py        # 长期记忆（Mem0）
│   │   ├── hybrid_memory_service.py    # 混合记忆服务
│   │   └── mem0_manager.py             # Mem0 封装
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
MILVUS__MEMORY_DB_URI=data/memory.db
MEM0__HISTORY_DB_PATH=data/memory/history.db
RETRIEVE__KB_PATH=data/knowledge_db/psychology
```

支持环境变量覆盖：

```bash
export RERANK__MODEL_PATH=/custom/path/to/rerank
export MILVUS__VECTOR_DB_URI=/custom/path/to/db
export MILVUS__MEMORY_DB_URI=/custom/path/to/memory.db
export MEM0__HISTORY_DB_PATH=/custom/path/to/history.db
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

---

## 📚 API 参考

### LangGraphQAPipeline 核心方法

#### `ask(query: str, **kwargs) -> Dict[str, Any]`

同步问答接口，执行完整的 RAG 工作流。

**参数：**
- `query` (str): 用户问题
- `thread_id` (str, optional): 会话 ID，用于多轮对话记忆，默认 "0"
- `langfuse_session_id` (str, optional): Langfuse 追踪会话 ID
- `langfuse_user_id` (str, optional): Langfuse 追踪用户 ID
- `messages` (list, optional): 历史消息列表

**返回值：**
```python
{
    "question": str,           # 原始问题
    "answer": str,             # 生成的答案
    "messages": list,          # 完整消息历史
    "transformed_query": str,  # 改写后的查询（如有）
    "context": str,            # 最终上下文
    "kb_context": str,         # 知识库检索内容
    "tool_context": str,       # 工具调用结果
    "tool_calls_history": list,  # 工具调用历史（agent loop）
    "agent_iterations": int       # Agent 迭代次数
}
```

**示例：**
```python
from src.configs.config import AppConfig
from src.cores.pipeline_langgraph import LangGraphQAPipeline

config = AppConfig()
config.retrieve.use_kb = True
config.retrieve.use_memory = True

pipeline = LangGraphQAPipeline(config)
result = pipeline.ask(
    "什么是 RAG？",
    thread_id="session_123",
    langfuse_session_id="trace_456"
)
print(result["answer"])
```

---

#### `ask_stream(query: str, **kwargs) -> Generator[Dict, None, None]`

流式问答接口，逐节点返回执行状态和答案片段。

**参数：** 同 `ask()`

**返回值：** 生成器，每次 yield 一个事件字典：
```python
# 节点处理中（每个节点都会 yield 一次）
{"node": "build_context", "status": "processing", "state": {"kb_context": "...", "tool_context": "...", "final_context": "..."}}

# 答案生成完成（仅在 agent_node / generate_answer 节点触发）
{"node": "generate_answer", "status": "complete", "answer": "...", "context": "..."}

# 错误
{"node": "<节点名>", "status": "error", "error": "错误信息"}
```

**示例：**
```python
for event in pipeline.ask_stream("解释深度学习原理"):
    if event.get("status") == "complete":
        print(event["answer"])
```

---

#### `batch_ask(questions: List[str], **kwargs) -> List[Dict[str, Any]]`

批量问答接口，依次处理多个问题。

**参数：**
- `questions`: 问题列表
- 其他参数同 `ask()`

**返回值：** 答案列表，结构与 `ask()` 相同

---

#### `export_graph(output_path: str) -> str | None`

导出 LangGraph 工作流图为 Mermaid 格式。

**参数：**
- `output_path`: 输出文件路径（.mmd 或 .png）

**返回值：** Mermaid 代码字符串，或失败时返回 None

**示例：**
```python
mermaid_code = pipeline.export_graph("workflow.mmd")
# 或使用 https://mermaid.live 在线查看
```

---

## 🔧 配置项详解

### 检索配置 (SearchConfig)

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `use_kb` | bool | False | 是否启用知识库检索 |
| `use_tool` | bool | False | 是否启用工具调用 |
| `use_memory` | bool | False | 是否启用对话记忆 |
| `use_sparse` | bool | False | 是否启用稀疏检索（BM25） |
| `use_reranker` | bool | False | 是否启用重排序 |
| `use_rewrite` | bool | False | 是否启用查询改写 |
| `rewrite_mode` | str | "rewrite" | 改写模式：rewrite/step_back/sub_query/hyde（知识库查询下有效） |
| `num_hypo` | int | 3 | HyDE 模式下生成假设答案数量 |
| `retriever_type` | str | "dense" | dense / sparse / hybrid |
| `top_k` | int | 3 | 检索返回数量 |
| `memory_window_size` | int | 5 | 对话记忆窗口大小 |
| `kb_path` | str | "data/knowledge_db/psychology" | 知识库目录路径 |
| `use_think` | bool | True | 是否跳过推理过程（R1） |
| `max_agent_iterations` | int | 10 | Agent 工具调用循环最大迭代次数 |

**环境变量：** `RETRIEVE__KB_PATH`, `RETRIEVE__TOP_K` 等

---

### 模型配置

#### LLMConfig
| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `provider` | str | "openai" | 模型提供商 |
| `model` | str | "gpt-3.5-turbo" | 模型名称 |
| `api_key` | str | - | API 密钥 |
| `base_url` | str | - | API 基础 URL |

**环境变量：** `LLM__API_KEY`, `LLM__BASE_URL`

#### EmbeddingConfig
| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `model` | str | "text-embedding-ada-002" | 嵌入模型 |
| `api_key` | str | - | API 密钥 |
| `base_url` | str | - | API 基础 URL |

**环境变量：** `EMBEDDING__API_KEY`, `EMBEDDING__BASE_URL`

---

### 数据库配置 (MilvusConfig)

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `milvus_mode` | str | "local" | 模式：local/remote |
| `vector_db_uri` | str | "data/knowledge_db/db/rag.db" | 本地数据库路径 |
| `db_name` | str | "test" | 远程数据库名称 |
| `collection_name` | str | "pys" | 集合名称 |
| `vector_dimension` | int | 1024 | 向量维度 |
| `memory_collection_name` | str | "mem0_with_milvus" | mem0 历史集合名称 |
| `memory_db_uri` | str | "data/memory.db" | mem0 历史记录数据库路径 |

**环境变量：** `MILVUS__VECTOR_DB_URI`, `MILVUS__MEMORY_DB_URI`

---

### Langfuse 配置 (LangfuseConfig)

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `host` | str | "https://cloud.langfuse.com" | Langfuse 服务地址 |
| `public_key` | str | - | 公钥 |
| `secret_key` | str | - | 私钥 |

**环境变量：** `LANGFUSE__HOST`, `LANGFUSE__PUBLIC_KEY`, `LANGFUSE__SECRET_KEY`

---

### 记忆配置 (MemorySettings)

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `service_type` | str | "hybrid" | 服务类型: hybrid / stm_only / ltm_only |
| `stm_window_size` | int | 10 | 短期记忆窗口大小 |
| `ltm_persist_threshold` | int | 3 | 刷写到长期记忆的阈值（轮数） |
| `enable_stm` | bool | True | 是否启用短期记忆 |
| `enable_ltm` | bool | True | 是否启用长期记忆 |
| `ltm_search_limit` | int | 5 | 长期记忆搜索结果数量限制 |
| `ltm_search_threshold` | float | None | 长期记忆搜索相似度阈值 |

**分层记忆架构：**
- **短期记忆 (STM)**: 基于滑动窗口 + LangGraph Checkpoint，当前会话快速访问
- **长期记忆 (LTM)**: 基于 Mem0 + Milvus 向量存储，跨会话持久化

---

## 🐛 故障排查

### 常见问题

#### 1. Milvus 连接失败

**错误信息：**
```
Milvus 本地连接失败：缺少 `milvus_lite` 依赖
```

**解决方案：**
```bash
# 安装 milvus-lite 可选依赖
uv add 'pymilvus[milvus-lite]'

# 或切换到远程模式，修改 .env
MILVUS__MILVUS_MODE=remote
MILVUS__VECTOR_DB_URI=http://localhost:19530
```

---

#### 2. 检索结果为空

**可能原因：**
- 知识库未正确加载
- 嵌入模型不匹配
- 查询词与文档差异过大

**排查步骤：**
```bash
# 1. 检查知识库目录是否存在
ls data/knowledge_db/

# 2. 检查集合是否已创建
PYTHONPATH=. uv run python -c "
from src.configs.config import AppConfig
from src.db_services.milvus.connection_manager import MilvusConnectionManager
config = AppConfig()
db = MilvusConnectionManager(...)
print(db.list_databases())
"

# 3. 尝试简单查询测试
pipeline.ask("测试问题", use_kb=True)
```

**解决方案：**
- 重新添加文档到知识库
- 确保 Embedding 模型与文档嵌入时一致
- 调整 `top_k` 增大检索数量

---

#### 3. Langfuse 追踪不工作

**错误信息：**
```
Langfuse 客户端初始化失败：...
```

**排查步骤：**
1. 检查环境变量是否正确设置
2. 验证 API Key 是否有效
3. 确认网络可访问 Langfuse 服务

**解决方案：**
```bash
# 在 .env 中设置
LANGFUSE__HOST=https://cloud.langfuse.com
LANGFUSE__PUBLIC_KEY=pk-xxx
LANGFUSE__SECRET_KEY=sk-xxx

# 或临时禁用 Langfuse（不影响核心功能）
# 留空 LANGFUSE__SECRET_KEY 即可
```

---

#### 4. Reranker 模型加载失败

**错误信息：**
```
初始化 reranker 失败，将禁用 reranker：Can't load the configuration of '...'
```

**解决方案：**
```bash
# 1. 检查模型路径是否正确
ls data/models/bge-reranker-base/

# 2. 重新下载模型
# 或修改 .env 使用 HuggingFace 自动下载
RERANK__MODEL_PATH=BAAI/bge-reranker-base

# 3. 临时禁用 reranker
# 设置 use_reranker=False
```

---

#### 5. Web 应用启动失败

**错误信息：**
```
Port 7860 is already in use
```

**解决方案：**
```bash
# 修改启动端口
# 在 web_app.py 最后修改：
demo.launch(server_port=7861)

# 或查找并终止占用端口的进程
lsof -i :7860
kill -9 <PID>
```

---

#### 6. 内存不足（OOM）

**场景：** 处理大文件或长对话时

**解决方案：**
```python
# 1. 减小 batch size
config.splitter.chunk_size = 500

# 2. 限制对话记忆窗口
config.retrieve.memory_window_size = 5

# 3. 使用流式输出避免一次性加载
for event in pipeline.ask_stream(query):
    ...
```

---

## 📊 性能优化建议

1. **启用缓存：** RedisVL 缓存已默认启用，确保 Redis 服务可用
2. **调整 top_k：** 根据实际需求调整，默认 5 是平衡点
3. **使用混合检索：** `use_sparse=True` 可提升召回率
4. **批量处理：** 使用 `batch_ask()` 而非循环调用 `ask()`
5. **异步加载：** 文档添加使用 `await add_documents_from_dir()`

---

## 🆘 获取帮助

- **GitHub Issues:** 提交 Bug 报告或功能请求
- **项目文档：** 查看 `AGENTS.md` 了解开发规范
- **日志排查：** 检查 `logs/` 目录下的详细日志
