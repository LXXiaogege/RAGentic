# RAGentic - 基于 LangGraph 的智能问答系统

> 🚀 基于 LangGraph 的 RAG 学习项目，集成检索、工具调用与对话记忆

## 项目简介

RAGentic 是一个企业级 RAG（检索增强生成）智能问答系统，基于 LangGraph 构建复杂工作流。支持知识库检索、工具调用、多轮对话记忆，并提供同步/流式输出，可选接入 Langfuse 进行全链路观测。

**设计理念：Light RAG, Heavy Agent**

- 知识库检索成为 `kb_search` MCP 工具，Agent 按需主动调用
- Agent 主导工作流，工具调用由 LLM 决定

## 主要特性

- 🔄 **查询改写** - 支持 rewrite/step-back/sub-query/HyDE 多种模式
- 🔍 **混合检索** - 稠密 + 稀疏检索，支持 Rerank 重排序
- 🛠️ **工具调用** - 基于 MCP 协议，支持天气、搜索等外部工具
- 💬 **对话记忆** - 分层记忆（短期滑动窗口 + 长期 Mem0 向量存储）
- ⚡ **流式输出** - 支持节点级流式响应
- 📊 **可观测性** - Langfuse 深度集成，完整追踪链路

## 技术栈

| 类别 | 技术 |
|------|------|
| 核心框架 | LangGraph, LangChain |
| 向量数据库 | Milvus |
| 模型服务 | litellm (支持 100+ LLM) |
| 工具协议 | MCP (Model Context Protocol) |
| 记忆存储 | Mem0 + Milvus |
| 前端框架 | Next.js 14 + TypeScript + React |
| API 服务 | FastAPI + Uvicorn |
| 可观测性 | Langfuse |

## 环境要求

- Python 3.10+
- Milvus（本地模式或远程服务）
- Redis（可选，用于缓存）
- 可访问的 LLM 与 Embedding 服务（支持 litellm，统一接入 100+ LLM）

## 快速开始

### 1. 安装依赖

```bash
uv sync
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并填写配置：

```bash
cp .env.example .env
```

### 3. 配置说明

所有配置支持环境变量覆盖，使用 `__` 分隔：

```ini
# LLM 配置（推荐使用 litellm，支持 100+ 模型）
LLM__PROVIDER = litellm
LLM__MODEL = minimax/minimaxi
LLM__BASE_URL = https://api.minimax.chat/v1
LLM__API_KEY = your_api_key

# Embedding 配置
EMBEDDING__BASE_URL = https://api.example.com/v1
EMBEDDING__API_KEY = your_api_key

# Langfuse 配置（可选）
LANGFUSE__HOST = https://cloud.langfuse.com
LANGFUSE__PUBLIC_KEY = your_public_key
LANGFUSE__SECRET_KEY = your_secret_key

# Milvus 配置
MILVUS__MILVUS_MODE = local
MILVUS__VECTOR_DB_URI = data/knowledge_db/db/rag.db
MILVUS__MEMORY_DB_URI = data/memory.db

# 知识库路径
RETRIEVE__KB_PATH = data/knowledge_db/psychology
RETRIEVE__TOP_K = 5
```

### 4. 运行服务

需要同时启动后端 API 服务和前端界面：

**终端 1 - 启动后端 API 服务：**
```bash
uv sync
PYTHONPATH=. uv run python api_server.py
```
后端服务运行在 http://localhost:8000

**终端 2 - 启动前端界面：**
```bash
cd frontend
npm install
npm run dev
```
前端界面运行在 http://localhost:3000

访问 http://localhost:3000 即可使用。

## 项目结构

```
RAGentic/
├── src/
│   ├── a2a/                # A2A 协议集成
│   │   ├── qa_agent.py
│   │   └── qa_agent_executor.py
│   ├── agent/              # Agent 逻辑与工具
│   │   └── tools.py        # WebSpider 等工具
│   ├── configs/            # 配置管理
│   │   ├── config.py       # 主配置类 (AppConfig)
│   │   ├── model_config.py
│   │   ├── memory_settings.py
│   │   ├── retrieve_config.py
│   │   ├── logger_config.py
│   │   └── ...
│   ├── cores/              # 核心流水线
│   │   ├── pipeline_langgraph.py  # LangGraph 流水线
│   │   ├── query_transformer.py   # 查询转换
│   │   ├── message_builder.py     # 消息构建
│   │   └── bounded_memory_saver.py
│   ├── models/             # 模型封装
│   │   ├── llm.py          # LLMWrapper + OpenAILLM
│   │   ├── llm_litellm.py  # LiteLLMWrapper (推荐)
│   │   ├── llm_response_cache.py  # LLM 语义缓存
│   │   └── embedding.py    # Embedding 封装
│   ├── db_services/        # 数据库服务 (工厂模式)
│   │   ├── base.py
│   │   ├── factory.py
│   │   └── milvus/         # Milvus 实现
│   │       ├── connection.py
│   │       ├── collection.py
│   │       ├── data.py
│   │       ├── retrieval.py
│   │       └── store.py
│   ├── mcp/                 # MCP 协议
│   │   ├── mcp_client.py   # MCP 客户端
│   │   └── server/         # MCP 服务端
│   │       ├── kb_tools.py
│   │       ├── weather.py
│   │       └── web_crawl.py
│   ├── memory/              # 分层记忆模块
│   │   ├── base.py
│   │   ├── short_term_memory.py
│   │   ├── long_term_memory.py
│   │   ├── hybrid_memory_service.py
│   │   └── mem0_manager.py
│   ├── data_process/        # 数据处理
│   │   └── loaders/
│   ├── evaluate/            # RAG 评估
│   ├── skills/              # Agent 技能
│   └── utils/               # 工具函数
├── frontend/                # Next.js 前端
│   ├── src/
│   │   ├── app/            # App Router 页面
│   │   │   ├── chat/       # 聊天页面
│   │   │   └── layout.tsx  # 根布局
│   │   ├── components/      # React 组件
│   │   │   ├── chat/       # 聊天组件
│   │   │   ├── settings/   # 设置面板
│   │   │   └── ui/         # shadcn/ui 组件
│   │   └── lib/            # 工具库 (api, store, types)
│   └── package.json
├── api_server.py            # FastAPI 后端服务
├── tests/                   # 测试用例
├── data/                    # 数据目录
├── skills/                  # 技能定义
├── pyproject.toml            # 项目依赖
└── docker-compose.yml        # Docker 配置
```

## 架构设计

### Pipeline 工作流

```
parse_query → agent_node → tools_node → build_context → generate_answer → update_memory
                    ↑
                    └──(loop if tool_calls)
```

**工作流节点：**

1. `parse_query` - 初始化状态参数
2. `agent_node` - LLM with tools，循环调用直到无 tool_calls
3. `tools_node` - 执行 MCP 工具调用（kb_search、天气等）
4. `build_context` - 整合 KB/记忆/工具结果构建最终上下文
5. `generate_answer` - 生成最终答案
6. `update_memory` - 刷写到分层记忆 (STM + Mem0 LTM)

### MCP 工具

| 工具 | 说明 |
|------|------|
| `kb_search` | 搜索知识库 |
| `weather_get_alerts` | 天气预警 |
| `weather_get_forecast` | 天气预报 |
| `web_crawl` | 网页爬取 |
| `read_skill` | 读取技能指令 |

### 分层记忆架构

```
HybridMemoryService
├── ShortTermMemory (LangGraph 消息历史 via QAState.messages)
└── LongTermMemory (Mem0 + Milvus 向量存储)
```

## 配置项说明

### 检索配置 (SearchConfig)

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `use_kb` | bool | False | 是否启用知识库检索 |
| `use_tool` | bool | False | 是否启用工具调用 |
| `use_memory` | bool | False | 是否启用对话记忆 |
| `use_sparse` | bool | False | 是否启用稀疏检索（BM25） |
| `use_reranker` | bool | False | 是否启用重排序 |
| `use_rewrite` | bool | False | 是否启用查询改写 |
| `rewrite_mode` | str | "rewrite" | 改写模式 |
| `retriever_type` | str | "dense" | dense / sparse / hybrid |
| `top_k` | int | 3 | 检索返回数量 |
| `memory_window_size` | int | 5 | 对话记忆窗口大小 |
| `use_think` | bool | True | 是否跳过推理过程 |
| `max_agent_iterations` | int | 10 | Agent 最大迭代次数 |

### 记忆配置 (MemorySettings)

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `service_type` | str | "hybrid" | hybrid / stm_only / ltm_only |
| `stm_window_size` | int | 10 | 短期记忆窗口大小 |
| `ltm_persist_threshold` | int | 3 | 刷写到长期记忆的阈值 |
| `enable_stm` | bool | True | 是否启用短期记忆 |
| `enable_ltm` | bool | True | 是否启用长期记忆 |

## 测试

```bash
# 运行所有测试
PYTHONPATH=. uv run pytest tests/

# 运行特定测试文件
PYTHONPATH=. uv run pytest tests/test_pipeline_langgraph.py -v

# 运行带 async 的测试
PYTHONPATH=. uv run pytest tests/test_pipeline_nodes.py -v
```

## Web 界面功能

- ⚡ 流式/同步输出切换
- 📚 知识库检索开关
- 🌐 联网/工具调用开关
- 💬 上下文记忆开关
- 🎯 Reranker 重排序开关
- 🧠 深度思考模式（R1）
- 📊 实时节点执行状态显示
- 🆕 新对话 / 清空记忆快捷按钮
- 📝 快捷示例点击发送

## Docker 部署

```bash
# 基础部署
docker-compose up -d

# + Redis 缓存
docker-compose --profile with-redis up -d

# + 生产级 Milvus
docker-compose --profile with-redis --profile with-milvus up -d
```

## 开发指南

详细开发规范请参考 [AGENTS.md](./AGENTS.md)。

### 代码规范

- 使用 `uv` 管理依赖
- 使用 `ruff` 进行代码格式化和检查
- 使用 Python 3.10+ 类型注解
- 使用中文注释（项目约定）

### 常用命令

```bash
# 安装依赖
uv sync

# 代码检查
uv run ruff check src/ tests/

# 代码格式化
uv run ruff format src/ tests/

# 类型检查
uv run mypy src/
```

## 故障排查

### Milvus 连接失败

```bash
# 安装 milvus-lite
uv add 'pymilvus[milvus-lite]'

# 或切换到远程模式
MILVUS__MILVUS_MODE=remote
MILVUS__VECTOR_DB_URI=http://localhost:19530
```

### 检索结果为空

- 检查知识库目录是否存在
- 确认 Embedding 模型与文档嵌入时一致
- 调整 `top_k` 增大检索数量

### Langfuse 追踪不工作

确保环境变量正确设置，或留空 `LANGFUSE__SECRET_KEY` 临时禁用。

## 性能优化建议

1. **启用缓存**：RedisVL 缓存已默认启用，确保 Redis 服务可用
2. **调整 top_k**：根据实际需求调整，默认 5 是平衡点
3. **使用混合检索**：`use_sparse=True` 可提升召回率
4. **批量处理**：使用 `batch_ask()` 而非循环调用
5. **异步加载**：文档添加使用异步接口

## 许可证

MIT License
