# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 提供在此代码库工作时所需的指导。

## 项目概述

RAGentic 是一个基于 LangGraph 构建的企业级 RAG（检索增强生成）问答系统。它在可配置的工作流图中协调知识库检索、外部工具调用、对话记忆和 LLM 生成。

**设计理念：轻 RAG，重 Agent**
- 知识库检索是 `kb_search` MCP 工具，由 Agent 在需要时主动调用
- Agent 驱动工作流，工具调用由 LLM 决定

## 命令

**包管理器**：`uv`（不是 pip）

```bash
# 安装依赖
uv sync

# 运行 Web 应用（Chainlit）
PYTHONPATH=. uv run python chainlit_app.py

# 运行 Web 应用（Gradio，遗留）
PYTHONPATH=. uv run python web_app.py

# 运行所有测试
PYTHONPATH=. uv run pytest tests/

# 运行特定测试
PYTHONPATH=. uv run pytest tests/test_pipeline_nodes.py::TestClassName::test_method_name

# 运行管道图结构测试
PYTHONPATH=. uv run pytest tests/test_pipeline_graph.py -v

# 启动 MCP 服务器
PYTHONPATH=. uv run python mcp_server.py
```

## 架构

### 管道（LangGraph）

核心在 `src/cores/pipeline_langgraph.py` — `LangGraphQAPipeline` 定义了一个有向图，其中节点是管道阶段：

```
START → agent_node → tools_node → build_context → generate_answer → update_memory → END
              ↑               │
              └─────(loop)────┘
```

- `agent_node`：带工具的 LLM，Agent 完全自主决定是否调用工具（无 use_tools 配置）
- `tools_node`：执行工具调用（MCP tools + kb_search + query_rewrite）
- `build_context`：从工具结果、KB、记忆中组装最终上下文
- `generate_answer`：生成最终答案（当不使用 agent 循环时）
- `update_memory`：持久化到混合记忆（STM + Mem0 LTM）
- `src/cores/bounded_memory_saver.py`：长时间会话的有限记忆持久化

配置（use_kb、use_memory）在 `ask()` / `ask_stream()` 入口点传入。

公共 API：
- `pipeline.ask(query, config?)` → 带 `answer` 的异步字典
- `pipeline.ask_stream(query, config?)` → 事件字典的异步生成器
- `pipeline.batch_ask(queries)` → 结果列表

### 配置系统

所有配置位于 `src/configs/`。`AppConfig`（在 `config.py` 中）是根对象，通过 Pydantic 加载 `.env`。环境变量约定：`SECTION__FIELD` 映射到 `config.section.field`（例如 `LLM__API_KEY` → `config.llm.api_key`）。

配置文件：
- `config.py`：根 `AppConfig` 类
- `model_config.py`：LLM 和嵌入模型配置
- `retrieve_config.py`：检索/搜索配置
- `memory_settings.py`：记忆行为设置
- `logger_config.py`：日志配置

`SearchConfig` 中的关键配置标志（通过 `config.retrieve`）：
- `use_kb` / `use_memory` — 启用知识库和记忆
- `use_sparse` / `use_reranker` — 检索质量增强
- `use_rewrite` / `rewrite_mode` — **[已弃用]** 查询转换现在是 Agent 工具

### LLM 模型

`src/models/llm.py`：`LLMWrapper` 支持多种 providers，通过 **litellm**（推荐）或直接使用 OpenAI SDK：

| Provider | 模型格式 | 备注 |
|----------|---------|------|
| `litellm` | `minimax/minimaxi` | 推荐，支持 100+ LLM，统一接口 |
| `openai` | `gpt-4o` | 标准 OpenAI 兼容 |
| `minimax` | MiniMax-M2.7 | 兼容 OpenAI API |

**litellm 优势**：统一接口、内置 tool calling、自动处理 provider 差异、重试/限流。

- `src/models/llm.py`：`LLMWrapper` + `OpenAILLM`
- `src/models/llm_litellm.py`：`LiteLLMWrapper`（litellm 封装，兼容 `LLMWrapper` 接口）
- `src/models/llm_response_cache.py`：LLM 语义缓存，基于 RedisVL 向量相似度匹配
- `src/models/embedding.py`：`TextEmbedding`，可选 RedisVL 缓存

#### LLM 响应缓存（语义缓存）

`src/models/llm_response_cache.py`：基于 RedisVL 向量相似度的 LLM 响应语义缓存。

- 缓存 key：query 文本的 SHA256 hash
- 存储：Redis Hash（`llm_response:` 前缀存响应）+ EmbeddingsCache（存 embedding 用于相似度搜索）
- 命中条件：余弦相似度 > 阈值（默认 0.92）
- TTL：默认 24 小时（可配置）

启用方式：
```env
LLM__USE_RESPONSE_CACHE=true
LLM__CACHE_SIMILARITY_THRESHOLD=0.92
LLM__CACHE_TTL_SECONDS=86400
```

**不缓存的请求类型**：stream=True、return_raw=True、带 tools 参数的请求（工具调用结果不适合复用）

### 向量数据库（db_services）

`src/db_services/` 通过工厂模式管理数据库访问：
- `src/db_services/base.py`：基础接口
- `src/db_services/factory.py`：数据库服务工厂
- `src/db_services/milvus/`：Milvus 实现
  - `connection.py` / `connection_manager.py`：连接管理
  - `collection.py` / `collection_manager.py`：集合管理
  - `data.py` / `data_service.py`：数据操作
  - `retrieval.py`：检索操作
  - `store.py` / `database_manager.py`：存储管理

通过 `MilvusConfig.milvus_mode` 支持两种模式：
- `"local"` — 基于 SQLite 的 milvus-lite（默认，无需服务器）
- `"remote"` — 连接到 Milvus 服务器

检索支持稠密（语义）、稀疏（BM25）和混合搜索，可选 BGE 重排。

### 工具 & MCP

外部工具（天气、网络搜索、kb_search、query_rewrite）通过 MCP 协议调用：
- `src/mcp/mcp_client.py`：MCP 客户端用于工具调用
- `src/mcp/server/`：MCP 服务器实现（`kb_tools.py`、`weather.py`、`web_crawl.py`）
- `src/agent/tools.py`：用于网页抓取的 `WebSpider`

主要工具：
- `kb_search(query, top_k, use_hyde)`：知识库检索，支持通过 `use_hyde` 参数启用 HyDE 增强
- `query_rewrite(query, mode)`：查询转换工具，支持 `rewrite` / `step_back` / `sub_query` 模式
- `weather_get_alerts` / `weather_get_forecast`：天气信息
- `web_crawl(url)`：网页内容提取
- `read_skill(name)`：技能指令读取器

HyDE（Hypothetical Document Embeddings）是一种检索增强技术 — 当 `use_hyde=True` 时，工具生成假设答案嵌入以获得更好的语义搜索。Agent 根据查询复杂度决定何时使用。

### 记忆（混合）

两层：
- **短期记忆（STM）**：会话内的 LangGraph 消息历史（通过 `QAState.messages`）
- **长期记忆（LTM）**：Mem0 集成（`src/memory/mem0_manager.py`）+ Milvus 向量存储

核心服务：
- `src/memory/hybrid_memory_service.py`：统一记忆接口（主要）
- `src/memory/base.py`：抽象基类
- `src/memory/short_term_memory.py`：STM 实现
- `src/memory/long_term_memory.py`：LTM 实现
- `src/memory/mem0_manager.py`：Mem0 集成
- `src/cores/bounded_memory_saver.py`：长时间会话的有限记忆持久化

### 技能系统

`src/skills/skill_manager.py` 和 `skills/` 目录管理可扩展的技能：
- 技能作为额外上下文注入到系统提示中
- 目录：`skills/` 包含技能定义文件

### Web 应用

- `chainlit_app.py`：Chainlit 5 应用（主要，推荐）
- `web_app.py`：Gradio 5 应用（遗留）

安全性：`src/utils/security.py` 处理输入验证（XSS、SQL 注入、路径遍历）。

## 关键约定

- **异步**：嵌入、工具调用和管道都是异步的；暴露同步和异步流接口
- **可观测性**：关键函数使用 `@langfuse_context.observe()` 装饰器
- **配置注入**：管道组件在初始化时接收 `AppConfig` — 不要直接读取环境变量
- **PYTHONPATH**：直接运行脚本时必须设置为 `.`（导入使用 `src.` 前缀）
- **主动文档化**：实现功能后，立即更新 CLAUDE.md 和相关文档 — 不要等待被要求

## 环境设置

复制 `.env.example` 到 `.env` 并配置：

```env
# LLM Provider（litellm 推荐，支持 100+ 模型）
LLM__PROVIDER=litellm
LLM__MODEL=minimax/minimaxi
LLM__BASE_URL=https://api.minimax.chat/v1
LLM__API_KEY=your_api_key

# Embedding
EMBEDDING__BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING__API_KEY=your_embedding_key
```

可选：Langfuse（可观测性）、Redis（嵌入缓存）、远程 Milvus（生产向量数据库）。

## Docker

```bash
docker-compose up -d                                           # 基本
docker-compose --profile with-redis up -d                     # + Redis 缓存
docker-compose --profile with-redis --profile with-milvus up -d  # + 生产 Milvus
```

Chainlit 应用访问地址：`http://localhost:7860`。
