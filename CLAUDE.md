# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAGentic is an enterprise-grade RAG (Retrieval-Augmented Generation) Q&A system built on LangGraph. It orchestrates knowledge base retrieval, external tool invocation, conversation memory, and LLM generation in a configurable workflow graph.

**Design Philosophy: Light RAG, Heavy Agent**
- Knowledge base retrieval is a `kb_search` MCP tool, actively called by Agent when needed
- Agent drives the workflow, tool calls are determined by LLM

## Commands

**Package manager**: `uv` (not pip)

```bash
# Install dependencies
uv sync

# Run the web app (Chainlit)
PYTHONPATH=. uv run python chainlit_app.py

# Run the web app (Gradio, legacy)
PYTHONPATH=. uv run python web_app.py

# Run all tests
PYTHONPATH=. uv run pytest tests/

# Run a specific test
PYTHONPATH=. uv run pytest tests/test_pipeline_nodes.py::TestClassName::test_method_name

# Run pipeline graph structure test
PYTHONPATH=. uv run pytest tests/test_pipeline_graph.py -v

# Start the MCP server
PYTHONPATH=. uv run python mcp_server.py
```

## Architecture

### Pipeline (LangGraph)

The core is `src/cores/pipeline_langgraph.py` — `LangGraphQAPipeline` defines a directed graph where nodes are pipeline stages:

```
START → agent_node → tools_node → build_context → generate_answer → update_memory → END
              ↑               │
              └─────(loop)────┘
```

- `agent_node`: LLM with tools, Agent 完全自主决定是否调用工具（无 use_tools 配置）
- `tools_node`: Execute tool calls (MCP tools + kb_search + query_rewrite)
- `build_context`: Assemble final context from tool results, KB, memory
- `generate_answer`: Generate final answer (when not using agent loop)
- `update_memory`: Persist to hybrid memory (STM + Mem0 LTM)
- `src/cores/bounded_memory_saver.py`: Bounded memory persistence for long sessions

Configuration (use_kb, use_memory) is passed at the `ask()` / `ask_stream()` entry point.

Public API:
- `pipeline.ask(query, config?)` → async dict with `answer`
- `pipeline.ask_stream(query, config?)` → async generator of event dicts
- `pipeline.batch_ask(queries)` → list of results

### Configuration System

All config lives in `src/configs/`. `AppConfig` (in `config.py`) is the root object that loads `.env` via Pydantic. Environment variable convention: `SECTION__FIELD` maps to `config.section.field` (e.g., `LLM__API_KEY` → `config.llm.api_key`).

Config files:
- `config.py`: Root `AppConfig` class
- `model_config.py`: LLM and embedding model config
- `retrieve_config.py`: Retrieval/search config
- `memory_settings.py`: Memory behavior settings
- `logger_config.py`: Logging configuration

Key config flags in `SearchConfig` (via `config.retrieve`):
- `use_kb` / `use_memory` — enable knowledge base and memory
- `use_sparse` / `use_reranker` — retrieval quality enhancements
- `use_rewrite` / `rewrite_mode` — **[Deprecated]** query transformation is now an Agent tool

### LLM Models

`src/models/llm.py`: `LLMWrapper` supports multiple providers via **litellm** (recommended) or direct OpenAI SDK:

| Provider | Model Format | Notes |
|----------|-------------|-------|
| `litellm` | `minimax/minimaxi` | 推荐，支持 100+ LLM，统一接口 |
| `openai` | `gpt-4o` | 标准 OpenAI 兼容 |
| `minimax` | MiniMax-M2.7 | 兼容 OpenAI API |

**litellm 优势**：统一接口、内置 tool calling、自动处理 provider 差异、重试/限流。

- `src/models/llm.py`: `LLMWrapper` + `OpenAILLM`
- `src/models/llm_litellm.py`: `LiteLLMWrapper` (litellm 封装，兼容 `LLMWrapper` 接口)
- `src/models/llm_response_cache.py`: LLM 语义缓存，基于 RedisVL 向量相似度匹配
- `src/models/embedding.py`: `TextEmbedding` with optional RedisVL caching

#### LLM Response Cache (Semantic Cache)

`src/models/llm_response_cache.py`: 基于 RedisVL 向量相似度的 LLM 响应语义缓存。

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

### Vector Database (db_services)

`src/db_services/` manages database access with a factory pattern:
- `src/db_services/base.py`: Base interface
- `src/db_services/factory.py`: Database service factory
- `src/db_services/milvus/`: Milvus implementation
  - `connection.py` / `connection_manager.py`: Connection management
  - `collection.py` / `collection_manager.py`: Collection management
  - `data.py` / `data_service.py`: Data operations
  - `retrieval.py`: Retrieval operations
  - `store.py` / `database_manager.py`: Storage management

Supports two modes via `MilvusConfig.milvus_mode`:
- `"local"` — SQLite-based milvus-lite (default, no server needed)
- `"remote"` — connects to a Milvus server

Retrieval supports dense (semantic), sparse (BM25), and hybrid search with optional BGE reranker.

### Tools & MCP

External tools (weather, web search, kb_search, query_rewrite) are invoked via the MCP protocol:
- `src/mcp/mcp_client.py`: MCP client for tool calls
- `src/mcp/server/`: MCP server implementations (`kb_tools.py`, `weather.py`, `web_crawl.py`)
- `src/agent/tools.py`: `WebSpider` for web crawling

Key tools:
- `kb_search(query, top_k, use_hyde)`: Knowledge base retrieval, supports HyDE enhancement via `use_hyde` parameter
- `query_rewrite(query, mode)`: Query transformation tool, supports `rewrite` / `step_back` / `sub_query` modes
- `weather_get_alerts` / `weather_get_forecast`: Weather information
- `web_crawl(url)`: Web page content extraction
- `read_skill(name)`: Skill instruction reader

HyDE (Hypothetical Document Embeddings) is a retrieval enhancement technique — when `use_hyde=True`, the tool generates hypothetical answer embeddings for better semantic search. Agent decides when to use it based on query complexity.

### Memory (Hybrid)

Two layers:
- **Short-term (STM)**: LangGraph message history within a session (via `QAState.messages`)
- **Long-term (LTM)**: Mem0 integration (`src/memory/mem0_manager.py`) + Milvus vector storage

Core services:
- `src/memory/hybrid_memory_service.py`: Unified memory interface (primary)
- `src/memory/base.py`: Abstract base class
- `src/memory/short_term_memory.py`: STM implementation
- `src/memory/long_term_memory.py`: LTM implementation
- `src/memory/mem0_manager.py`: Mem0 integration
- `src/cores/bounded_memory_saver.py`: Bounded memory persistence

### Skills System

`src/skills/skill_manager.py` and `skills/` directory manage extensible skills:
- Skills are injected into system prompts as additional context
- Directory: `skills/` contains skill definition files

### Web App

- `chainlit_app.py`: Chainlit 5 app (primary, recommended)
- `web_app.py`: Gradio 5 app (legacy)

Security: `src/utils/security.py` handles input validation (XSS, SQL injection, path traversal).

## Key Conventions

- **Async**: Embedding, tool calls, and pipeline are async; exposes both sync and async streaming interfaces
- **Observability**: Critical functions use `@langfuse_context.observe()` decorators
- **Config injection**: Pipeline components receive `AppConfig` at init time — don't read env vars directly
- **PYTHONPATH**: Must be set to `.` when running scripts directly (imports use `src.` prefix)
- **Proactive documentation**: After implementing features, immediately update CLAUDE.md and relevant docs — do not wait to be asked

## Environment Setup

Copy `.env.example` to `.env` and configure:

```env
# LLM Provider (litellm 推荐，支持 100+ 模型)
LLM__PROVIDER=litellm
LLM__MODEL=minimax/minimaxi
LLM__BASE_URL=https://api.minimax.chat/v1
LLM__API_KEY=your_api_key

# Embedding
EMBEDDING__BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING__API_KEY=your_embedding_key
```

Optional: Langfuse (observability), Redis (embedding cache), remote Milvus (production vector DB).

## Docker

```bash
docker-compose up -d                                           # basic
docker-compose --profile with-redis up -d                     # + Redis cache
docker-compose --profile with-redis --profile with-milvus up -d  # + production Milvus
```

Chainlit app available at `http://localhost:7860`.
