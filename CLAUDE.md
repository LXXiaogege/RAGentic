# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAGentic is an enterprise-grade RAG (Retrieval-Augmented Generation) Q&A system built on LangGraph. It orchestrates knowledge base retrieval, external tool invocation, conversation memory, and LLM generation in a configurable workflow graph.

## Commands

**Package manager**: `uv` (not pip)

```bash
# Install dependencies
uv sync

# Run the web app
PYTHONPATH=. uv run python web_app.py

# Run all tests
PYTHONPATH=. uv run pytest tests/

# Run a single test file
PYTHONPATH=. uv run python tests/test_pipeline_langgraph.py

# Run a specific test
PYTHONPATH=. uv run pytest tests/test_pipeline_nodes.py::TestClassName::test_method_name

# Start the MCP server
PYTHONPATH=. uv run python mcp_server.py
```

## Architecture

### Pipeline (LangGraph)

The core is `src/cores/pipeline_langgraph.py` — `LangGraphQAPipeline` defines a directed graph where nodes are pipeline stages:

```
parse_query → transform_query → check_retrieve_knowledge → retrieve_knowledge
                                                         ↘
                                                          → check_call_tools → call_tools
                                                                             ↘
                                                                              → build_context → generate_answer → update_memory
```

Conditional edges (`check_retrieve_knowledge`, `check_call_tools`) skip stages based on config flags. State flows as `QAState` (TypedDict) through all nodes.

Public API:
- `pipeline.ask(query, config?)` → sync dict with `answer`
- `pipeline.ask_stream(query, config?)` → async generator of event dicts
- `pipeline.batch_ask(queries)` → list of results

### Configuration System

All config lives in `src/configs/`. `AppConfig` (in `config.py`) is the root object that loads `.env` via Pydantic. Environment variable convention: `SECTION__FIELD` maps to `config.section.field` (e.g., `LLM__API_KEY` → `config.llm.api_key`).

Key config flags in `SearchConfig` (via `config.retrieve`):
- `use_kb` / `use_tool` / `use_memory` — enable retrieval stages
- `use_sparse` / `use_reranker` — retrieval quality enhancements
- `use_rewrite` / `rewrite_mode` — query transformation (`hyde`, `step_back`, `sub_query`)

### Models

- `src/models/llm.py`: `OpenAILLM` (raw API) and `LLMWrapper` (LangChain-compatible). Both wrap OpenAI-compatible endpoints and apply Langfuse tracing decorators.
- `src/models/embedding.py`: `TextEmbedding` with optional RedisVL caching. Batch-aware with cache hit/miss detection.

### Vector Database

`src/db_services/milvus/` manages Milvus. Supports two modes via `MilvusConfig.milvus_mode`:
- `"local"` — SQLite-based milvus-lite (default, no server needed)
- `"remote"` — connects to a Milvus server

Retrieval supports dense (semantic), sparse (BM25), and hybrid search with optional BGE reranker.

### Tools & MCP

`src/agent/tools.py` contains `WebSpider` for web crawling. External tools (weather, web search) are invoked via the MCP protocol — `src/mcp/mcp_client.py` handles tool calls; `src/mcp/server/` has server implementations.

### Memory

Two layers:
- **Short-term**: LangGraph message history within a session (via `QAState.messages`)
- **Long-term**: Mem0 integration (`src/memory/mem0_manager.py`) for cross-session memory

### Web App

`web_app.py` is a Gradio 5 app. It wraps `LangGraphQAPipeline`, applies `src/utils/security.py` input validation (XSS, SQL injection, path traversal), and manages per-session UUIDs for Langfuse tracing.

## Key Conventions

- **Async**: Embedding and tool calls are async; the pipeline exposes both sync (`ask`) and async streaming (`ask_stream`) interfaces.
- **Observability**: Critical functions use `@langfuse_context.observe()` decorators. Do not remove these when modifying pipeline nodes.
- **Config injection**: Pipeline components receive `AppConfig` at init time — don't read env vars directly in pipeline/model code.
- **PYTHONPATH**: Must be set to `.` when running scripts directly (imports use `src.` prefix).

## Environment Setup

Copy `.env.example` to `.env` and fill in at minimum:
- `LLM__API_KEY` and `LLM__BASE_URL` (OpenAI-compatible LLM)
- `EMBEDDING__API_KEY` and `EMBEDDING__BASE_URL`

Optional services: Langfuse (observability), Redis (embedding cache), remote Milvus (production vector DB).

## Docker

```bash
docker-compose up -d                                           # basic
docker-compose --profile with-redis up -d                     # + Redis cache
docker-compose --profile with-redis --profile with-milvus up -d  # + production Milvus
```

App is available at `http://localhost:7860`.
