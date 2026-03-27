# AGENTS.md - Development Guidelines for RAGentic

## 📦 Build & Environment

### Package Manager
- Use `uv` for dependency management (Python 3.10+)
- Install dependencies: `uv sync`
- Add package: `uv add <package>`
- Remove package: `uv remove <package>`

## 🏗️ Project Structure

```
RAGentic/
├── src/
│   ├── a2a/              # A2A protocol integration
│   ├── agent/           # Agent logic & tools
│   ├── configs/         # Configuration classes
│   ├── cores/           # Core pipeline logic (LangGraph)
│   ├── data_process/    # Data processing
│   ├── db_services/     # Database operations (Milvus)
│   ├── evaluate/        # RAG evaluation
│   ├── memory/          # Conversation memory (Mem0)
│   ├── mcp/             # MCP tool integration
│   ├── models/          # LLM/Embedding wrappers
│   ├── skills/          # Agent skills
│   └── utils/           # Utilities (security, etc.)
├── tests/               # Test files
├── data/                # Data files (gitignored)
├── logs/                # Log files (gitignored)
├── .env                 # Environment variables (gitignored)
├── .env.example         # Environment template
├── pyproject.toml       # Project dependencies
└── skills/              # Skill definitions
```

## 🔧 Pipeline Architecture (LangGraph) - Light RAG, Heavy Agent

### Workflow Graph
```
parse_query → check_call_tools
                   ↓ (use_tool=True)
              agent_node ↔ tools_node (含 kb_search 工具)
                            ↓ (use_tool=False 或 agent完成)
                        build_context → generate_answer → update_memory
```

### 核心变化
- **RAG 工具化**: 知识库检索成为 `kb_search` MCP 工具，Agent 按需主动调用
- **移除 transform_query**: Agent 自己处理复杂查询分解
- **Agent 主导**: 主流程围绕 `agent_node` 展开，工具调用由 LLM 决定

### Public API (`src/cores/pipeline_langgraph.py`)
- `pipeline.ask(query, config?)` → sync dict with `answer`
- `pipeline.ask_stream(query, config?)` → async generator of event dicts
- `pipeline.batch_ask(queries)` → list of results

### 可用工具 (MCP Tools)
- `kb_search(query, top_k)` - 搜索知识库
- `weather_get_alerts(state)` - 天气预警
- `weather_get_forecast(lat, lon)` - 天气预报
- `web_crawl(url)` - 网页爬取
- `read_skill(name)` - 读取技能指令

### State Management
State flows as `QAState` (TypedDict) through all nodes. Conditional edge `check_call_tools` decides whether to enter agent loop based on `config.retrieve.use_tool`.

## 🧠 Memory Architecture (分层记忆)

### 架构概览
```
HybridMemoryService (统一接口)
├── ShortTermMemory (短期记忆 - 滑动窗口)
│   └── LangGraph Checkpoint (跨请求状态保持)
└── LongTermMemory (长期记忆 - Mem0)
    └── Milvus 向量存储 + SQLite 历史
```

### 核心组件 (`src/memory/`)
| 文件 | 说明 |
|------|------|
| `base.py` | 抽象基类 `MemoryService`, `ShortTermMemory`, `LongTermMemory` |
| `short_term_memory.py` | 短期记忆实现（deque 滑动窗口 + Checkpoint） |
| `long_term_memory.py` | 长期记忆封装（封装 Mem0Manager） |
| `hybrid_memory_service.py` | 混合记忆服务，统一管理 STM + LTM |
| `mem0_manager.py` | Mem0 原生接口封装 |
| `memory_adapter.py` | AppConfig → Mem0 配置适配器 |

### 配置 (`src/configs/memory_settings.py`)
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `service_type` | `hybrid` | 服务类型: hybrid / stm_only / ltm_only |
| `stm_window_size` | `10` | 短期记忆窗口大小 |
| `ltm_persist_threshold` | `3` | 刷写到 LTM 的阈值（轮数） |
| `enable_stm` | `True` | 是否启用短期记忆 |
| `enable_ltm` | `True` | 是否启用长期记忆 |

### Pipeline 集成
- **`_build_context`**: 搜索 LTM 相关记忆加入上下文
- **`_update_memory`**: 消息先存 STM，积累阈值后刷 LTM

### 使用方式
```python
# Pipeline 自动集成，无需手动调用
result = pipeline.ask("你好", use_memory=True)

# 直接使用 HybridMemoryService
from src.memory.hybrid_memory_service import HybridMemoryService
memory = HybridMemoryService(config, user_id="user123")
await memory.initialize()
await memory.add(messages=[...], user_id="user123")
results = await memory.search(query="用户信息", user_id="user123")
```

### 关键特性
- **分层记忆**: STM 快速访问 + LTM 持久化
- **智能刷写**: `ltm_persist_threshold` 控制何时刷 LTM
- **懒初始化**: 首次使用时初始化记忆服务
- **向后兼容**: `use_memory=True` 保持原有行为

## 🧪 Testing

### Run Tests
```bash
# Run all tests
PYTHONPATH=. uv run pytest tests/

# Run single test file
PYTHONPATH=. uv run pytest tests/test_pipeline_langgraph.py

# Run with verbose output
PYTHONPATH=. uv run pytest -v tests/test_embedding.py

# Run specific test function
PYTHONPATH=. uv run pytest -v tests/test_memory.py::test_memory_init
```

### Test Files
- `tests/test_pipeline_langgraph.py` - LangGraph QA pipeline
- `tests/test_embedding.py` - Embedding model tests
- `tests/test_memory.py` - Memory system tests
- `tests/a2a_client_test.py` - A2A client tests
- `tests/test_rag_evaluate.py` - RAG evaluation tests

## 🔧 Linting & Formatting

### Ruff (Recommended)
```bash
# Install ruff if not present
uv add --dev ruff

# Lint code
uv run ruff check src/ tests/

# Auto-fix issues
uv run ruff check --fix src/ tests/

# Format code
uv run ruff format src/ tests/
```

### Type Checking (Optional)
```bash
# Install mypy
uv add --dev mypy

# Run type check
uv run mypy src/
```

## 📝 Code Style Guidelines

### Imports
- Use absolute imports from project root: `from src.configs.config import AppConfig`
- Group imports: stdlib → third-party → local
- Use `from __future__ import annotations` for forward references

### Formatting
- **Line length**: Max 88 characters (Ruff default)
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Double quotes for strings
- **Trailing commas**: Use in multi-line structures

### Type Hints
- Use Python 3.10+ union syntax: `str | None` instead of `Optional[str]`
- Annotate all function parameters and return types
- Use `TypedDict` for structured dictionaries
- Example:
```python
def ask(self, query: str, **kwargs) -> Dict[str, Any]:
    ...
```

### Naming Conventions
- **Classes**: PascalCase (`LangGraphQAPipeline`, `AppConfig`)
- **Functions/Methods**: snake_case (`build_context`, `kb_search`)
- **Constants**: UPPER_SNAKE_CASE (`BASE_DIR`, `MAX_RETRIES`)
- **Private members**: Leading underscore (`_init_components`)
- **Files**: snake_case (`pipeline_langgraph.py`)

### Error Handling
- Use specific exception types, not bare `except Exception`
- Log errors with `logger.exception()` to include stack trace
- Re-raise with context using `raise ... from e`
- Return error state in pipeline nodes:
```python
try:
    result = process()
    return state
except Exception as e:
    logger.error(f"Process failed: {e}")
    state.error = str(e)
    return state
```

### Async/Await
- Use `async def` for I/O-bound operations
- Use `await` consistently in async functions
- For mixed sync/async code, use `asyncio.run()` or `ThreadPoolExecutor`
- Never block async event loops with synchronous I/O

### LangGraph Patterns
- Define state with `Annotated` for reducers: `messages: Annotated[List, add_messages]`
- Use conditional edges for branching logic
- Keep node functions pure (return new state, don't mutate)
- Use `RunnableConfig` for thread management

### Logging
- Use `setup_logger(__name__)` from `src.configs.logger_config`
- Log levels: `debug` (verbose), `info` (normal), `warning` (unexpected), `error` (failure)
- Include context in log messages: `f"Processing query: {query}"`

### Configuration
- Use Pydantic `BaseSettings` for configuration classes
- Support environment variables with `__` separator: `LLM__API_KEY`
- Resolve relative paths to absolute in `model_post_init`
- Default values should be safe/fallback values

### Documentation
- Use docstrings for public functions (Google or NumPy style)
- Include type hints in docstrings for complex types
- Comment "why", not "what" (code should be self-explanatory)
- Use Chinese comments (project convention)

## 🚀 Common Tasks

### Run Web App
```bash
PYTHONPATH=. uv run python chainlit_app.py
# Access at http://127.0.0.1:7860
```

### Run MCP Server
```bash
PYTHONPATH=. uv run python mcp_server.py
```

### Docker
```bash
docker-compose up -d                                           # basic
docker-compose --profile with-redis up -d                     # + Redis cache
docker-compose --profile with-redis --profile with-milvus up -d  # + production Milvus
```

### Export Pipeline Graph
```python
from src.configs.config import AppConfig
from src.cores.pipeline_langgraph import LangGraphQAPipeline

config = AppConfig()
pipeline = LangGraphQAPipeline(config)
pipeline.export_graph("workflow.png")
```

## 📋 Pre-commit Checklist
- [ ] Code formatted with `ruff format`
- [ ] No linting errors: `ruff check`
- [ ] Type hints added for new functions
- [ ] Tests pass for modified components
- [ ] No secrets/credentials in code
- [ ] Log messages are informative

## ⚠️ Important Notes
- Never commit `.env` file (contains secrets)
- Never commit model files (*.pkl, *.pth, etc.)
- Never commit database files (*.db, *.sqlite)
- Always use `PYTHONPATH=.` when running scripts
- LangGraph state is immutable - always return new state objects
