# AGENTS.md - Development Guidelines for RAGentic

## 📦 Build & Environment

### Package Manager
- Use `uv` for dependency management (Python 3.10+)
- Install dependencies: `uv sync`
- Add package: `uv add <package>`
- Remove package: `uv remove <package>`

### Environment Setup
```bash
# Create virtual environment (if needed)
uv venv

# Sync dependencies
uv sync

# Set PYTHONPATH for imports
export PYTHONPATH=.
```

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
- **Functions/Methods**: snake_case (`transform_query`, `build_context`)
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

## 🏗️ Project Structure

```
RAGentic/
├── src/
│   ├── configs/          # Configuration classes
│   ├── cores/            # Core pipeline logic
│   ├── models/           # LLM/Embedding wrappers
│   ├── db_services/      # Database operations
│   ├── memory/           # Conversation memory
│   ├── mcp/              # MCP tool integration
│   ├── agent/            # Agent logic
│   ├── data_process/     # Data processing
│   └── evaluate/         # RAG evaluation
├── tests/                # Test files
├── data/                 # Data files (gitignored)
├── logs/                 # Log files (gitignored)
├── .env                  # Environment variables (gitignored)
├── .env.example          # Environment template
└── pyproject.toml        # Project dependencies
```

## 🚀 Common Tasks

### Run Web App
```bash
PYTHONPATH=. uv run python web_app.py
# Access at http://127.0.0.1:7860
```

### Run MCP Server
```bash
PYTHONPATH=. uv run python mcp_server.py
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
