# -*- coding: utf-8 -*-
"""
RAGentic API Server
FastAPI backend exposing LangGraphQAPipeline via REST/SSE endpoints
"""

import uuid
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.configs.config import AppConfig
from src.cores.pipeline_langgraph import LangGraphQAPipeline
from src.configs.logger_config import setup_logger

logger = setup_logger(__name__)

app = FastAPI(title="RAGentic API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pipeline instance cache per session/config tuple
PIPELINE_CACHE_MAX_SIZE = 32
PIPELINE_CACHE_TTL_SECONDS = 60 * 60


@dataclass
class PipelineCacheEntry:
    pipeline: LangGraphQAPipeline
    last_accessed: float


PipelineCacheKey = tuple[str, bool, int, bool, bool, bool]
_pipeline_cache: OrderedDict[PipelineCacheKey, PipelineCacheEntry] = OrderedDict()


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    session_id: Optional[str] = None
    use_memory: bool = False
    top_k: int = Field(default=5, ge=1, le=20)
    use_sparse: bool = False
    use_reranker: bool = False
    enable_think: bool = False


class AskResponse(BaseModel):
    question: str
    answer: str
    context: Optional[str] = None
    kb_context: Optional[str] = None
    tool_context: Optional[str] = None
    error: Optional[str] = None


async def get_pipeline(
    session_id: str,
    use_memory: bool,
    top_k: int,
    use_sparse: bool,
    use_reranker: bool,
    enable_think: bool,
) -> LangGraphQAPipeline:
    """Get or create a pipeline instance for the session."""
    cache_key = (
        session_id,
        use_memory,
        top_k,
        use_sparse,
        use_reranker,
        enable_think,
    )
    await _evict_expired_pipelines()
    if cache_key in _pipeline_cache:
        entry = _pipeline_cache.pop(cache_key)
        entry.last_accessed = time.monotonic()
        _pipeline_cache[cache_key] = entry
        logger.info(f"复用会话 {session_id} 的管道实例")
        return entry.pipeline

    config = AppConfig()
    req_config = config.create_request_config(
        use_kb=True,
        use_tool=True,
        use_memory=use_memory,
        top_k=top_k,
        use_sparse=use_sparse,
        use_reranker=use_reranker,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_think}},
    )

    pipeline = LangGraphQAPipeline(req_config)
    _pipeline_cache[cache_key] = PipelineCacheEntry(
        pipeline=pipeline,
        last_accessed=time.monotonic(),
    )
    await _evict_overflow_pipelines()
    logger.info(f"为会话 {session_id} 创建新的管道实例，缓存大小: {len(_pipeline_cache)}")
    return pipeline


async def _cleanup_pipeline(pipeline: LangGraphQAPipeline) -> None:
    try:
        await pipeline.cleanup()
    except Exception as e:
        logger.warning(f"清理管道资源失败: {e}")


async def _evict_expired_pipelines() -> None:
    now = time.monotonic()
    expired_keys = [
        key
        for key, entry in _pipeline_cache.items()
        if now - entry.last_accessed > PIPELINE_CACHE_TTL_SECONDS
    ]
    for key in expired_keys:
        entry = _pipeline_cache.pop(key)
        await _cleanup_pipeline(entry.pipeline)


async def _evict_overflow_pipelines() -> None:
    while len(_pipeline_cache) > PIPELINE_CACHE_MAX_SIZE:
        _, entry = _pipeline_cache.popitem(last=False)
        await _cleanup_pipeline(entry.pipeline)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "RAGentic API"}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """Non-streaming Q&A endpoint."""
    session_id = request.session_id or str(uuid.uuid4())

    pipeline = await get_pipeline(
        session_id=session_id,
        use_memory=request.use_memory,
        top_k=request.top_k,
        use_sparse=request.use_sparse,
        use_reranker=request.use_reranker,
        enable_think=request.enable_think,
    )

    try:
        result = await pipeline.ask(
            query=request.query,
            thread_id=session_id,
            langfuse_session_id=session_id,
        )

        if result.get("error"):
            return AskResponse(
                question=request.query,
                answer="",
                error=result["error"],
            )

        return AskResponse(
            question=request.query,
            answer=result.get("answer", ""),
            context=result.get("context", ""),
            kb_context=result.get("kb_context", ""),
            tool_context=result.get("tool_context", ""),
        )
    except Exception as e:
        logger.exception("同步问答处理异常")
        return AskResponse(
            question=request.query,
            answer="",
            error=f"处理请求时出错: {str(e)}",
        )


@app.post("/ask/stream")
async def ask_stream(request: AskRequest):
    """Streaming Q&A endpoint via Server-Sent Events."""
    session_id = request.session_id or str(uuid.uuid4())

    pipeline = await get_pipeline(
        session_id=session_id,
        use_memory=request.use_memory,
        top_k=request.top_k,
        use_sparse=request.use_sparse,
        use_reranker=request.use_reranker,
        enable_think=request.enable_think,
    )

    async def event_generator():
        try:
            async for event in pipeline.ask_stream(
                query=request.query,
                thread_id=session_id,
                langfuse_session_id=session_id,
            ):
                status = event.get("status")
                node = event.get("node", "")

                if status == "chunk":
                    chunk_content = event.get('content', '').replace('\n', '\\n')
                    yield f"event: chunk\ndata: {chunk_content}\n\n"

                elif status == "processing" and node:
                    step_data = {
                        "node": node,
                        "status": "processing",
                        "state": event.get("state", {}),
                    }
                    yield f"event: step\ndata: {json.dumps(step_data, ensure_ascii=False)}\n\n"

                elif status == "complete":
                    answer = event.get("answer", "")
                    context = event.get("context", "")
                    complete_data = {"answer": answer, "context": context}
                    yield f"event: complete\ndata: {json.dumps(complete_data, ensure_ascii=False)}\n\n"

                elif status == "error":
                    error_msg = event.get("error", "未知错误")
                    yield f"event: error\ndata: {json.dumps(error_msg, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.exception("流式问答处理异常")
            yield f"event: error\ndata: {json.dumps(str(e), ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a session (remove pipeline from cache)."""
    matching_keys = [key for key in _pipeline_cache if key[0] == session_id]
    if matching_keys:
        for key in matching_keys:
            entry = _pipeline_cache.pop(key)
            await _cleanup_pipeline(entry.pipeline)
        return {"status": "ok", "message": f"Session {session_id} cleared"}
    return {"status": "ok", "message": "Session not found"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
