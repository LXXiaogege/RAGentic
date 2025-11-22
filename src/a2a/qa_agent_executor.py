# -*- coding: utf-8 -*-
"""
QA Agent A2A Service Implementation
基于A2A框架的QA服务
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import traceback

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.utils import new_agent_text_message, new_task
from a2a.types import (
    Part,
    Task,
    TaskState,
    TextPart,
)
from src.cores.qa_agent import QA_Agent, QAState
from src.configs.config import QAPipelineConfig
from src.configs.logger_config import setup_logger

logger = setup_logger(__name__)


@dataclass
class QARequest:
    """QA请求数据结构"""
    query: str
    thread_id: Optional[str] = None
    use_knowledge_base: bool = True
    use_tools: bool = False
    use_memory: bool = True
    k: int = 5
    use_sparse: bool = False
    use_reranker: bool = False
    stream: bool = False
    messages: Optional[List[Dict]] = None


@dataclass
class QAResponse:
    """QA响应数据结构"""
    success: bool
    question: str
    answer: Optional[str] = None
    error: Optional[str] = None
    transformed_query: Optional[str] = None
    context: Optional[str] = None
    kb_context: Optional[str] = None
    tool_context: Optional[str] = None
    messages: Optional[List[Dict]] = None
    processing_time: Optional[float] = None
    timestamp: Optional[str] = None
    stream: bool = False


class QAAgentExecutor(AgentExecutor):
    """QA Agent执行器"""

    def __init__(self):
        self.config = QAPipelineConfig()
        self.qa_agent = QA_Agent(self.config)
        logger.info("QA Agent Executor初始化完成")

    def _parse_request_data(self, context: RequestContext) -> QARequest:
        """解析请求数据"""
        return QARequest(
            query=context.message.parts[1].root.data.get("query", ""),
            thread_id=context.message.parts[1].root.data.get("thread_id"),
            use_knowledge_base=context.message.parts[1].root.data.get("use_knowledge_base", True),
            use_tools=context.message.parts[1].root.data.get("use_tools", False),
            use_memory=context.message.parts[1].root.data.get("use_memory", True),
            k=context.message.parts[1].root.data.get("k", 5),
            use_sparse=context.message.parts[1].root.data.get("use_sparse", False),
            use_reranker=context.message.parts[1].root.data.get("use_reranker", False),
            stream=context.message.parts[1].root.data.get("stream", False),
            messages=context.message.parts[1].root.data.get("messages", [])
        )

    def _build_qa_kwargs(self, request_data: QARequest) -> Dict[str, Any]:
        """构建QA Agent调用参数"""
        return {
            "thread_id": request_data.thread_id or "default",
            "use_knowledge_base": request_data.use_knowledge_base,
            "use_tools": request_data.use_tools,
            "use_memory": request_data.use_memory,
            "k": request_data.k,
            "use_sparse": request_data.use_sparse,
            "use_reranker": request_data.use_reranker,
            "messages": request_data.messages or []
        }

    async def _ensure_task_exists(self, context: RequestContext, event_queue: EventQueue) -> Task:
        """确保task存在，如果不存在则创建"""
        task = context.current_task
        if not task:
            task = new_task(context.message)  # type: ignore
            await event_queue.enqueue_event(task)
        return task

    async def _handle_error(self, error: Exception, error_type: str, event_queue: EventQueue):
        """统一错误处理"""
        logger.error(f"{error_type}失败: {error}")
        logger.error(traceback.format_exc())
        await event_queue.enqueue_event(
            new_agent_text_message(f"❌ {error_type}失败: {str(error)}")
        )

    def _create_qa_response(self, request_data: QARequest, result: Dict, processing_time: float) -> QAResponse:
        """创建QA响应对象"""
        if result.get("error"):
            return QAResponse(
                success=False,
                question=request_data.query,
                error=result["error"],
                processing_time=processing_time,
                timestamp=datetime.now().isoformat()
            )
        else:
            return QAResponse(
                success=True,
                question=result.get("question", request_data.query),
                answer=result.get("answer"),
                transformed_query=result.get("transformed_query"),
                context=result.get("context"),
                kb_context=result.get("kb_context"),
                tool_context=result.get("tool_context"),
                messages=result.get("messages"),
                processing_time=processing_time,
                timestamp=datetime.now().isoformat()
            )

    async def execute(
            self,
            context: RequestContext,
            event_queue: EventQueue,
    ) -> None:
        """执行QA请求"""
        try:
            # 解析请求
            request_data = self._parse_request_data(context)

            # 确保task存在
            task = await self._ensure_task_exists(context, event_queue)
            updater = TaskUpdater(event_queue, task.id, task.contextId)

            # 根据请求类型处理
            if request_data.stream:
                await self._handle_stream_request(request_data, event_queue, task, updater)
            else:
                await self._handle_sync_request(request_data, event_queue, task, updater)

        except Exception as e:
            await self._handle_error(e, "执行QA请求", event_queue)

    async def _handle_sync_request(self, request_data: QARequest, event_queue: EventQueue, task: Task,
                                   updater: TaskUpdater):
        """处理同步请求"""
        try:
            # 更新任务状态为工作中
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"正在处理查询: {request_data.query}",
                    task.contextId,
                    task.id,
                ),
            )

            kwargs = self._build_qa_kwargs(request_data)
            start_time = datetime.now()
            result = await self.qa_agent.ask(request_data.query, **kwargs)
            processing_time = (datetime.now() - start_time).total_seconds()

            # 创建响应
            response = self._create_qa_response(request_data, result, processing_time)

            if response.success:
                # 添加结果作为artifact
                await updater.add_artifact(
                    [Part(root=TextPart(text=response.answer))],
                    name='qa_result',
                )
                # 完成任务
                await updater.complete()
            else:
                # 处理错误情况
                await updater.update_status(
                    TaskState.failed,
                    new_agent_text_message(
                        f"查询处理失败: {response.error}",
                        task.contextId,
                        task.id,
                    ),
                    final=True,
                )

        except Exception as e:
            await self._handle_error(e, "同步请求处理", event_queue)

    async def _handle_stream_request(self, request_data: QARequest, event_queue: EventQueue, task: Task,
                                     updater: TaskUpdater):
        """处理流式请求"""
        try:
            query = request_data.query
            async for item in self.qa_agent.ask_stream(query, context_id=task.contextId):
                is_task_complete = item['is_task_complete']
                require_user_input = item['require_user_input']

                if not is_task_complete and not require_user_input:
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(
                            item['content'],
                            task.contextId,
                            task.id,
                        ),
                    )
                elif require_user_input:
                    await updater.update_status(
                        TaskState.input_required,
                        new_agent_text_message(
                            item['content'],
                            task.contextId,
                            task.id,
                        ),
                        final=True,
                    )
                    break
                else:
                    await updater.add_artifact(
                        [Part(root=TextPart(text=item['content']))],
                        name='conversion_result',
                    )
                    await updater.complete()
                    break

        except Exception as e:
            await self._handle_error(e, "流式请求处理", event_queue)

    async def cancel(
            self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """取消请求"""
        logger.info("收到取消请求")
        await event_queue.enqueue_event(
            new_agent_text_message(" 请求已取消")
        )
