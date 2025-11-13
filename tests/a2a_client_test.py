# -*- coding: utf-8 -*-
"""
QA Service A2A Client
基于A2A框架的QA服务客户端
"""

import asyncio
from typing import Any, Dict, List, Optional, AsyncGenerator
from uuid import uuid4
from dataclasses import dataclass, asdict

import httpx

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
)

from src.config.logger_config import setup_logger

logger = setup_logger(__name__)


@dataclass
class QAClientRequest:
    """QA客户端请求参数"""
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


class QAServiceClient:
    """QA服务A2A客户端"""

    def __init__(self, base_url: str = 'http://localhost:9999'):
        self.base_url = base_url
        self.httpx_client: Optional[httpx.AsyncClient] = None
        self.a2a_client: Optional[A2AClient] = None
        self.agent_card: Optional[AgentCard] = None
        self._initialized = False

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()

    async def initialize(self):
        """初始化客户端连接"""
        if self._initialized:
            return

        try:
            logger.info(f"初始化QA服务客户端，连接到: {self.base_url}")

            # 创建HTTP客户端
            self.httpx_client = httpx.AsyncClient(timeout=30.0)

            # 初始化A2A卡片解析器
            resolver = A2ACardResolver(
                httpx_client=self.httpx_client,
                base_url=self.base_url,
            )

            # 获取Agent Card
            self.agent_card = await self._fetch_agent_card(resolver)

            # 初始化A2A客户端
            self.a2a_client = A2AClient(
                httpx_client=self.httpx_client,
                agent_card=self.agent_card
            )

            self._initialized = True
            logger.info("QA服务客户端初始化完成")

        except Exception as e:
            logger.error(f"QA服务客户端初始化失败: {e}")
            await self.close()
            raise

    async def _fetch_agent_card(self, resolver: A2ACardResolver) -> AgentCard:
        """获取Agent Card"""
        public_card_path = '/.well-known/agent.json'
        extended_card_path = '/agent/authenticatedExtendedCard'

        try:
            # 首先尝试获取公共卡片
            logger.info(f"获取公共Agent Card: {self.base_url}{public_card_path}")
            public_card = await resolver.get_agent_card()
            logger.info("成功获取公共Agent Card")

            final_card = public_card

            # 检查是否支持扩展卡片
            if public_card.supportsAuthenticatedExtendedCard:
                try:
                    logger.info(f"尝试获取扩展Agent Card: {self.base_url}{extended_card_path}")
                    auth_headers = {
                        'Authorization': 'Bearer your-auth-token-here'
                    }
                    extended_card = await resolver.get_agent_card(
                        relative_card_path=extended_card_path,
                        http_kwargs={'headers': auth_headers},
                    )
                    logger.info("成功获取扩展Agent Card")
                    final_card = extended_card
                except Exception as e:
                    logger.warning(f"获取扩展Agent Card失败，使用公共卡片: {e}")

            return final_card

        except Exception as e:
            logger.error(f"获取Agent Card失败: {e}")
            raise

    async def close(self):
        """关闭客户端连接"""
        if self.httpx_client:
            await self.httpx_client.aclose()
            self.httpx_client = None
        self.a2a_client = None
        self.agent_card = None
        self._initialized = False
        logger.info("QA服务客户端已关闭")

    def _prepare_message_payload(self, request: QAClientRequest) -> Dict[str, Any]:
        """准备消息载荷"""
        # 将QA请求转换为A2A消息格式
        qa_data = asdict(request)
        qa_data = {k: v for k, v in qa_data.items() if v is not None}

        return {
            'message': {
                'role': 'user',
                'parts': [
                    {
                        'kind': 'text',
                        'text': request.query
                    },
                    {
                        'kind': 'data',
                        'data': qa_data
                    }
                ],
                'messageId': uuid4().hex,
            },
        }

    async def ask(self,
                  query: str,
                  thread_id: Optional[str] = None,
                  use_knowledge_base: bool = True,
                  use_tools: bool = False,
                  use_memory: bool = True,
                  k: int = 5,
                  use_sparse: bool = False,
                  use_reranker: bool = False,
                  messages: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """发送同步QA请求"""
        if not self._initialized:
            await self.initialize()

        request = QAClientRequest(
            query=query,
            thread_id=thread_id,
            use_knowledge_base=use_knowledge_base,
            use_tools=use_tools,
            use_memory=use_memory,
            k=k,
            use_sparse=use_sparse,
            use_reranker=use_reranker,
            stream=False,
            messages=messages
        )

        try:
            logger.info(f"发送QA请求: {query}")

            payload = self._prepare_message_payload(request)
            send_request = SendMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(**payload)
            )

            response = await self.a2a_client.send_message(send_request, http_kwargs={'timeout': 60.0})
            logger.info("收到QA响应")

            return response.model_dump(mode='json', exclude_none=True)

        except Exception as e:
            logger.error(f"QA请求失败: {e}")
            raise

    async def ask_stream(self,
                         query: str,
                         thread_id: Optional[str] = None,
                         use_knowledge_base: bool = True,
                         use_tools: bool = False,
                         use_memory: bool = True,
                         k: int = 5,
                         use_sparse: bool = False,
                         use_reranker: bool = False,
                         messages: Optional[List[Dict]] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """发送流式QA请求"""
        if not self._initialized:
            await self.initialize()

        request = QAClientRequest(
            query=query,
            thread_id=thread_id,
            use_knowledge_base=use_knowledge_base,
            use_tools=use_tools,
            use_memory=use_memory,
            k=k,
            use_sparse=use_sparse,
            use_reranker=use_reranker,
            stream=True,
            messages=messages
        )

        try:
            logger.info(f"发送流式QA请求: {query}")

            payload = self._prepare_message_payload(request)
            streaming_request = SendStreamingMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(**payload)
            )

            stream_response = self.a2a_client.send_message_streaming(streaming_request)

            async for chunk in stream_response:
                yield chunk.model_dump(mode='json', exclude_none=True)

        except Exception as e:
            logger.error(f"流式QA请求失败: {e}")
            raise

    async def batch_ask(self,
                        queries: List[str],
                        use_knowledge_base: bool = True,
                        use_tools: bool = False,
                        use_memory: bool = True,
                        k: int = 5,
                        use_sparse: bool = False,
                        use_reranker: bool = False) -> List[Dict[str, Any]]:
        """批量QA请求"""
        results = []

        for i, query in enumerate(queries):
            try:
                result = await self.ask(
                    query=query,
                    thread_id=f"batch_{i}",
                    use_knowledge_base=use_knowledge_base,
                    use_tools=use_tools,
                    use_memory=use_memory,
                    k=k,
                    use_sparse=use_sparse,
                    use_reranker=use_reranker
                )
                results.append(result)

            except Exception as e:
                logger.error(f"批量请求第{i + 1}个问题失败: {e}")
                results.append({
                    "success": False,
                    "question": query,
                    "error": str(e)
                })

        return results


# 使用示例和测试函数
async def test_qa_client():
    """测试QA客户端"""
    async with QAServiceClient('http://localhost:9999') as client:

        # 测试单个问答
        # print("=== 测试同步问答 ===")
        # try:
        #     result = await client.ask(
        #         query="什么是人工智能？",
        #         use_knowledge_base=False,
        #         k=3
        #     )
        #     print("同步问答结果:")
        #     print(result)
        # except Exception as e:
        #     print(f"同步问答失败: {e}")

        # 测试流式问答
        print("\n=== 测试流式问答 ===")
        try:
            print("流式问答结果:")
            async for chunk in client.ask_stream(
                    query="请解释机器学习的基本概念",
                    use_knowledge_base=False,
                    k=3,

            ):
                print(f"流式数据: {chunk}")
        except Exception as e:
            print(f"流式问答失败: {e}")
        #
        # # 测试批量问答
        # print("\n=== 测试批量问答 ===")
        # try:
        #     questions = [
        #         "什么是深度学习？",
        #         "神经网络是如何工作的？",
        #         "什么是自然语言处理？"
        #     ]
        #     results = await client.batch_ask(questions)
        #     print("批量问答结果:")
        #     for i, result in enumerate(results):
        #         print(f"问题{i + 1}: {result}")
        # except Exception as e:
        #     print(f"批量问答失败: {e}")


async def main():
    """主函数"""
    try:
        await test_qa_client()
    except Exception as e:
        logger.error(f"测试失败: {e}")


if __name__ == '__main__':
    asyncio.run(main())
