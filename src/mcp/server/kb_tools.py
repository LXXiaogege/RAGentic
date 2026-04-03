# -*- coding: utf-8 -*-
"""
Knowledge Base search tool for MCP server.
RAG as a tool - Agent can call this when needed.
"""

from typing import Any, List, Optional

from src.configs.logger_config import setup_logger
from src.configs.retrieve_config import SearchConfig

logger = setup_logger(__name__)


class KBTools:
    def __init__(
        self,
        milvus_connection,
        embeddings,
        search_config: Optional[SearchConfig] = None,
    ):
        """
        Initialize KB tools with Milvus connection.

        Args:
            milvus_connection: MilvusConnectionManager instance
            embeddings: TextEmbedding instance
            search_config: Search configuration
        """
        self.milvus = milvus_connection
        self.embeddings = embeddings
        self.search_config = search_config or SearchConfig()

    async def kb_search(
        self,
        query: str,
        top_k: int = 3,
        hyde_vector: Optional[List[float]] = None,
    ) -> str:
        """
        Search knowledge base for relevant documents.
        Returns raw text chunks for the agent to decide how to use.

        Args:
            query: Search query (used as text query fallback when hyde_vector is provided)
            top_k: Number of documents to retrieve (default 3)
            hyde_vector: Optional HyDE vector to use for semantic search

        Returns:
            Formatted string with retrieved document chunks
        """
        try:
            logger.info(f"KB search: query={query}, top_k={top_k}, hyde_vector={hyde_vector is not None}")

            config = SearchConfig(
                top_k=top_k,
                use_kb=True,
                use_sparse=self.search_config.use_sparse,
                use_reranker=self.search_config.use_reranker,
            )

            search_query: Any = hyde_vector if hyde_vector else query
            results = await self.milvus.asearch(search_query, config)

            if not results:
                logger.info("KB search returned no results")
                return "未在知识库中找到相关内容。"

            chunks = []
            for i, doc in enumerate(results, 1):
                text = doc.get("text", "")
                score = doc.get("score")
                score_str = f"{score:.4f}" if score else "N/A"
                chunk_text = f"[文档 {i}] (相关度: {score_str})\n{text}"
                chunks.append(chunk_text)

            formatted = "\n---\n".join(chunks)
            logger.info(f"KB search returned {len(results)} chunks")
            return formatted

        except Exception as e:
            logger.exception(f"KB search error: {e}")
            return f"知识库搜索失败: {str(e)}"


_kb_tools_instance: Optional[KBTools] = None


def init_kb_tools(milvus_connection, embeddings, search_config: SearchConfig):
    """Initialize global KB tools instance."""
    global _kb_tools_instance
    _kb_tools_instance = KBTools(milvus_connection, embeddings, search_config)
    logger.info("KB tools initialized")


def get_kb_tools() -> Optional[KBTools]:
    """Get global KB tools instance."""
    return _kb_tools_instance
