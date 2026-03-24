# -*- coding: utf-8 -*-
"""
@Time：2025/10/11 11:21
@Auth：吕鑫
@File：langfuse_dataset.py
@IDE：PyCharm
@Desc: Langfuse 数据集导出工具
"""

from src.configs.logger_config import setup_logger

logger = setup_logger(__name__)

try:
    from langfuse import get_client
except ImportError:
    from langfuse.client import get_client  # type: ignore


def fetch_all_traces(tags: list = None, max_pages: int = 10000) -> list:
    """从 Langfuse 分页拉取所有 traces"""
    langfuse = get_client()
    all_traces = []
    cursor = None
    tags = tags or ["production"]
    page_count = 0

    logger.info("开始从 Langfuse 获取 traces...")

    while page_count < max_pages:
        result = langfuse.api.trace.list(
            limit=100,
            cursor=cursor,
            tags=tags,
        )

        all_traces.extend(result.data)
        page_count += 1
        logger.info(f"已获取 {len(all_traces)} 条 traces")

        if not result.meta.has_next_page:
            break

        cursor = result.meta.next_cursor

    if page_count >= max_pages:
        logger.warning(f"已达到最大分页数限制（{max_pages}），可能未获取全部数据")

    logger.info(f"获取完成，总计 {len(all_traces)} 条 traces")
    return all_traces


if __name__ == "__main__":
    traces = fetch_all_traces()
