# -*- coding: utf-8 -*-
"""
@Time：2025/10/11 11:21
@Auth：吕鑫
@File：langfuse_dataset.py
@IDE：PyCharm
@Desc: Langfuse 数据集导出工具
"""

from langfuse import get_client
from src.configs.logger_config import setup_logger

logger = setup_logger(__name__)

langfuse = get_client()

# 分页获取所有数据
all_traces = []
cursor = None

logger.info("开始从 Langfuse 获取 traces...")

while True:
    result = langfuse.api.trace.list(
        limit=100,
        cursor=cursor,  # 用于分页
        tags=["production"],
    )

    all_traces.extend(result.data)
    logger.info(f"已获取 {len(all_traces)} 条 traces")

    # 检查是否有下一页
    if not result.meta.has_next_page:
        break

    cursor = result.meta.next_cursor

logger.info(f"获取完成，总计 {len(all_traces)} 条 traces")
