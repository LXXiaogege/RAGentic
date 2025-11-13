# -*- coding: utf-8 -*-
"""
@Time ： 2025/10/11 11:21
@Auth ： 吕鑫
@File ：langfuse_dataset.py
@IDE ：PyCharm
"""
from langfuse import get_client

langfuse = get_client()

# 方法3: 分页获取所有数据
all_traces = []
cursor = None

while True:
    result = langfuse.api.trace.list(
        limit=100,
        cursor=cursor,  # 用于分页
        tags=["production"]
    )

    all_traces.extend(result.data)

    # 检查是否有下一页
    if not result.meta.has_next_page:
        break

    cursor = result.meta.next_cursor

print(f"Total traces: {len(all_traces)}")