# -*- coding: utf-8 -*-
"""
@Time ： 2025/11/24 14:38
@Auth ： 吕鑫
@File ：test_embedding.py
@IDE ：PyCharm
"""

import asyncio
from src.models.embedding import TextEmbedding
from src.configs.config import AppConfig

config = AppConfig()
embedding = TextEmbedding(config.embedding)

print(embedding.get_embedding(["hello world", "hello world"]))


async def main():
    result = await embedding.aget_embedding(["hello world", "hello world"])
    print(result)


# 运行异步函数
if __name__ == "__main__":
    asyncio.run(main())
