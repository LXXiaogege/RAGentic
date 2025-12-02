# -*- coding: utf-8 -*-
"""
@Time ： 2025/11/29 13:19
@Auth ： 吕鑫
@File ：test_memory.py
@IDE ：PyCharm
"""
# -*- coding: utf-8 -*-
"""
简单测试 Mem0Manager 是否可正常使用
"""

from src.configs.config import AppConfig
from src.memory.mem0_manager import Mem0Manager


async def main():
    # 1. 构造一个最简单的 AppConfig（你的真实项目里应该已经有配置）
    from src.configs.config import AppConfig
    config = AppConfig()

    # 2. 初始化 Mem0 管理器
    mem = Mem0Manager(config=config)
    await mem.init_memory_client()

    # print("\n=== 1. 添加记忆 add() ===")
    res_add = await mem.add(
        messages="我喜欢喝可乐。",
        metadata={"source": "test"},
        infer=False  # 让 mem0 自动抽取事实
    )
    print("添加结果：", res_add)

    # # 获取添加后的 memory_id
    memory_id = res_add["results"][0]["id"]

    print("\n=== 2. 获取所有记忆 get_all() ===")
    res_all = await mem.get_all()
    print("所有记忆：", res_all)

    print("\n=== 3. 搜索记忆 search() ===")
    res_search = await mem.search("喜欢什么")
    print("搜索结果：", res_search)

    print("\n=== 4. 获取单条记忆 get() ===")
    res_one = await mem.get(memory_id)
    print("单条记忆：", res_one)
    await mem.reset()
    print("\n=== 完成 ===")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
