# -*- coding: utf-8 -*-
"""
@Time ： 2025/11/29 13:19
@Auth ： 吕鑫
@File：test_memory.py
@IDE：PyCharm

Memory 系统异步测试用例
"""

import asyncio
from src.configs.config import AppConfig
from src.memory.mem0_manager import Mem0Manager


async def test_mem0_basic():
    """测试 Mem0Manager 基本功能"""
    config = AppConfig()
    mem = Mem0Manager(config=config)
    await mem.init_memory_client()

    res_add = await mem.add(
        messages="我喜欢喝可乐。",
        metadata={"source": "test"},
        infer=False,
    )
    print("添加结果：", res_add)

    memory_id = res_add["results"][0]["id"]

    res_all = await mem.get_all()
    print("所有记忆：", res_all)

    res_search = await mem.search("喜欢什么")
    print("搜索结果：", res_search)

    res_one = await mem.get(memory_id)
    print("单条记忆：", res_one)

    await mem.reset()
    print("✓ Mem0 basic test passed")


async def test_mem0_search():
    """测试 Mem0 搜索功能"""
    config = AppConfig()
    mem = Mem0Manager(config=config, user_id="test_user_search")
    await mem.init_memory_client()

    await mem.add(
        messages=[
            {"role": "user", "content": "我住在上海"},
            {"role": "assistant", "content": "好的，我知道你住在上海了"},
        ],
        infer=False,
    )

    await mem.add(
        messages=[
            {"role": "user", "content": "我喜欢编程"},
            {"role": "assistant", "content": "好的，我记住了你喜欢编程"},
        ],
        infer=False,
    )

    results = await mem.search("住在哪里")
    print(f"搜索 '住在哪里' 结果数: {len(results)}")
    for r in results:
        print(f"  - {r.get('text', '')[:50]}...")

    await mem.reset()
    print("✓ Mem0 search test passed")


async def test_mem0_delete():
    """测试 Mem0 删除功能"""
    config = AppConfig()
    mem = Mem0Manager(config=config, user_id="test_user_delete")
    await mem.init_memory_client()

    res_add = await mem.add(
        messages="这条记录稍后会被删除",
        infer=False,
    )
    memory_id = res_add["results"][0]["id"]

    await mem.delete(memory_id)

    all_memories = await mem.get_all()
    deleted_found = any(m.get("id") == memory_id for m in all_memories)
    assert not deleted_found, "Memory should have been deleted"

    await mem.reset()
    print("✓ Mem0 delete test passed")


async def main():
    print("=" * 60)
    print("Running Memory Tests")
    print("=" * 60)

    try:
        await test_mem0_basic()
    except Exception as e:
        print(f"✗ Basic test failed: {e}")

    try:
        await test_mem0_search()
    except Exception as e:
        print(f"✗ Search test failed: {e}")

    try:
        await test_mem0_delete()
    except Exception as e:
        print(f"✗ Delete test failed: {e}")

    print("=" * 60)
    print("All memory tests completed")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
