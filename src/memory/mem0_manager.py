# -*- coding: utf-8 -*-
"""
@Time ： 2025/11/27 17:19
@Auth ： 吕鑫
@File ：mem0_manager.py
@IDE ：PyCharm
"""
from typing import Dict, Optional, Any, List, Union
from mem0 import AsyncMemory
from src.memory.memory_adapter import adapt_appconfig_to_mem0
from src.configs.config import AppConfig


class Mem0Manager:
    def __init__(self, config: AppConfig, user_id: Optional[str] = None):
        """
        初始化 Mem0 管理器

        :param config: 应用配置对象
        :param user_id: 默认 user_id，如果后续方法调用未指定 user_id，将使用此值
        """
        self.config = adapt_appconfig_to_mem0(config)
        self.m: Optional[AsyncMemory] = None
        self.default_user_id = user_id

    async def init_memory_client(self):
        self.m = await AsyncMemory.from_config(self.config)
        await self.ensure_collection()

    async def ensure_collection(self):
        """
        如果 collection 不存在，则创建一个空的（通过 add 一条空数据，再立即删除）
        """
        try:
            # 尝试 get_all，如果 collection 不存在会报错
            await self.m.get_all(limit=1)
        except Exception:
            # 创建一个空记忆来强制初始化 collection
            system_user = self.default_user_id or "__system__"
            res = await self.m.add(
                messages="__init__",
                user_id=system_user,
                infer=False,
                metadata={"init": True}
            )
            # 获取 id 并删除掉
            mem_id = res["results"][0]["id"]
            await self.m.delete(mem_id)

    def _get_user_id(self, user_id: Optional[str]) -> Optional[str]:
        """内部辅助方法：获取最终使用的 user_id"""
        return user_id if user_id is not None else self.default_user_id

    async def add(
            self,
            messages: Union[str, List[Dict[str, str]]],
            user_id: Optional[str] = None,
            agent_id: Optional[str] = None,
            run_id: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
            infer: bool = False,
            memory_type: Optional[str] = None,
            prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        添加新记忆。

        Args:
            messages: 消息内容字符串或消息列表
                     (e.g., `[{"role": "user", "content": "Hello"}]`)
            user_id: 用户ID。如果为None，尝试使用初始化时的 default_user_id。
            agent_id: 代理ID。
            run_id: 运行ID。
            metadata: 存储的元数据。
            infer: 如果为 True (默认)，使用 LLM 提取事实；如果为 False，直接作为原始记忆存储。
            memory_type: 记忆类型。例如 "procedural_memory" 用于程序性记忆。
            prompt: 用于记忆创建的自定义 Prompt。

        Returns:
            dict: 包含添加结果的字典 (例如 {"results": [...]})
        """
        return await self.m.add(
            messages=messages,
            user_id=self._get_user_id(user_id),
            agent_id=agent_id,
            run_id=run_id,
            metadata=metadata,
            infer=infer,
            memory_type=memory_type,
            prompt=prompt
        )

    async def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        获取单条记忆详情
        """
        return await self.m.get(memory_id)

    async def get_all(
            self,
            user_id: Optional[str] = None,
            agent_id: Optional[str] = None,
            run_id: Optional[str] = None,
            filters: Optional[Dict[str, Any]] = None,
            limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        获取所有记忆列表

        Args:
            user_id: 用户ID
            agent_id: 代理ID
            run_id: 运行ID
            filters: 过滤条件
            limit: 返回数量限制，默认 100

        Returns:
            List: 记忆条目列表 (直接返回 results 列表)
        """
        res = await self.m.get_all(
            user_id=self._get_user_id(user_id),
            agent_id=agent_id,
            run_id=run_id,
            filters=filters,
            limit=limit
        )
        return res.get("results", [])

    async def update(self, memory_id: str, data: str) -> Dict[str, Any]:
        """
        更新已有记忆内容

        Args:
            memory_id: 记忆ID
            data: 新的文本内容 (对应 mem0 update 方法的 data 参数)
        """
        return await self.m.update(memory_id=memory_id, data=data)

    async def delete(self, memory_id: str) -> Dict[str, Any]:
        """
        删除某条记忆
        """
        return await self.m.delete(memory_id=memory_id)

    async def delete_all(
            self,
            user_id: Optional[str] = None,
            agent_id: Optional[str] = None,
            run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        删除指定范围内的所有记忆

        Args:
            user_id: 用户ID
            agent_id: 代理ID
            run_id: 运行ID
        """
        return await self.m.delete_all(
            user_id=self._get_user_id(user_id),
            agent_id=agent_id,
            run_id=run_id
        )

    async def memory_history(self, memory_id: str) -> List[Dict[str, Any]]:
        """
        获取某条记忆的变更历史
        """
        return await self.m.history(memory_id=memory_id)

    async def reset(self):
        """
        重置记忆存储 (删除向量库集合，重置数据库)
        """
        await self.m.reset()

    async def search(
            self,
            query: str,
            user_id: Optional[str] = None,
            agent_id: Optional[str] = None,
            run_id: Optional[str] = None,
            limit: int = 100,
            filters: Optional[Dict[str, Any]] = None,
            threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        基于查询搜索相关记忆

        Args:
            query: 搜索查询文本
            user_id: 用户ID
            agent_id: 代理ID
            run_id: 运行ID
            limit: 返回结果数量限制
            filters: 高级元数据过滤 (支持 eq, ne, gt, lt, contains 等操作符)
            threshold: 相似度阈值 (可选)

        Returns:
            dict: 搜索结果，包含 "results" 键
        """
        return await self.m.search(
            query=query,
            user_id=self._get_user_id(user_id),
            agent_id=agent_id,
            run_id=run_id,
            limit=limit,
            filters=filters,
            threshold=threshold
        )
