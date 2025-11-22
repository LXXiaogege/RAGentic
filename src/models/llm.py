# -*- coding: utf-8 -*-
"""
@Time ： 2025/4/12 14:17
@Auth ： 吕鑫
@File ：llm.py
@IDE ：PyCharm
"""
from typing import List, Dict, Union, cast
from src.configs.model_config import LLMConfig
from openai.types.chat import ChatCompletionMessageParam
# from openai import OpenAI
from langfuse.openai import OpenAI
from src.configs.logger_config import setup_logger
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, FunctionMessage, ToolMessage
from langfuse import observe

logger = setup_logger(__name__)


class BaseLLM:
    def chat(self, model: str, messages: List[ChatCompletionMessageParam], stream: bool = False, **kwargs):
        raise NotImplementedError


class OpenAILLM(BaseLLM):
    def __init__(self, api_key: str, base_url: str):
        self.logger = setup_logger(f"{__name__}.OpenAILLM")
        self.logger.info("初始化 OpenAI LLM 客户端...")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.logger.debug(f"使用 API 基础 URL: {base_url}")
        self.logger.info("OpenAI LLM 客户端初始化完成")

    @observe(name="LLMWrapper.chat", as_type="generation")
    def chat(self, model: str, messages: List[dict], stream: bool = False, **kwargs):
        """
        与 LLM 进行对话
        
        Args:
            model: 模型名称
            messages: 消息列表
            stream: 是否使用流式输出
            **kwargs: 其他参数
        """
        self.logger.info(f"开始与模型 {model} 对话...")
        self.logger.debug(f"消息数量: {len(messages)}")
        self.logger.debug(f"流式输出: {stream}")

        typed_messages = cast(List[ChatCompletionMessageParam], messages)
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=typed_messages,
                stream=stream,
                **kwargs
            )
            self.logger.info("成功获取模型响应")
            return response
        except Exception as e:
            self.logger.exception(f"调用模型时发生错误: {str(e)}")
            raise


class LLMWrapper:
    def __init__(self, config: LLMConfig):
        self.logger = setup_logger(f"{__name__}.LLMWrapper")
        self.config = config
        if self.config.provider == "openai":
            self.client = OpenAILLM(self.config.api_key, self.config.base_url)
        else:
            raise ValueError(f"暂不支持的模型提供商: {self.config.provider}")

    def convert_messages_to_dicts(self, messages: List[BaseMessage]) -> List[Dict]:
        type_to_role = {
            HumanMessage: "user",
            AIMessage: "assistant",
            SystemMessage: "system",
            FunctionMessage: "function",
            ToolMessage: "tool",
        }
        formatted = []
        for msg in messages:
            role = None
            for msg_type, role_name in type_to_role.items():
                if isinstance(msg, msg_type):
                    role = role_name
                    break
            if role is None:
                raise ValueError(f"Unknown message type: {type(msg)}")
            formatted.append({"role": role, "content": msg.content})
        return formatted

    def chat(self, messages: List[Union[Dict, BaseMessage]], stream: bool = False, return_raw: bool = False, **kwargs):
        # 如果是 LangChain 消息，先转换为 OpenAI 所需格式
        if messages and isinstance(messages[0], BaseMessage):
            messages = self.convert_messages_to_dicts(messages)
        response = self.client.chat(model=self.config.model, messages=messages, stream=stream, **kwargs)
        if return_raw or stream:
            return response  # 返回完整响应，适合高级用法
        return response.choices[0].message.content  # 默认解析出content
