# -*- coding: utf-8 -*-
"""
@Time ： 2025/4/12 14:17
@Auth ： 吕鑫
@File ：llm.py
@IDE ：PyCharm
"""
from typing import List, Dict, Union, cast
from src.config.config import QAPipelineConfig
from openai.types.chat import ChatCompletionMessageParam
# from openai import OpenAI
from langfuse.openai import OpenAI
from src.config.logger_config import setup_logger
from langchain_core.messages import BaseMessage
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
    def __init__(self, config: QAPipelineConfig):
        self.logger = setup_logger(f"{__name__}.LLMWrapper")
        provider = config.llm_provider
        self.model = config.llm_model

        if provider == "openai":
            self.client = OpenAILLM(config.llm_api_key, config.llm_base_url)
        else:
            raise ValueError(f"暂不支持的模型提供商: {provider}")

    def convert_messages_to_dicts(self, messages: List[BaseMessage]) -> List[Dict]:
        type_to_role = {
            "human": "user",
            "ai": "assistant",
            "system": "system",
            "function": "function",
            "tool": "tool",
        }

        formatted = []
        for msg in messages:
            role = type_to_role.get(msg.get("type"))
            if role is None:
                raise ValueError(f"Unsupported message type: {msg.get('type')}")
            formatted.append({"role": role, "content": msg.get("content")})
        return formatted

    def chat(self, messages: List[Union[Dict, BaseMessage]], stream: bool = False, return_raw: bool = False, **kwargs):
        # 如果是 LangChain 消息，先转换为 OpenAI 所需格式
        if messages and isinstance(messages[0], BaseMessage):
            messages = self.convert_messages_to_dicts(messages)
        response = self.client.chat(model=self.model, messages=messages, stream=stream, **kwargs)
        if return_raw or stream:
            return response  # 返回完整响应，适合高级用法
        return response.choices[0].message.content  # 默认解析出content


if __name__ == '__main__':
    # SiliconFlow 参数配置（请根据自己账号填写）
    from src.config.config import QAPipelineConfig

    llm_api_key: str = QAPipelineConfig.llm_api_key
    llm_base_url: str = QAPipelineConfig.llm_base_url
    llm_model: str = QAPipelineConfig.llm_model

    llm = OpenAILLM(api_key=llm_api_key, base_url=llm_base_url)

    messages = [
        {"role": "user", "content": "请简要介绍一下SiliconFlow是什么？请简要回答"}
    ]

    # 非流式调用测试
    response = llm.chat(model=llm_model, messages=messages)
    content = response.choices[0].message.content
    print("[✅ 非流式输出内容]:", content)

    # 流式调用测试
    # print("[✅ 流式输出内容]:", end="", flush=True)
    # response_stream = llm.chat(model=model, messages=messages, stream=True)
    # for chunk in response_stream:
    #     delta = chunk.choices[0].delta
    #     print(delta.content or "", end="", flush=True)
    # print("\n[✅ 流式输出完成]")
