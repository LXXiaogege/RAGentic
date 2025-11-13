# -*- coding: utf-8 -*-
"""
@Time ： 2025/5/6 01:51
@Auth ： 吕鑫
@File ：utils.py
@IDE ：PyCharm
"""
import hashlib
from urllib.parse import urlparse
from src.config.logger_config import setup_logger

# 配置日志记录器
logger = setup_logger(__name__)


def get_text_hash(text: str) -> str:
    """
    计算文本的MD5哈希值
    :param text: 输入文本
    :return: MD5哈希值
    """
    logger.debug(f"开始计算文本哈希值，文本长度: {len(text)}")
    hash_value = hashlib.md5(text.encode("utf-8")).hexdigest()
    logger.debug(f"文本哈希值计算完成: {hash_value}")
    return hash_value


def extract_url_by_llm(llm, query: str):
    """
    使用LLM从查询中提取URL
    :param llm: 语言模型实例
    :param query: 用户查询
    :return: 提取的URL或None
    """
    logger.info(f"开始从查询中提取URL: {query[:100]}...")

    messages = [
        {"role": "system",
         "content": "/no_think 你是一个网址提取助手，请从用户输入中提取出真正的网址，无需赘述，只返回一个 URL。"},
        {"role": "user", "content": query}
    ]

    try:
        logger.debug("调用LLM进行URL提取")
        answer = llm.chat(messages=messages, stream=False, return_raw=False)
        content = answer.strip().replace('<think>', '').replace('</think>', '')
        logger.debug(f"LLM返回结果: {content}")

        for word in content.split():
            parsed = urlparse(word)
            if parsed.scheme and parsed.netloc:
                logger.info(f"成功提取URL: {word}")
                return word

        logger.warning("未能从LLM返回结果中提取到有效URL")
        return None

    except Exception as e:
        logger.error(f"URL提取过程发生错误: {str(e)}")
        return None
