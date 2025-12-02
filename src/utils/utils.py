# -*- coding: utf-8 -*-
"""
@Time ： 2025/5/6 01:51
@Auth ： 吕鑫
@File ：utils.py
@IDE ：PyCharm
"""
import hashlib
from urllib.parse import urlparse
from src.configs.logger_config import setup_logger

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
