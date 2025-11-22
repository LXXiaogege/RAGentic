# -*- coding: utf-8 -*-
"""
@Time ： 2025/4/12 13:31
@Auth ： 吕鑫
@File ：text_processor.py
@IDE ：PyCharm
"""
import re
from langchain_core.documents import Document
from typing import List
from src.configs.logger_config import setup_logger

logger = setup_logger(__name__)


class TextProcessor:
    def __init__(self, text_splitter):
        """
        初始化文本处理器
        """
        self.logger = logger
        self.logger.info("初始化文本处理器")
        self.text_splitter = text_splitter
        self.logger.debug(f"使用文本分割器: {type(text_splitter).__name__}")

    def clean_text(self, text: str) -> str:
        """
        清理文本内容

        Args:
            text: 待清理的文本

        Returns:
            str: 清理后的文本
        """
        self.logger.debug(f"开始清理文本，原始长度: {len(text)}")

        # 处理非中文字符之间的换行符
        pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
        text = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), text)
        self.logger.debug("已处理非中文字符间的换行符")

        # 移除特殊字符
        text = text.replace('•', '')
        text = text.replace(' ', '')
        self.logger.debug("已移除特殊字符")

        self.logger.debug(f"文本清理完成，处理后长度: {len(text)}")
        return text

    def process_document(self, doc: Document) -> Document:
        """
        处理单个文档

        Args:
            doc: 待处理的文档

        Returns:
            Document: 处理后的文档
        """
        self.logger.info(f"开始处理文档，元数据: {doc.metadata}")
        original_length = len(doc.page_content)

        doc.page_content = self.clean_text(doc.page_content)

        self.logger.info(f"文档处理完成，内容长度从 {original_length} 变为 {len(doc.page_content)}")
        return doc

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        分割文档列表

        Args:
            documents: 待分割的文档列表

        Returns:
            List[Document]: 分割后的文档列表
        """
        self.logger.info(f"开始分割文档列表，原始文档数: {len(documents)}")

        try:
            split_docs = self.text_splitter.split_documents(documents)
            self.logger.info(f"文档分割完成，分割后文档数: {len(split_docs)}")
            return split_docs
        except Exception as e:
            self.logger.exception(f"文档分割过程中发生错误:{e}")
            raise
