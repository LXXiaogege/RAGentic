# -*- coding: utf-8 -*-
"""
@Time ： 2025/4/12 13:31
@Auth ： 吕鑫
@File ：data_processor.py
@IDE ：PyCharm
"""
from typing import List
from langchain_core.documents import Document
from src.data_process.processors.text_processor import TextProcessor
from src.data_process.loaders.document_loader import DocumentLoader
import os
from src.config.logger_config import setup_logger

logger = setup_logger(__name__)


class DataProcessor:
    def __init__(self, text_splitter, use_mineru=False):
        """
        初始化数据处理器
        """
        self.logger = logger
        self.logger.info("初始化数据处理器...")
        self.document_loader = DocumentLoader(use_mineru=use_mineru)
        self.text_processor = TextProcessor(text_splitter)
        self.logger.info("数据处理器初始化完成")

    def process_file(self, file_path: str) -> List[Document]:
        """
        根据文件扩展名自动选择处理方式

        Args:
            file_path: 文件路径

        Returns:
            List[Document]: 处理后的文档列表

        Raises:
            ValueError: 不支持的文件类型
        """
        self.logger.info(f"开始处理文件: {file_path}")

        try:
            # 加载文档
            self.logger.debug("加载文档...")
            documents = self.document_loader.load(file_path)
            self.logger.info(f"成功加载文档，原始文档数: {len(documents)}")

            # 处理每个文档
            self.logger.debug("处理文档内容...")
            processed_docs = [self.text_processor.process_document(doc) for doc in documents]
            self.logger.info(f"文档处理完成，处理后的文档数: {len(processed_docs)}")

            # 分割文档
            self.logger.debug("分割文档...")
            split_docs = self.text_processor.split_documents(processed_docs)
            self.logger.info(f"文档分割完成，最终文档数: {len(split_docs)}")

            return split_docs

        except Exception as e:
            self.logger.exception(f"处理文件 {file_path} 时发生错误:{e}")
            raise

    def batch_process_files(self, folder_path: str) -> List[Document]:
        """
        批量处理多个文件
        """
        self.logger.info(f"开始批量处理文件夹: {folder_path}")

        file_names = os.listdir(folder_path)
        file_paths = [os.path.join(folder_path, file_name) for file_name in file_names]
        self.logger.info(f"找到 {len(file_paths)} 个文件待处理")

        all_documents = []
        success_count = 0
        error_count = 0

        for file_path in file_paths:
            try:
                self.logger.info(f"处理文件 {file_path}...")
                processed_docs = self.process_file(file_path)
                all_documents.extend(processed_docs)
                success_count += 1
                self.logger.info(f"文件 {file_path} 处理成功，生成 {len(processed_docs)} 个文档片段")
            except Exception as e:
                error_count += 1
                self.logger.exception(f"处理文件 {file_path} 时出错: {str(e)}")
                continue

        self.logger.info(f"批量处理完成，成功: {success_count} 个文件，失败: {error_count} 个文件")
        self.logger.info(f"总共生成 {len(all_documents)} 个文档片段")
        return all_documents
