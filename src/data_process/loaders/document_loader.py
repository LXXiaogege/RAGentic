# -*- coding: utf-8 -*-
"""
@Time ： 2025/4/12 13:31
@Auth ： 吕鑫
@File ：document_loader.py
@IDE ：PyCharm
"""
from typing import List, Dict, Type
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader, \
    UnstructuredWordDocumentLoader, \
    UnstructuredExcelLoader, TextLoader

from src.configs.logger_config import setup_logger
import os
import subprocess

# 配置日志记录器
logger = setup_logger(__name__)


class BaseLoader:
    """基础加载器类"""

    def load(self, file_path: str) -> List[Document]:
        raise NotImplementedError


class PDFLoader(BaseLoader):
    """PDF文件加载器"""

    def load(self, file_path: str) -> List[Document]:
        class_name = self.__class__.__name__
        logger.info(f"[{class_name}] 开始加载PDF文件: {file_path}")
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            logger.info(f"[{class_name}] PDF文件加载完成: {file_path}, 共 {len(documents)} 页")
            return documents
        except Exception as e:
            logger.error(f"[{class_name}] PDF文件加载失败: {file_path}, 错误: {str(e)}")
            raise


class MarkdownLoader(BaseLoader):
    """Markdown文件加载器"""

    def load(self, file_path: str) -> List[Document]:
        class_name = self.__class__.__name__
        logger.info(f"[{class_name}] 开始加载Markdown文件: {file_path}")
        try:
            loader = UnstructuredMarkdownLoader(file_path)
            documents = loader.load()
            logger.info(f"[{class_name}] Markdown文件加载完成: {file_path}, 共 {len(documents)} 个文档块")
            return documents
        except Exception as e:
            logger.error(f"[{class_name}] Markdown文件加载失败: {file_path}, 错误: {str(e)}")
            raise


class DocxLoader(BaseLoader):
    """Word文档加载器"""

    def load(self, file_path: str) -> List[Document]:
        class_name = self.__class__.__name__
        logger.info(f"[{class_name}] 开始加载Word文档: {file_path}")
        try:
            loader = UnstructuredWordDocumentLoader(file_path)
            documents = loader.load()
            logger.info(f"[{class_name}] Word文档加载完成: {file_path}, 共 {len(documents)} 个文档块")
            return documents
        except Exception as e:
            logger.error(f"[{class_name}] Word文档加载失败: {file_path}, 错误: {str(e)}")
            raise


class ExcelLoader(BaseLoader):
    """Excel文件加载器"""

    def load(self, file_path: str) -> List[Document]:
        class_name = self.__class__.__name__
        logger.info(f"[{class_name}] 开始加载Excel文件: {file_path}")
        try:
            loader = UnstructuredExcelLoader(file_path)
            documents = loader.load()
            logger.info(f"[{class_name}] Excel文件加载完成: {file_path}, 共 {len(documents)} 个文档块")
            return documents
        except Exception as e:
            logger.error(f"[{class_name}] Excel文件加载失败: {file_path}, 错误: {str(e)}")
            raise


class TxtLoader(BaseLoader):
    """文本文件加载器"""

    def load(self, file_path: str) -> List[Document]:
        class_name = self.__class__.__name__
        logger.info(f"[{class_name}] 开始加载文本文件: {file_path}")
        try:
            loader = TextLoader(file_path)
            documents = loader.load()
            logger.info(f"[{class_name}] 文本文件加载完成: {file_path}, 共 {len(documents)} 个文档块")
            return documents
        except Exception as e:
            logger.error(f"[{class_name}] 文本文件加载失败: {file_path}, 错误: {str(e)}")
            raise


class MineruLoader(BaseLoader):
    """Mineru文件加载器"""

    def preprocess_pdf_with_mineru_cli(self, pdf_path: str, output_dir: str):
        """
        使用 MinerU 命令行工具处理 PDF，并读取结构化结果。

        :param pdf_path: PDF 文件路径
        :param output_dir: 结果输出目录
        """
        assert os.path.exists(pdf_path), f"PDF 文件不存在：{pdf_path}"

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 构造 MinerU CLI 命令
        cmd = ["mineru", "-p", pdf_path, "-o", output_dir]
        class_name = self.__class__.__name__
        logger.info(f"[{class_name}] 运行命令: {' '.join(cmd)}")

        # 执行命令
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"[{class_name}] MinerU 命令运行失败: {result.stderr}")
            raise RuntimeError("MinerU PDF 解析失败")
        filename = os.path.basename(pdf_path)
        md_path = os.path.join(output_dir, filename.replace(".pdf", ".md"))
        content_json_path = os.path.join(output_dir, filename.replace(".pdf", "content_list.json"))
        return md_path, content_json_path

    def load(self, file_path: str) -> List[Document]:
        class_name = self.__class__.__name__
        logger.info(f"[{class_name}] Mineru解析pdf文件: {file_path}")
        try:
            md_file, content_json_file = self.preprocess_pdf_with_mineru_cli(file_path, "output")
            loader = UnstructuredMarkdownLoader(md_file)
            documents = loader.load()
            logger.info(f"[{class_name}] Mineru文件加载完成: {file_path}, 共 {len(documents)} 个文档块")
            return documents
        except Exception as e:
            logger.error(f"[{class_name}] Mineru文件加载失败: {file_path}, 错误: {str(e)}")
            raise


class DocumentLoader:
    def __init__(self, use_mineru: bool = False):
        """初始化文档加载器"""
        class_name = self.__class__.__name__
        logger.info(f"[{class_name}] 初始化文档加载器")

        self.use_mineru = use_mineru
        # 注册所有加载器
        self.loaders: Dict[str, Type[BaseLoader]] = {
            'pdf': PDFLoader,
            'mineru': MineruLoader,
            'md': MarkdownLoader,
            'docx': DocxLoader,
            'xlsx': ExcelLoader,
            'xls': ExcelLoader,
            'txt': TxtLoader
        }

        # 创建加载器实例
        self.loader_instances = {
            file_type: loader_class()
            for file_type, loader_class in self.loaders.items()
        }
        logger.info(f"[{class_name}] 已注册的加载器类型: {', '.join(self.loaders.keys())}")

    def get_loader(self, file_type: str) -> BaseLoader:
        """
        获取对应文件类型的加载器

        Args:
            file_type: 文件类型
        Returns:
            对应的加载器实例

        Raises:
            ValueError: 不支持的文件类型
        """
        class_name = self.__class__.__name__
        logger.debug(f"[{class_name}] 尝试获取文件类型 {file_type} 的加载器")
        if file_type == 'pdf' and self.use_mineru:
            file_type = 'mineru'
        loader = self.loader_instances.get(file_type)
        if loader is None:
            logger.error(f"[{class_name}] 不支持的文件类型: {file_type}")
            raise ValueError(f"不支持的文件类型: {file_type}")
        logger.debug(f"[{class_name}] 成功获取文件类型 {file_type} 的加载器")
        return loader

    def load(self, file_path: str) -> List[Document]:
        """
        加载文档

        Args:
            file_path: 文件路径

        Returns:
            List[Document]: 加载的文档列表

        Raises:
            ValueError: 不支持的文件类型
        """
        class_name = self.__class__.__name__
        logger.info(f"[{class_name}] 开始加载文件: {file_path}")
        # 获取文件扩展名，如果没有扩展名则默认为 txt
        file_type = file_path.split('.')[-1].lower() if '.' in file_path else 'txt'
        logger.debug(f"[{class_name}] 检测到文件类型: {file_type}")

        try:
            loader = self.get_loader(file_type)
            documents = loader.load(file_path)
            logger.info(f"[{class_name}] 文件加载完成: {file_path}, 共 {len(documents)} 个文档块")
            return documents
        except Exception as e:
            logger.error(f"[{class_name}] 文件加载失败: {file_path}, 错误: {str(e)}")
            raise
