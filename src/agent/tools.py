# -*- coding: utf-8 -*-
"""
@Time ： 2025/4/12 13:31
@Auth ： 吕鑫
@File ：tools.py
@IDE ：PyCharm
"""
import requests
from lxml import html
from typing import Dict, List
from urllib.parse import urlparse
from src.config.logger_config import setup_logger
from langfuse import observe

logger = setup_logger(__name__)


class WebSpider:
    def __init__(self):
        """
        初始化WebSpider
        """
        self.logger = logger
        self.default_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36",
        }
        self.timeout = 10
        self.logger.info("初始化 WebSpider")

    def _clean_text(self, text: str) -> str:
        """
        清理文本内容
        :param text: 原始文本
        :return: 清理后的文本
        """
        cleaned = text.replace('\t', '').replace('\n', '').replace('\r', '').strip()
        self.logger.debug(f"清理文本: 原始长度={len(text)}, 清理后长度={len(cleaned)}")
        return cleaned

    def _is_valid_url(self, url: str) -> bool:
        """
        验证URL是否有效
        :param url: 待验证的URL
        :return: 是否有效
        """
        try:
            result = urlparse(url)
            is_valid = all([result.scheme, result.netloc])
            self.logger.debug(f"URL验证结果: {url} - {'有效' if is_valid else '无效'}")
            return is_valid
        except Exception as e:
            self.logger.error(f"URL验证异常: {url}, 错误: {str(e)}")
            return False

    def crawl_page(self, url: str) -> Dict:
        """
        爬取指定URL的页面内容
        :param url: 目标URL
        :return: 包含标题和正文的字典
        """
        self.logger.info(f"开始爬取页面: {url}")

        if not self._is_valid_url(url):
            self.logger.error(f"无效的URL: {url}")
            raise ValueError(f"无效的URL: {url}")

        try:
            # 发送HTTP请求
            self.logger.debug(f"发送HTTP请求: {url}")
            response = requests.get(url, headers=self.default_headers, timeout=self.timeout)
            response.encoding = response.apparent_encoding
            tree = html.fromstring(response.text)

            # 提取标题
            title_list = tree.xpath('/html/head/title/text()')
            title = self._clean_text(title_list[0]) if title_list else ''
            self.logger.debug(f"提取到标题: {title[:50]}...")

            # 提取正文文本
            text_list = tree.xpath('//text()[not(ancestor::script) and not(ancestor::style) and not(ancestor::meta)]')
            text_list = [self._clean_text(x) for x in text_list if x.strip()]
            text_list = [x for x in text_list if x != '\n' and x != '|'
                         and not x.startswith('<style')
                         and not x.startswith('<meta')
                         and not x.startswith('<script')]

            if len(text_list) > 1:
                text_list = text_list[1:]

            text = ' '.join(text_list)
            self.logger.debug(f"提取到正文，长度: {len(text)}")

            result = {
                'title': title,
                'content': text,
                'url': url,
                'status': 'success'
            }

            self.logger.info(f"成功爬取页面: {url}, 标题长度: {len(title)}, 内容长度: {len(text)}")
            return result

        except requests.exceptions.RequestException as e:
            self.logger.error(f"请求失败: {url}, 错误: {str(e)}")
            return {
                'title': '',
                'content': '',
                'url': url,
                'status': 'error',
                'error': str(e)
            }
        except Exception as e:
            self.logger.error(f"解析失败: {url}, 错误: {str(e)}")
            return {
                'title': '',
                'content': '',
                'url': url,
                'status': 'error',
                'error': str(e)
            }

    def batch_crawl(self, urls: List[str]) -> List[Dict]:
        """
        批量爬取多个URL
        :param urls: URL列表
        :return: 爬取结果列表
        """
        self.logger.info(f"开始批量爬取，URL数量: {len(urls)}")
        results = []
        for i, url in enumerate(urls, 1):
            self.logger.info(f"正在爬取第 {i}/{len(urls)} 个URL: {url}")
            result = self.crawl_page(url)
            results.append(result)
        self.logger.info(f"批量爬取完成，成功数量: {len([r for r in results if r['status'] == 'success'])}")
        return results

    @observe(name="WebSpider.run", as_type="tool")
    def run(self, url: str) -> str:
        self.logger.info(f"执行run方法，URL: {url}")
        result = self.crawl_page(url)
        content = result.get("content", "")[:1000] or "网页无内容"
        self.logger.info(f"run方法执行完成，返回内容长度: {len(content)}")
        return content
