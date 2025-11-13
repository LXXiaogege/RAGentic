# -*- coding: utf-8 -*-
"""
@Time ： 2025/6/1 02:00
@Auth ： 吕鑫
@File ：function_call_service.py
@IDE ：PyCharm
"""
# from mcp.server.fastmcp import FastMCP
from src.agent.tools import WebSpider
from src.config.logger_config import setup_logger
from fastmcp import FastMCP


logger = setup_logger(__name__)

# 初始化 FastMCP 服务器
mcp = FastMCP("web_crawl")

# 初始化必要的组件
web_spider = WebSpider()


@mcp.tool()
async def web_crawl(url: str) -> str:
    """爬取网页内容

    Args:
        url: 要爬取的url
    """
    try:
        logger.info(f"开始爬取URL: {url}")
        result = web_spider.crawl_page(url=url)
        content = result.get("content", "")[:1000] or "网页无内容"
        logger.info(f"成功爬取URL: {url}")
        return content
    except Exception as e:
        logger.exception(f"网页爬虫异常: {str(e)}")
        return f"网页爬取失败: {str(e)}"


def main():
    """启动 MCP 服务"""
    logger.info("启动函数调用 MCP 服务")
    mcp.run(transport='stdio')


if __name__ == "__main__":
    main()
