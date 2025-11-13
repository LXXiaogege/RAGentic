# -*- coding: utf-8 -*-
"""
@Time ： 2025/6/2 02:25
@Auth ： 吕鑫
@File ：mcp_server.py
@IDE ：PyCharm
"""
from fastmcp import FastMCP
import asyncio

from src.mcp.server.weather import mcp as weather_mcp
from src.mcp.server.web_crawl import mcp as web_crawl_mcp

main_mcp = FastMCP(name="MainApp")


# Import subserver
async def setup():
    await main_mcp.import_server("weather", weather_mcp)
    await main_mcp.import_server("web_crawl", web_crawl_mcp)


# Result: main_mcp now contains prefixed components:
# - Tool: "weather_get_forecast"
# - Resource: "data://weather/cities/supported"

if __name__ == "__main__":
    asyncio.run(setup())
    main_mcp.run()
