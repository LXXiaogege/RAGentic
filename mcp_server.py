# -*- coding: utf-8 -*-
"""
@Time ： 2025/6/2 02:25
@Auth ： 吕鑫
@File ：mcp_server.py
@IDE ：PyCharm
"""

from typing import Any
import httpx

from fastmcp import FastMCP
from src.agent.tools import WebSpider
from src.configs.logger_config import setup_logger
from src.skills.skill_manager import SkillManager

logger = setup_logger(__name__)

main_mcp = FastMCP(name="MainApp")

# ---- Weather tools ----
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"


async def _make_nws_request(url: str) -> dict[str, Any] | None:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None


@main_mcp.tool()
async def weather_get_alerts(state: str) -> str:
    """Get weather alerts for a US state.

    Args:
        state: Two-letter US state code (e.g. CA, NY)
    """
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    data = await _make_nws_request(url)
    if not data or "features" not in data:
        return "Unable to fetch alerts or no alerts found."
    if not data["features"]:
        return "No active alerts for this state."
    alerts = []
    for f in data["features"]:
        props = f["properties"]
        alerts.append(
            f"Event: {props.get('event', 'Unknown')}\n"
            f"Area: {props.get('areaDesc', 'Unknown')}\n"
            f"Severity: {props.get('severity', 'Unknown')}\n"
            f"Description: {props.get('description', 'No description')}"
        )
    return "\n---\n".join(alerts)


@main_mcp.tool()
async def weather_get_forecast(latitude: float, longitude: float) -> str:
    """Get weather forecast for a location.

    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
    """
    points_data = await _make_nws_request(
        f"{NWS_API_BASE}/points/{latitude},{longitude}"
    )
    if not points_data:
        return "Unable to fetch forecast data for this location."
    forecast_url = points_data["properties"]["forecast"]
    forecast_data = await _make_nws_request(forecast_url)
    if not forecast_data:
        return "Unable to fetch detailed forecast."
    periods = forecast_data["properties"]["periods"]
    forecasts = []
    for period in periods[:5]:
        forecasts.append(
            f"{period['name']}:\n"
            f"Temperature: {period['temperature']}°{period['temperatureUnit']}\n"
            f"Wind: {period['windSpeed']} {period['windDirection']}\n"
            f"Forecast: {period['detailedForecast']}"
        )
    return "\n---\n".join(forecasts)


# ---- Web crawl tool ----
_web_spider = WebSpider()
_skill_manager = SkillManager("skills")


@main_mcp.tool()
async def web_crawl(url: str) -> str:
    """爬取网页内容

    Args:
        url: 要爬取的url
    """
    try:
        logger.info(f"开始爬取URL: {url}")
        result = _web_spider.crawl_page(url=url)
        content = result.get("content", "")[:1000] or "网页无内容"
        logger.info(f"成功爬取URL: {url}")
        return content
    except Exception as e:
        logger.exception(f"网页爬虫异常: {str(e)}")
        return f"网页爬取失败: {str(e)}"


# ---- Skills tool ----
@main_mcp.tool()
def read_skill(name: str) -> str:
    """读取指定 skill 的完整指令内容。当你判断用户请求匹配某个 skill 时调用此工具，获取指令后严格按照指令执行。

    Args:
        name: skill 名称，必须是 available_skills 列表中的名称之一
    """
    body = _skill_manager.get_skill_body(name)
    if body is None:
        available = list(_skill_manager.skills.keys())
        return f"Skill '{name}' 不存在。可用的 skills：{available}"
    return body


# ---- Knowledge Base search tool ----
@main_mcp.tool()
async def kb_search(query: str, top_k: int = 3, use_hyde: bool = False) -> str:
    """搜索知识库获取相关文档。当你需要查询特定事实、信息或知识时调用此工具。

    Args:
        query: 搜索查询词或问题
        top_k: 返回的文档数量，默认3条
        use_hyde: 是否使用 HyDE 检索增强模式，默认关闭
    """
    return "[KB_SEARCH_PLACEHOLDER] 此工具由Pipeline直接执行"


# ---- Query rewrite tool ----
@main_mcp.tool()
async def query_rewrite(query: str, mode: str = "rewrite") -> str:
    """将用户查询改写为更精确的表述。支持三种模式：
    - rewrite: 简单改写，让查询更清晰准确
    - step_back: 生成泛化查询，用于检索更广泛的相关信息
    - sub_query: 分解为多个子查询，分别检索后合并结果

    Args:
        query: 原始用户查询
        mode: 改写模式，可选 rewrite/step_back/sub_query
    """
    return "[QUERY_REWRITE_PLACEHOLDER] 此工具由Pipeline直接执行"


if __name__ == "__main__":
    main_mcp.run()
