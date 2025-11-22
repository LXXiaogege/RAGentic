import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.models.llm import LLMWrapper
from src.configs.logger_config import setup_logger
import json
import shutil
import os

# 设置日志记录器
logger = setup_logger(__name__)


class MCPClient:
    def __init__(self, llm: LLMWrapper):
        self.session: Optional[ClientSession] = None
        self.stdio = None
        self.write = None
        self.exit_stack = AsyncExitStack()
        self.available_tools = []  # 存储实际可用的工具

        self.llm = llm

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        logger.info(f"Attempting to connect to server at: {server_script_path}")
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            logger.error("Invalid server script file type")
            raise ValueError("Server script must be a .py or .js file")

        # 根据文件类型选择合适的命令
        command = None
        if is_python:
            # 对于Python文件，使用uv
            command = shutil.which("uv")
            if not command:
                logger.error("uv command not found in PATH")
                raise RuntimeError("uv command not found. Please install uv or add it to PATH")
            args = ['run', server_script_path]
            logger.debug(f"Using Python runner: {command} {' '.join(args)}")

        elif is_js:
            # 对于JavaScript文件，使用node
            command = shutil.which("node")
            if not command:
                logger.error("node command not found in PATH")
                raise RuntimeError("node command not found. Please install Node.js or add it to PATH")
            args = [server_script_path]
            logger.debug(f"Using JavaScript runner: {command} {' '.join(args)}")

        logger.debug(f"Using command: {command}")

        server_params = StdioServerParameters(
            command=command,
            args=['run', server_script_path],
            env=None
        )

        try:
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
            await self.session.initialize()

            # List available tools
            response = await self.session.list_tools()
            self.available_tools = response.tools
            logger.info(f"Successfully connected to server with tools: {[tool.name for tool in self.available_tools]}")
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Failed to connect to server: {str(e)}")
            raise
        finally:
            # 确保 exit_stack 被关闭，资源释放
            if hasattr(self, "exit_stack"):
                try:
                    await self.exit_stack.aclose()
                except (asyncio.CancelledError, ConnectionResetError) as e:
                    logger.warning(f"Cleanup error (logged for diagnostics): {e}")
                except Exception as e:
                    logger.exception(f"An unexpected error occurred during exit_stack cleanup.{e}")

    def _convert_mcp_tools_to_openai_format(self):
        """Convert MCP tools to OpenAI tool format"""
        openai_tools = []
        for tool in self.available_tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema if hasattr(tool, 'inputSchema') else {}
                }
            }
            openai_tools.append(openai_tool)
        return openai_tools

    async def process_query(self, query: str) -> str:
        """Process a query using openai and available tools"""
        logger.info(f"Processing query: {query}")
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        available_tools = self._convert_mcp_tools_to_openai_format()
        logger.debug(f"Available tools: {[tool['function']['name'] for tool in available_tools]}")

        try:
            response = self.llm.chat(
                max_tokens=1000,
                messages=messages,
                return_raw=True,
                tools=available_tools,
                tool_choice='auto'
            )
            logger.debug("Received response from LLM")
        except Exception as e:
            logger.error(f"Error in LLM chat: {str(e)}")
            raise

        message = response.choices[0].message

        if not message.tool_calls:
            logger.info("No tool calls required")
            return ""

        # 并发调用所有工具
        async def call_single_tool(tool_call):
            if tool_call.type == 'function':
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                logger.info(f"Calling tool: {tool_name} with args: {tool_args}")
                try:
                    tool_result = await self.session.call_tool(tool_name, tool_args)
                    if not tool_result.isError:
                        logger.info(f"Tool {tool_name} executed successfully")
                        if tool_result.content and len(tool_result.content) > 0:
                            content = tool_result.content[0]
                            if hasattr(content, 'text'):
                                return f"[Tool {tool_name} result]: {content.text}"
                            else:
                                return f"[Tool {tool_name} result]: {str(content)}"
                        return f"[Tool {tool_name} result]: {tool_result.content[0].text}"
                    else:
                        logger.error(f"Tool {tool_name} returned an error")
                        return f"[Error calling tool {tool_name}]"
                except Exception as e:
                    logger.error(f"Error executing tool {tool_name}: {str(e)}")
                    return f"[Exception calling tool {tool_name}]"
            else:
                return "[Unsupported tool call type]"

        tasks = [call_single_tool(tc) for tc in message.tool_calls]
        results = await asyncio.gather(*tasks)
        final_text = "\n".join(results)
        return final_text

    async def chat_loop(self, query):
        """Run an interactive chat loop"""
        logger.info("Starting chat loop")

        try:
            response = await self.process_query(query)
            logger.info("Successfully processed query")
            return "\n" + response
        except Exception as e:
            logger.error(f"Error in chat loop: {str(e)}")
            raise

    async def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up resources")
        try:
            await self.exit_stack.aclose()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


def _find_project_root():
    """查找项目根目录"""
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 向上查找，直到找到包含常见项目文件的目录
    project_markers = ['pyproject.toml', 'setup.py', 'requirements.txt', '.git', 'mcp_server.py']

    while current_dir != os.path.dirname(current_dir):  # 避免到达根目录
        for marker in project_markers:
            if os.path.exists(os.path.join(current_dir, marker)):
                return current_dir
        current_dir = os.path.dirname(current_dir)

    # 如果没找到项目标记，返回当前工作目录
    return os.getcwd()


async def mcp_main(client, query):
    logger.info(f"Starting MCP main with query: {query}")
    project_root = _find_project_root()
    server_script_path = os.path.join(project_root, "mcp_server.py")
    # 如果项目根目录没有，尝试当前工作目录
    if not os.path.exists(server_script_path):
        server_script_path = os.path.join(os.getcwd(), "mcp_server.py")

    # 如果还是没有，尝试相对于当前文件的路径
    if not os.path.exists(server_script_path):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        server_script_path = os.path.join(os.path.dirname(current_dir), "mcp_server.py")
    try:
        await client.connect_to_server(server_script_path)
        logger.info("Server connected, sending query")
        result = await client.chat_loop(query)
        return result
    except Exception as e:
        logger.error(f"Error in mcp_main: {str(e)}")
        raise
    finally:
        await client.cleanup()
        logger.info("MCP main completed")
