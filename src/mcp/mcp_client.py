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

logger = setup_logger(__name__)


class MCPClient:
    def __init__(self, llm: LLMWrapper):
        self.session: Optional[ClientSession] = None
        self.stdio = None
        self.write = None
        self.exit_stack: Optional[AsyncExitStack] = None
        self.available_tools = []

        self.llm = llm
        self._connected = False

    async def __aenter__(self) -> "MCPClient":
        """Async context manager entry"""
        self.exit_stack = AsyncExitStack()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit"""
        await self.cleanup()

    async def connect_to_server(self, server_script_path: str) -> None:
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        if self._connected:
            logger.info("MCP client already connected")
            return

        logger.info(f"Attempting to connect to server at: {server_script_path}")
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            logger.error("Invalid server script file type")
            raise ValueError("Server script must be a .py or .js file")

        command = None
        if is_python:
            command = shutil.which("uv")
            if not command:
                logger.error("uv command not found in PATH")
                raise RuntimeError(
                    "uv command not found. Please install uv or add it to PATH"
                )
            args = ["run", server_script_path]
            logger.debug(f"Using Python runner: {command} {' '.join(args)}")

        elif is_js:
            command = shutil.which("node")
            if not command:
                logger.error("node command not found in PATH")
                raise RuntimeError(
                    "node command not found. Please install Node.js or add it to PATH"
                )
            args = [server_script_path]
            logger.debug(f"Using JavaScript runner: {command} {' '.join(args)}")

        logger.debug(f"Using command: {command}")

        server_params = StdioServerParameters(
            command=command, args=["run", server_script_path], env=None
        )

        self.exit_stack = AsyncExitStack()
        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write)
            )
            await self.session.initialize()

            response = await self.session.list_tools()
            self.available_tools = response.tools
            self._connected = True
            logger.info(
                f"Successfully connected to server with tools: {[tool.name for tool in self.available_tools]}"
            )
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Failed to connect to server: {str(e)}")
            raise

    def _convert_mcp_tools_to_openai_format(self) -> list:
        """Convert MCP tools to OpenAI tool format"""
        openai_tools = []
        for tool in self.available_tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                    if hasattr(tool, "inputSchema")
                    else {},
                },
            }
            openai_tools.append(openai_tool)
        return openai_tools

    async def execute_tool_calls(self, tool_calls: list) -> list:
        """Execute a list of tool calls (LangChain ToolCall format), return result strings"""

        async def call_single_tool(tool_call):
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            logger.info(f"Calling tool: {tool_name} with args: {tool_args}")
            try:
                tool_result = await self.session.call_tool(tool_name, tool_args)
                if not tool_result.isError and tool_result.content:
                    content = tool_result.content[0]
                    text = content.text if hasattr(content, "text") else str(content)
                    logger.info(f"Tool {tool_name} executed successfully")
                    return f"[Tool {tool_name} result]: {text}"
                logger.error(f"Tool {tool_name} returned an error")
                return f"[Error calling tool {tool_name}]"
            except (AttributeError, RuntimeError, asyncio.TimeoutError) as e:
                logger.error(
                    f"Error executing tool {tool_name}: {type(e).__name__}: {e}"
                )
                return f"[Exception calling tool {tool_name}: {type(e).__name__}: {e}]"
            except Exception as e:
                logger.error(
                    f"Unexpected error executing tool {tool_name}: {type(e).__name__}: {e}"
                )
                return f"[Unexpected error calling tool {tool_name}: {type(e).__name__}: {e}]"

        return await asyncio.gather(*[call_single_tool(tc) for tc in tool_calls])

    async def process_query(self, query: str) -> str:
        """Process a query using openai and available tools"""
        logger.info(f"Processing query: {query}")
        messages = [{"role": "user", "content": query}]

        available_tools = self._convert_mcp_tools_to_openai_format()
        logger.debug(
            f"Available tools: {[tool['function']['name'] for tool in available_tools]}"
        )

        try:
            response = self.llm.chat(
                max_tokens=1000,
                messages=messages,
                return_raw=True,
                tools=available_tools,
                tool_choice="auto",
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            logger.debug("Received response from LLM")
        except Exception as e:
            logger.error(f"Error in LLM chat: {str(e)}")
            raise

        message = response.choices[0].message

        if not message.tool_calls:
            logger.info("No tool calls required")
            return ""

        results = await self.execute_tool_calls(
            [
                {
                    "name": tc.function.name,
                    "args": json.loads(tc.function.arguments),
                    "id": tc.id,
                    "type": "function",
                }
                for tc in message.tool_calls
            ]
        )
        final_text = "\n".join(results)
        return final_text

    async def chat_loop(self, query: str) -> str:
        """Run an interactive chat loop"""
        logger.info("Starting chat loop")

        try:
            response = await self.process_query(query)
            logger.info("Successfully processed query")
            return "\n" + response
        except Exception as e:
            logger.error(f"Error in chat loop: {str(e)}")
            raise

    async def cleanup(self) -> None:
        """Clean up resources"""
        logger.info("Cleaning up resources")
        try:
            if self.exit_stack:
                await self.exit_stack.aclose()
            self._connected = False
            self.session = None
            self.stdio = None
            self.write = None
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    @property
    def is_connected(self) -> bool:
        return self._connected


def _find_project_root() -> str:
    """查找项目根目录"""
    current_dir = os.path.dirname(os.path.abspath(__file__))

    project_markers = [
        "pyproject.toml",
        "setup.py",
        "requirements.txt",
        ".git",
        "mcp_server.py",
    ]

    while current_dir != os.path.dirname(current_dir):
        for marker in project_markers:
            if os.path.exists(os.path.join(current_dir, marker)):
                return current_dir
        current_dir = os.path.dirname(current_dir)

    return os.getcwd()


async def mcp_main(client: MCPClient, query: str) -> str:
    logger.info(f"Starting MCP main with query: {query}")
    project_root = _find_project_root()
    server_script_path = os.path.join(project_root, "mcp_server.py")
    if not os.path.exists(server_script_path):
        server_script_path = os.path.join(os.getcwd(), "mcp_server.py")

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


if __name__ == "__main__":
    from src.configs.config import AppConfig
    from src.models.llm import LLMWrapper

    config = AppConfig()
    llm = LLMWrapper(config.llm)
    client = MCPClient(llm)
    query = "洛杉矶今天天气怎么样？"
    tool_result = asyncio.run(mcp_main(client, query))
    logger.info(f"工具调用结果：{tool_result}")
