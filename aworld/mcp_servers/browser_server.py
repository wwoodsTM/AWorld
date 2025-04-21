"""
Browser MCP Server

This module provides MCP server functionality for browser automation and interaction.
It handles tasks such as web scraping, form submission, and automated browsing.

Main functions:
- browse_url: Opens a URL and performs specified actions
- submit_form: Fills and submits forms on web pages
"""

import asyncio
import json
import traceback

from browser_use import Agent
from browser_use.agent.views import AgentHistoryList
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from langchain_openai import ChatOpenAI
from pydantic import Field

from aworld.agents.gaia.xy_prompts import browser_system_prompt
from aworld.logs.util import logger
from aworld.mcp_servers.abc.base import MCPServerBase, mcp


def check_log_level(level_name):
    try:
        level_value = logger.level(level_name).no
        return True
    except ValueError:
        return False


class BrowserServer(MCPServerBase):
    """
    Browser Server class for browser automation and interaction.

    This class provides methods for web scraping, form submission, and automated browsing.
    """

    _instance = None
    _logger_context = None

    def __new__(cls):
        """Implement singleton pattern"""
        if cls._instance is None:
            cls._instance = super(BrowserServer, cls).__new__(cls)
            cls._instance._init_server()
            cls._instance._logger_context.browser("BrowserServer Created")
        return cls._instance

    def _init_server(self):
        """Initialize the Browser server and configuration"""
        spec_name = "BROWSER"
        color = "<bold><fg #979797>"
        self._logger_context = logger.bind(agent=spec_name)
        if not check_log_level(spec_name):
            self._logger_context.level(spec_name, no=25, color=color, icon="🧠")
        self._logger_context.browser = (
            lambda message, *message_args, **message_kwargs: self._logger_context.log(
                spec_name,
                message,
                *message_args,
                **message_kwargs,
            )
        )
        self._logger_context.add(
            "./agent-browser-use.log",
            rotation="1 week",
            compression="zip",
            format="{time} - {level} - {message}",
        )
        self._logger_context.browser("BrowserServer initialized")

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of BrowserServer"""
        if cls._instance is None:
            return cls()
        return cls._instance

    @mcp
    @classmethod
    async def browser_use(
        cls, task: str = Field(description="The task to perform using the browser.")
    ) -> str:
        """
        Perform browser actions using the browser-use package.
        Args:
            task (str): The task to perform using the browser.
        Returns:
            str: The result of the browser actions.
        """
        instance = cls.get_instance()
        instance._logger_context.browser(f">>> 🎯 Requested Task: {task}")
        browser = Browser(
            config=BrowserConfig(
                headless=False,
                new_context_config=BrowserContextConfig(
                    disable_security=True,
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    minimum_wait_page_load_time=10,
                    maximum_wait_page_load_time=30,
                ),
            )
        )
        browser_context = BrowserContext(
            config=BrowserContextConfig(trace_path="./logs"), browser=browser
        )
        agent = Agent(
            task=task,
            llm=ChatOpenAI(
                model="gpt-4o",
                base_url="http://localhost:3455",
                api_key="dummy-key",
            ),
            browser_context=browser_context,
            extend_system_message=browser_system_prompt,
        )
        try:
            browser_execution: AgentHistoryList = await agent.run(max_steps=20)
            if (
                browser_execution is not None
                and browser_execution.is_done()
                and browser_execution.is_successful()
            ):
                exec_trace = browser_execution.extracted_content()
                instance._logger_context.browser(
                    ">>> 🌏 Browse Execution Succeed!\n"
                    f">>> 💡 Result: {json.dumps(exec_trace, ensure_ascii=False, indent=4)}\n"
                    ">>> 🌏 Browse Execution Succeed!\n"
                )
                return browser_execution.final_result()
            else:
                return f"Browser execution failed for task: {task}"
        except Exception as e:
            instance._logger_context.error(
                f"Browser execution failed: {traceback.format_exc()}"
            )
            return f"Browser execution failed for task: {task} due to {str(e)}"
        finally:
            await browser.close()
            instance._logger_context.browser("Browser Closed!")


async def main():
    """Main function"""
    browser_server = BrowserServer.get_instance()
    logger.info("BrowserServer initialized and ready to handle requests")
    result = await browser_server.browser_use(
        task="What is the weather like in Shanghai today?"
    )
    logger.success(result)


# Main function
if __name__ == "__main__":
    asyncio.run(main())
