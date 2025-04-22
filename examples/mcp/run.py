# coding: utf-8
# Copyright (c) 2025 inclusionAI.

from aworld.config.conf import AgentConfig, TaskConfig
from aworld.core.agent.base import Agent
from aworld.core.task import Task
import asyncio
from aworld.logs.util import logger
from aworld.output.utils import consume_channel_messages
from aworld.utils.common import sync_exec

if __name__ == '__main__':
    agent_config = AgentConfig(
        llm_provider="openai",
        llm_model_name="gpt-4o",
        llm_api_key="YOUR_API_KEY",
        llm_base_url="http://localhost:5080"
    )

    search_sys_prompt = "You can use tools to calculate numbers and answer questions"
    search = Agent(
        conf=agent_config,
        name="search_agent",
        system_prompt=search_sys_prompt,
        #mcp_servers=["amap-amap-sse"],  # MCP server name for agent to use
        mcp_servers = ["simple-calculator"]  # MCP server name for agent to use
    )

    # Define a task
    Task(
        #input="Hotels within 1 kilometer of West Lake in Hangzhou", agent=search, conf=TaskConfig()
        input="30,000 divided by 1.2 ", agent=search, conf=TaskConfig()
    ).run()

def print_output(outputs):
    async def __log_item(item):
        logger.info(f"{type(item)}- content: {item}")

    async def process_messages():
        await consume_channel_messages(channel=outputs, callback=__log_item)

    # Run the async function in the event loop
    sync_exec(process_messages)
    print("finished...")
