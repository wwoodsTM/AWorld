# coding: utf-8
# Copyright (c) 2025 inclusionAI.

from aworld.config.conf import AgentConfig, TaskConfig
from aworld.core.agent.base import Agent
from aworld.core.task import Task

if __name__ == '__main__':
    agent_config = AgentConfig(
        llm_provider="openai",
        llm_model_name="gpt-4o",
        llm_api_key="sk-zk275220716ee96209ceb688938ebab7a2067e414723d0e2",
        llm_base_url="https://api.zhizengzeng.com/v1"
    )

    search_sys_prompt = "You are a helpful agent."
    search = Agent(
        conf=agent_config,
        name="search_agent",
        # todo:tool_promot
        system_prompt=search_sys_prompt,  # sys_prompt,
        mcp_servers=["amap-amap-sse"],  # MCP server name for agent to use
        use_call_tool=False
        # mcp_servers = ["simple-calculator"]  # MCP server name for agent to use
    )

    # Define a task
    Task(
        #input="Hotels within 1 kilometer of West Lake in Hangzhou", agent=search, conf=TaskConfig()
        input="杭州西湖一公里以内的3星级酒店,列出10家即可", agent=search, conf=TaskConfig()
    ).run()