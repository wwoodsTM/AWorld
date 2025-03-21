# coding: utf-8
# Copyright (c) 2025 inclusionAI.
from aworld.core.client import Client
from aworld.core.common import Agents, Tools
from aworld.core.task import GeneralTask
from aworld.agents.browser.agent import BrowserAgent
from aworld.config.conf import AgentConfig
from aworld.virtual_environments import BrowserTool
from aworld.virtual_environments.conf import BrowserToolConfig

if __name__ == '__main__':
    client = Client()
    browser_tool_config = BrowserToolConfig(window_w=1280, window_h=720, keep_browser_open=True)

    agent_config = AgentConfig(
        agent_name=Agents.BROWSER.value,
        llm_provider="openai",
        llm_model_name="gpt-4o",
        llm_num_ctx=32000,
        llm_temperature=1,
    )

    task_config = {
        'max_steps': 100,
        'max_actions_per_step': 100
    }

    client.submit(
        GeneralTask(input="go to google.com and type 'AntGroup' click search and click the first search result",
                    agent=BrowserAgent(conf=agent_config),
                    tools=[BrowserTool(conf=browser_tool_config)],
                    task_config=task_config))

