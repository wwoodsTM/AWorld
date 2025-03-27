# coding: utf-8
# Copyright (c) 2025 inclusionAI.

from aworld.agents import AndroidAgent
from aworld.config import AgentConfig
from aworld.core.client import Client
from aworld.core.common import Agents
from aworld.core.task import GeneralTask
from aworld.virtual_environments.android.android import AndroidTool
from aworld.virtual_environments.conf import AndroidToolConfig


def main():
    client = Client()
    android_tool_config = AndroidToolConfig(avd_name='8ABX0PHWU',
                                            headless=False,
                                            max_retry=2,
                                            max_episode_steps=None
                                            )

    agent_config: AgentConfig = AgentConfig(
        agent_name=Agents.ANDROID.value,
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
        GeneralTask(input="""open rednote""",
                    agent=AndroidAgent(conf=agent_config),
                    tools=[AndroidTool(conf=android_tool_config)],
                    task_config=task_config))


if __name__ == '__main__':
    main()
