# coding: utf-8
# Copyright (c) 2025 inclusionAI.
import os

from aworld import Client
from aworld.agents.gaia.agent import PlanAgent, ExcuteAgent
from aworld.config.conf import AgentConfig, TaskConfig
from aworld.core.envs.tool_desc import get_tool_desc
from aworld.core.swarm import Swarm
from aworld.core.task import GeneralTask
from aworld.dataset.mock import mock_dataset

os.environ['OPENAI_API_KEY'] = "adf"
if __name__ == '__main__':
    # Initialize client
    client = Client()

    # One sample for example
    test_sample = mock_dataset("gaia")
    print('task_prompt', test_sample)

    # Create agents
    agent_config = AgentConfig(
        llm_provider="openai",
        llm_model_name="gpt-4o",
    )
    agent1 = PlanAgent(conf=agent_config)
    agent2 = ExcuteAgent(conf=agent_config)

    # Create swarm for multi-agents
    # define (head_node, tail_node) edge in the topology graph
    swarm = Swarm((agent1, agent2))

    # Define a task
    task = GeneralTask(input=test_sample, swarm=swarm, conf=TaskConfig())

    print(get_tool_desc())

    # Run task
    # result = client.submit(task=[task])

    # print(f"Task completed: {result['success']}")
    # print(f"Time cost: {result['time_cost']}")
    # # print(f"Task Answer: {result['task_0']['answer']}")
    # print(result)
