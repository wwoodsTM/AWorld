# coding: utf-8
# Copyright (c) 2025 inclusionAI.

from aworld.config.common import Tools, Agents
from aworld.agents.gaia.agent import PlanAgent, ExecuteAgent
from aworld.config.conf import AgentConfig, ModelConfig
from aworld.core.agent.swarm import Swarm
from aworld.dataset.mock import mock_dataset
from aworld.runner import Runners


# Need OPENAI_API_KEY
# os.environ['OPENAI_API_KEY'] = "your key"
# Optional endpoint settings, default `https://api.openai.com/v1`
# os.environ['OPENAI_ENDPOINT'] = "https://api.openai.com/v1"

def main():
    # One sample for example
    test_sample = mock_dataset("gaia")

    model_config = ModelConfig(
        llm_provider="openai",
        llm_model_name="gpt-4o",
        llm_temperature=1,
        # llm_api_key="your own key",
        # llm_base_url="http://localhost:5080"  ## paste your own llm server address
    )

    agent1_config = AgentConfig(
        name=Agents.PLAN.value,
        llm_config=model_config
    )
    agent1 = PlanAgent(conf=agent1_config)

    agent2_config = AgentConfig(
        name=Agents.EXECUTE.value,
        llm_config=model_config
    )
    agent2 = ExecuteAgent(conf=agent2_config, tool_names=[Tools.DOCUMENT_ANALYSIS.value])

    # Create swarm for multi-agents
    # define (head_node1, tail_node1), (head_node1, tail_node1) edge in the topology graph
    swarm = Swarm((agent1, agent2), sequence=False)

    # Run agents
    result = Runners.sync_run(input=test_sample, swarm=swarm)

    print(f"Time cost: {result['time_cost']}")
    print(f"Task Answer: {result['task_0']['answer']}")


if __name__ == '__main__':
    main()
