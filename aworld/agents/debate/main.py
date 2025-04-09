import os
from typing import Dict, Any, Union, List, Optional

from dotenv import load_dotenv
from pydantic import Field

from aworld.agents.debate.plan_agent import user_assignment_prompt, user_assignment_system_prompt, \
    user_debate_system_prompt, user_debate_prompt, DebatePlanAgent
from aworld.agents.debate.search_agent import SearchAgent
from aworld.config import load_config, AgentConfig, TaskConfig
from aworld.core.agent.base import BaseAgent
from aworld.core.agent.swarm import Swarm
from aworld.core.client import Client
from aworld.core.common import Observation, ActionModel, ActionResult
from aworld.core.envs.tool import ToolFactory
from aworld.core.task import Task
from aworld.logs.util import logger
from aworld.output.artifact import ArtifactType
from aworld.output.workspace import WorkSpace


#
# class DebateArena(BaseModel):
#     propositions: list[BaseAgent]
#     opposition: list[BaseAgent]
#     moderator: Optional[BaseAgent]
#     judges: Optional[BaseAgent]
#     display_panel: str
#
#


class DebateAgent(BaseAgent):
    topic: str = Field(default=None, description="The topic of the debate")

    opinion: str = Field(default=None, description="The opinion of the agent")

    workspace: WorkSpace

    def __init__(self, name: str, topic: str, opinion: str,
                 conf: AgentConfig, workspace: WorkSpace):
        conf.name = name
        super().__init__(conf)
        self.topic = topic
        self.opinion = opinion
        self.planner_agent = planner_agent
        self.search_agent = search_agent
        self.steps = 0
        self.workspace = workspace

    def policy(self, observation: Observation, info: Dict[str, Any] = {}, **kwargs) -> Union[
        List[ActionModel], None]:
        ## step 1: 对方观点
        opponent_claim = observation.content

        ## DEEPSEARCH Tool  & 前几轮的对话

        ## step 4: 呼叫己方，布置搜索任务，并赋值到observation里面
        messages = [{'role': 'system', 'content': user_debate_system_prompt},
                    {'role': 'user',
                     'content': user_debate_prompt.format(claim=opponent_claim, player='Michael Jordan',
                                                          search_materials=search_materials)}]

        llm_result = self.llm.invoke(
            input=messages,
        )

        user_response = llm_result.content

        print("user_response:", user_response)


if __name__ == '__main__':
    load_dotenv()

    agentConfig = AgentConfig(
        llm_provider="chatopenai",
        llm_model_name="bailing_80b_function_call",
        llm_base_url=os.environ['LLM_BASE_URL'],
        llm_api_key=os.environ['LLM_API_KEY'],
        max_steps=100,
    )

    planner_agent = DebatePlanAgent(conf=agentConfig)
    search_agent = SearchAgent(conf=agentConfig)

    # agent1 = DebateAgent(name="agent1", topic="Who's GOAT? Jordan or Lebron", opinion="Jordan",
    #                      planner_agent=planner_agent, search_agent=search_agent, conf=agentConfig)
    # agent2 = DebateAgent(name="agent2", topic="Who's GOAT? Jordan or Lebron", opinion="Lebron",
    #                      planner_agent=planner_agent, search_agent=search_agent, conf=agentConfig)

    workspace = WorkSpace.from_local_storages("demo1")
    agent1 = DebateAgent(name="agent1", topic="杭州适合年轻人生活吗", opinion="适合",
                          conf=agentConfig, workspace = workspace)
    input = Observation(content="杭州适合年轻人生活吗")
    agent1.policy(input)
