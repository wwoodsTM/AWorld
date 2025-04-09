import os
from typing import Dict, Any, Union, List, Optional

from dotenv import load_dotenv
from pydantic import Field, BaseModel

from aworld.agents.debate.plan_agent import user_assignment_prompt, user_assignment_system_prompt, \
    user_debate_system_prompt, user_debate_prompt, DebatePlanAgent
from aworld.agents.debate.search.tavily_search_engine import TavilySearchEngine
from aworld.agents.debate.search_agent import SearchAgent
from aworld.config import load_config, AgentConfig, TaskConfig
from aworld.core.agent.base import BaseAgent, Agent
from aworld.core.agent.swarm import Swarm
from aworld.core.client import Client
from aworld.core.common import Observation, ActionModel, ActionResult
from aworld.core.envs.tool import ToolFactory
from aworld.core.task import Task
from aworld.logs.util import logger
from aworld.output.artifact import ArtifactType
from aworld.output.workspace import WorkSpace



class DebateArena(BaseModel):
    proposition: BaseAgent
    opposition: BaseAgent
    moderator: Optional[BaseAgent]
    judges: Optional[BaseAgent]
    display_panel: str


    def start_debate(self, topic):
        pass


class SearchResult:
    id: str
    url: str
    title: str
    content: str

def deepsearch(topic, option, other_option,history) -> list[SearchResult]:
    search_engine = TavilySearchEngine()
    results = search_engine.async_batch_search(queries=["杭州天气怎么样", "xxx"], max_results=5)
    pass



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
        history = []

        ## DEEPSEARCH Tool  & 前几轮的对话

        results = deepsearch()

        parsed_result = ""

        ## step 4: 呼叫己方，布置搜索任务，并赋值到observation里面
        messages = [{'role': 'system', 'content': user_debate_system_prompt},
                    {'role': 'user',
                     'content': user_debate_prompt.format(claim=opponent_claim, player='Michael Jordan',
                                                          search_materials=results)}]

        llm_result = self.llm.invoke(
            input=messages,
        )

        user_response = llm_result.content

        print("user_response:", user_response)


if __name__ == '__main__':
    load_dotenv()

    DebateArena(

    ).start_debate(topic="")

