import logging
import os
from typing import Dict, Any, Union, List, Optional

from dotenv import load_dotenv
from pydantic import Field, BaseModel

from aworld.agents.debate.plan_agent import user_debate_system_prompt, user_debate_prompt
from aworld.agents.debate.search.tavily_search_engine import TavilySearchEngine
from aworld.config import AgentConfig
from aworld.core.agent.base import BaseAgent
from aworld.core.common import Observation, ActionModel
from aworld.output.artifact import ArtifactType
from aworld.output.workspace import WorkSpace


class SearchResult:
    id: str
    url: str
    title: str
    content: str


def deepsearch(topic, option, other_option, history) -> list[SearchResult]:
    search_engine = TavilySearchEngine()
    results = search_engine.async_batch_search(queries=["杭州天气怎么样", "xxx"], max_results=5)
    pass


class DebateSpeech(BaseModel):
    name: str
    type: str
    content: str
    round: int


class DebateAgent(BaseAgent):
    topic: str = Field(default=None, description="The topic of the debate")

    opinion: str = Field(default=None, description="The opinion of the agent")

    workspace: WorkSpace

    def __init__(self, name: str, topic: str, opinion: str,
                 conf: AgentConfig):
        conf.name = name
        super().__init__(conf)
        self.topic = topic
        self.opinion = opinion
        self.steps = 0
        self.workspace = None

    def speech(self, topic: str, opinion: str, round: int, speech_history: list[DebateSpeech]) -> DebateSpeech:
        self.policy()
        pass


    def policy(self, observation: Observation, info: Dict[str, Any] = {}, **kwargs) -> Union[
        List[ActionModel], None]:
        ## step 1: 对方观点
        opponent_claim = observation.content
        history = []

        ## DEEPSEARCH Tool  & 前几轮的对话

        results = deepsearch()

        # for result in results:
        #     self.workspace.create_artifact(ArtifactType.WEB_PAGE, result)

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



class DebateArena(BaseModel):

    affirmative_speaker: DebateAgent
    negative_speaker: DebateAgent

    moderator: Optional[BaseAgent]
    judges: Optional[BaseAgent]

    def __init__(self,
                 affirmative_speaker: DebateAgent,
                 negative_speaker: DebateAgent,
                 moderator: Optional[BaseAgent],
                 judges: Optional[BaseAgent],
                 **kwargs
                 ):
        super().__init__()
        self.affirmative_speaker = affirmative_speaker
        self.negative_speaker = negative_speaker
        self.moderator = moderator
        self.judges = judges

    speeches: list[DebateSpeech]

    display_panel: str

    def start_debate(self, topic, affirmative_option, negative_option, rounds):

        for i in range(rounds):
            logging.info(f"round#{i} start")

            # affirmative_speech
            speech = self.affirmative_speech(i, topic, affirmative_option)
            self.speeches.append(speech)

            # negative_speech
            speech = self.negative_speech(i, topic, negative_option)
            self.speeches.append(speech)

            logging.info(f"round#{i} end")
        pass

    def affirmative_speech(self, round: int, topic: int) -> DebateSpeech:
        affirmative_speaker = self.get_affirmative_speaker()

        logging.info(affirmative_speaker.name() + ": " + "start")

        observation = Observation(content=affirmative_speaker.opinion, info = {
            "speeches": self.speeches
        })
        policy = affirmative_speaker.policy(observation)

        logging.info(affirmative_speaker.name() + ": " + "end")

        return DebateSpeech(name=affirmative_speaker.name())

    def get_affirmative_speaker(self) -> BaseAgent:
        # hook
        return self.affirmative_speaker

    def get_affirmative_speaker(self):
        # hook
        return self.affirmative_speaker

    def negative_speech(self) -> DebateSpeech:
        logging.info(self.affirmative_speaker.name() + ": " + "start")
        self.proposition.policy()
        logging.info(self.affirmative_speaker.name() + ": " + "end")
        return DebateSpeech()


if __name__ == '__main__':
    load_dotenv()

    agentConfig = AgentConfig(
        llm_provider="chatopenai",
        llm_model_name="bailing_moe_plus_function_call",
        llm_base_url=os.environ['LLM_BASE_URL'],
        llm_api_key=os.environ['LLM_API_KEY'],
        max_steps=100,
    )

    agent1 = DebateAgent(name="agent1", topic="Who's GOAT? Jordan or Lebron", opinion="Jordan",
                         conf=agentConfig)
    agent1.llm.invoke()

    # agent2 = DebateAgent(name="agent2", topic="Who's GOAT? Jordan or Lebron", opinion="Lebron",
    #                       conf=agentConfig)
    #
    #
    #
    # agent = DebateArena()
    #
    #
    # DebateArena().start_debate(topic="", round)
