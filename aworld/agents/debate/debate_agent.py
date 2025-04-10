import logging
from abc import ABC
from typing import Dict, Any, Union, List, Literal, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from aworld.agents.debate.base import DebateSpeech
from aworld.agents.debate.prompts import user_assignment_prompt, user_assignment_system_prompt, \
    user_debate_system_prompt, user_debate_prompt
from aworld.agents.debate.search.search_engine import SearchEngine
from aworld.agents.debate.search.tavily_search_engine import TavilySearchEngine
from aworld.config import AgentConfig
from aworld.core.agent.base import Agent
from aworld.core.common import Observation, ActionModel
from aworld.output.artifact import ArtifactType


def truncate_content(raw_content, char_limit):
    if raw_content is None:
        raw_content = ''
    if len(raw_content) > char_limit:
        raw_content = raw_content[:char_limit] + "... [truncated]"
    return raw_content

class DebateAgent(Agent, ABC):

    stance: Literal["affirmative", "negative"]

    def __init__(self, name: str, stance: Literal["affirmative", "negative"], conf: AgentConfig, search_engine: Optional[SearchEngine] = TavilySearchEngine()):
        conf.name = name
        super().__init__(conf)
        self.steps = 0
        self.stance = stance
        self.search_engine = search_engine

    async def speech(self, topic: str, opinion: str,oppose_opinion: str, round: int, speech_history: list[DebateSpeech]) -> DebateSpeech:
        observation = Observation(content=self.get_latest_speech(speech_history).content if self.get_latest_speech(speech_history) else "")
        info = {
            "topic": topic,
            "round": round,
            "opinion": opinion,
            "oppose_opinion": oppose_opinion,
            "history": speech_history
        }
        actions = await self.async_policy(observation, info)

        return actions[0].policy_info


    async def async_policy(self, observation: Observation, info: Dict[str, Any] = {}, **kwargs) -> Union[
        List[ActionModel], None]:
        ## step 1: params
        opponent_claim = observation.content
        round = info["round"]
        opinion = info["opinion"]
        oppose_opinion = info["oppose_opinion"]
        topic = info["topic"]
        history: list[DebateSpeech] = info["history"]

        #Event.emit("xxx")
        ## step2: gen keywords
        keywords = await self.gen_keywords(topic, opinion, oppose_opinion, opponent_claim, history)

        ## step3：search_webpages
        search_results = await self.search_webpages(keywords, max_results=5)
        logging.info(f"query keywords = {keywords}, result size = {len(search_results)}")
        for result in  search_results:
            self.workspace.create_artifact(ArtifactType.WEB_PAGE, result)

        ## step4 gen result
        user_response = await self.gen_statement(topic, opinion, oppose_opinion, opponent_claim, history, search_results)

        logging.info(f"user_response is {user_response}")

        action = ActionModel(
            policy_info=DebateSpeech(name=self.name(), type="speech", content=user_response, round=round)
        )

        return [action]

    async def gen_keywords(self, topic, opinion, oppose_opinion, last_oppose_speech_content, history):

        human_prompt = user_assignment_prompt.format(topic=topic,
                                                     opinion=opinion,
                                                     oppose_opinion=oppose_opinion,
                                                     last_oppose_speech_content=last_oppose_speech_content,
                                                     limit=2
                                                     )

        messages = [
            SystemMessage(content=user_assignment_system_prompt),
            HumanMessage(content=human_prompt)
        ]

        result = await self.llm.ainvoke(input=messages)
        return result.content.split(",")

    async def search_webpages(self, keywords, max_results):
        return await self.search_engine.async_batch_search(queries=keywords, max_results=max_results)

    async def gen_statement(self, topic, opinion, oppose_opinion, opponent_claim, history, search_results) -> str:
        search_results_content = ""
        for search_result in search_results:
            search_results_content += f"SearchQuery: {search_result['query']}"
            search_results_content += "\n\n".join([truncate_content(s['content'], 1000) for s in search_result['results']])

        human_prompt = user_debate_prompt.format(topic=topic,
                                                     opinion=opinion,
                                                     oppose_opinion=oppose_opinion,
                                                     last_oppose_speech_content=opponent_claim,
                                                     search_results_content=search_results_content
                                                     )

        messages = [
            SystemMessage(content=user_debate_system_prompt),
            HumanMessage(content=human_prompt)
        ]

        result = await self.llm.ainvoke(input=messages)
        return result.content

    def get_latest_speech(self, history: list[DebateSpeech]):
        """
        get the latest speech from history
        """
        if len(history) == 0:
            return None
        return history[-1]

    def set_workspace(self, workspace):
        self.workspace = workspace
