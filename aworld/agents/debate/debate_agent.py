from abc import ABC
from typing import Dict, Any, Union, List, Literal

from aworld.agents.debate.base import DebateSpeech
from aworld.agents.debate.deepsearch import deepsearch
from aworld.config import AgentConfig
from aworld.core.agent.base import Agent
from aworld.core.common import Observation, ActionModel
from aworld.output.artifact import ArtifactType
from aworld.output.workspace import WorkSpace


class DebateAgent(Agent, ABC):

    stance: Literal["affirmative", "negative"]

    def __init__(self, name: str, stance: Literal["affirmative", "negative"], conf: AgentConfig):
        conf.name = name
        super().__init__(conf)
        self.steps = 0
        self.stance = stance

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
        ## step 1: 对方观点
        opponent_claim = observation.content
        round = info["round"]
        opinion = info["opinion"]
        oppose_opinion = info["oppose_opinion"]
        topic = info["topic"]
        history: list[DebateSpeech] = info["history"]

        ## step2: 生成keywords

        ## step3：
        results = await deepsearch(self.llm, topic, opinion, oppose_opinion, opponent_claim, history)
        for result in  results:
            self.workspace.create_artifact(ArtifactType.WEB_PAGE, result)

        ## step3
        user_response = f"topic: {topic}: Round {round}\n"
        user_response += f"user: {self.name()}\n"
        user_response += f"result: mock result\n"

        action = ActionModel(
            policy_info=DebateSpeech(name=self.name(), type="speech", content=user_response, round=round)
        )

        return [action]

    def get_latest_speech(self, history: list[DebateSpeech]):
        """
        get the latest speech from history
        """
        if len(history) == 0:
            return None
        return history[-1]

    def set_workspace(self, workspace):
        self.workspace = workspace
