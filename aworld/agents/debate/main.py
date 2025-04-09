import logging
import os
from typing import Dict, Any, Union, List, Optional, Literal

from dotenv import load_dotenv
from pydantic import BaseModel

from aworld.agents.debate.deepsearch import deepsearch
from aworld.config import AgentConfig
from aworld.core.agent.base import BaseAgent
from aworld.core.common import Observation, ActionModel
from aworld.memory.base import MemoryItem
from aworld.memory.main import Memory
from aworld.output.workspace import WorkSpace

class DebateSpeech(BaseModel):
    name: str
    type: str
    content: str
    round: int


class DebateAgent(BaseAgent):

    workspace: WorkSpace

    stance: Literal["affirmative", "negative"]

    def __init__(self, name: str, stance: Literal["affirmative", "negative"], conf: AgentConfig):
        conf.name = name
        super().__init__(conf)
        self.steps = 0
        self.stance = stance
        self.workspace = None

    def speech(self, topic: str, opinion: str,oppose_opinion: str, round: int, speech_history: list[DebateSpeech]) -> DebateSpeech:
        observation = Observation(content=self.get_latest_speech(speech_history).content if self.get_latest_speech(speech_history) else "")
        info = {
            "topic": topic,
            "round": round,
            "opinion": opinion,
            "oppose_opinion": oppose_opinion,
            "history": speech_history
        }
        return self.policy(observation, info)[0].policy_info

    def policy(self, observation: Observation, info: Dict[str, Any] = {}, **kwargs) -> Union[
        List[ActionModel], None]:
        ## step 1: 对方观点
        opponent_claim = observation.content
        round = info["round"]
        opinion = info["opinion"]
        oppose_opinion = info["oppose_opinion"]
        topic = info["topic"]
        history: list[DebateSpeech] = info["history"]

        ## step2
        results = deepsearch(self.llm, topic, opinion, oppose_opinion, opponent_claim, history)

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

class DebateArena:
    """
    DebateArena is platform for debate
    """

    affirmative_speaker: DebateAgent
    negative_speaker: DebateAgent

    moderator: Optional[BaseAgent]
    judges: Optional[BaseAgent]


    speeches: list[DebateSpeech]

    display_panel: str

    def __init__(self,
                 affirmative_speaker: DebateAgent,
                 negative_speaker: DebateAgent,
                 **kwargs
                 ):
        super().__init__()
        self.affirmative_speaker = affirmative_speaker
        self.negative_speaker = negative_speaker
        self.memory = Memory.from_config(config={
            "memory_store": "inmemory"
        })
        self.speeches=[]


    def start_debate(self, topic: str, affirmative_opinion: str, negative_opinion: str, rounds: int) -> list[DebateSpeech]:
        """
        Start the debate
        1. debate will start from round 0
        2. each round will have two speeches, one from affirmative_speaker and one from negative_speaker
        3. after all rounds finished, the debate will end

        Args:
            topic: str -> topic of the debate
            affirmative_opinion: str -> affirmative speaker's opinion
            negative_opinion: str -> negative speaker's opinion
            rounds: int -> number of rounds

        Returns: list[DebateSpeech]

        """
        for i in range(rounds):
            logging.info(f"round#{i} start")

            # affirmative_speech
            speech = self.affirmative_speech(i, topic, affirmative_opinion)
            self.speeches.append(speech)

            # negative_speech
            speech = self.negative_speech(i, topic, negative_opinion)
            self.speeches.append(speech)

            logging.info(f"round#{i} end")
        return self.speeches

    def affirmative_speech(self, round: int, topic: str, opinion: str) -> DebateSpeech:
        """
        affirmative_speaker will start speech
        """

        affirmative_speaker = self.get_affirmative_speaker()

        logging.info(affirmative_speaker.name() + ": " + "start")

        speech = affirmative_speaker.speech(topic, opinion, round, self.speeches)
        self.store_speech(speech)

        logging.info(affirmative_speaker.name() + ":  result: " + speech.content)

    def negative_speech(self, round: int, topic: str, opinion: str) -> DebateSpeech:
        """
        after affirmative_speaker finished speech, negative_speaker will start speech
        """

        negative_speaker = self.get_negative_speaker()

        logging.info(negative_speaker.name() + ": " + "start")

        speech = negative_speaker.speech(topic, opinion, round, self.speeches)
        
        self.store_speech(speech)

        logging.info(negative_speaker.name() + ":  result: " + speech.content)

    def get_affirmative_speaker(self) -> DebateAgent:
        """
        return the affirmative speaker
        """
        return self.affirmative_speaker

    def get_negative_speaker(self) -> DebateAgent:
        """
        return the negative speaker
        """
        return self.negative_speaker

    def store_speech(self, speech: DebateSpeech):
        self.memory.add(MemoryItem.from_dict({
            "content": speech.content,
            "metadata": {
                "round": speech.round,
                "speaker": speech.name,
                "type": speech.type
            }
        }))
        self.speeches.append(speech)




if __name__ == '__main__':
    load_dotenv()

    agentConfig = AgentConfig(
        llm_provider="chatopenai",
        llm_model_name="bailing_moe_plus_function_call",
        llm_base_url=os.environ['LLM_BASE_URL'],
        llm_api_key=os.environ['LLM_API_KEY'],
    )

    agent1 = DebateAgent(name="Zhitian", stance="affirmative", conf=agentConfig)
    agent2 = DebateAgent(name="Daowen", stance="negative", conf=agentConfig)

    debateArena = DebateArena(affirmative_speaker=agent1, negative_speaker=agent2)

    debateArena.start_debate(topic="Who's GOAT? Jordan or Lebron", affirmative_opinion="Jordan",
                             negative_opinion="Lebron", rounds=3)