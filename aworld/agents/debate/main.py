import asyncio
import logging
import os
import uuid
from typing import Optional, AsyncGenerator

from dotenv import load_dotenv

from aworld.agents.debate.base import DebateSpeech
from aworld.agents.debate.debate_agent import DebateAgent
from aworld.config import AgentConfig
from aworld.core.agent.base import BaseAgent
from aworld.memory.base import MemoryItem
from aworld.memory.main import Memory
from aworld.output import Output,WorkSpace


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
                 workspace: WorkSpace,
                 **kwargs
                 ):
        super().__init__()
        self.affirmative_speaker = affirmative_speaker
        self.negative_speaker = negative_speaker
        self.memory = Memory.from_config(config={
            "memory_store": "inmemory"
        })
        self.speeches=[]
        self.workspace = workspace
        self.affirmative_speaker.set_workspace(workspace)
        self.negative_speaker.set_workspace(workspace)
        # Event.register("topic", func= );

    async def async_run(self, topic: str, affirmative_opinion: str, negative_opinion: str, rounds: int)\
            -> AsyncGenerator[Output, None]:

        """
        Start the debate
        1. debate will start from round 1
        2. each round will have two speeches, one from affirmative_speaker and one from negative_speaker
        3. after all rounds finished, the debate will end

        Args:
            topic: str -> topic of the debate
            affirmative_opinion: str -> affirmative speaker's opinion
            negative_opinion: str -> negative speaker's opinion
            rounds: int -> number of rounds

        Returns: list[DebateSpeech]

        """
        for i in range(1, rounds+1):
            logging.info(f"✈️==================================== round#{i} start =============================================")

            # affirmative_speech
            speech = await self.affirmative_speech(i, topic, affirmative_opinion, negative_opinion)
            yield speech
            self.speeches.append(speech)

            # negative_speech
            speech = await self.negative_speech(i, topic, negative_opinion, affirmative_opinion)
            self.speeches.append(speech)
            yield speech

            logging.info(f"🛬==================================== round#{i} end =============================================")

    async def affirmative_speech(self, round: int, topic: str, opinion: str, oppose_opinion: str) -> DebateSpeech:
        """
        affirmative_speaker will start speech
        """

        affirmative_speaker = self.get_affirmative_speaker()

        logging.info(affirmative_speaker.name() + ": " + "start")

        speech = await affirmative_speaker.speech(topic, opinion, oppose_opinion, round, self.speeches)
        self.store_speech(speech)

        logging.info(affirmative_speaker.name() + ":  result: " + speech.content)
        return speech

    async def negative_speech(self, round: int, topic: str, opinion: str, oppose_opinion: str) -> DebateSpeech:
        """
        after affirmative_speaker finished speech, negative_speaker will start speech
        """

        negative_speaker = self.get_negative_speaker()

        logging.info(negative_speaker.name() + ": " + "start")

        speech = await negative_speaker.speech(topic, opinion, oppose_opinion, round, self.speeches)
        
        self.store_speech(speech)

        logging.info(negative_speaker.name() + ":  result: " + speech.content)
        return speech


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

    def gen_closing_statement(self):
        pass


async def start_debate():
    speeches = debateArena.async_run(topic="Who's GOAT? Jordan or Lebron", affirmative_opinion="Jordan",
                                     negative_opinion="Lebron", rounds=3)
    async for speech in speeches:
        print(f"{speech.name}: {speech.content}")


if __name__ == '__main__':
    load_dotenv()

    agentConfig = AgentConfig(
        llm_provider="chatopenai",
        llm_model_name="bailing_moe_plus_function_call",
        llm_base_url=os.environ['LLM_BASE_URL'],
        llm_api_key=os.environ['LLM_API_KEY'],
    )

    agent1 = DebateAgent(name="affirmativeSpeaker", stance="affirmative", conf=agentConfig)
    agent2 = DebateAgent(name="negativeSpeaker", stance="negative", conf=agentConfig)

    debateArena = DebateArena(affirmative_speaker=agent1, negative_speaker=agent2,
                              workspace=WorkSpace.from_local_storages(str(uuid.uuid4())))

    asyncio.run(start_debate())