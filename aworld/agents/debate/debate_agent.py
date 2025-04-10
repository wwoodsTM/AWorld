import logging
import re
from abc import ABC
from typing import Dict, Any, Union, List, Literal, Optional
from datetime import datetime

from langchain_core.messages import SystemMessage, HumanMessage

from aworld.agents.debate.base import DebateSpeech
from aworld.agents.debate.prompts import user_assignment_prompt, user_assignment_system_prompt, affirmative_few_shots, negative_few_shots, \
    user_debate_system_prompt, user_debate_prompt
from aworld.agents.debate.search.search_engine import SearchEngine
from aworld.agents.debate.search.tavily_search_engine import TavilySearchEngine
from aworld.config import AgentConfig
from aworld.core.agent.base import Agent
from aworld.core.common import Observation, ActionModel
from aworld.output import SearchOutput, SearchItem
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
        logging.info(f"gen keywords = {keywords}")

        ## step3：search_webpages
        search_results = await self.search_webpages(keywords, max_results=5)
        for search_result in search_results:
            logging.info(f"keyword#{search_result['query']}-> result size is {len(search_result['results'])}")
            search_item = {
                "query": search_result.get("query", ""),
                "results": [SearchItem(title=result["title"],url=result["url"], metadata={}) for result in search_result["results"]]
            }
            self.workspace.create_artifact(ArtifactType.WEB_PAGES, content=SearchOutput.from_dict(search_item))

        ## step4 gen result
        user_response = await self.gen_statement(topic, opinion, oppose_opinion, opponent_claim, history, search_results)

        logging.info(f"user_response is {user_response}")

        action = ActionModel(
            policy_info=DebateSpeech(name=self.name(), type="speech", stance=self.stance, content=user_response, round=round)
        )

        return [action]

    async def gen_keywords(self, topic, opinion, oppose_opinion, last_oppose_speech_content, history):

        current_time = datetime.now().strftime("%Y-%m-%d-%H")
        human_prompt = user_assignment_prompt.format(topic=topic,
                                                     opinion=opinion,
                                                     oppose_opinion=oppose_opinion,
                                                     last_oppose_speech_content=last_oppose_speech_content,
                                                     current_time = current_time,
                                                     limit=2
                                                     )

        messages = [
            SystemMessage(content=user_assignment_system_prompt),
            HumanMessage(content=human_prompt)
        ]

        result = await self.async_call_llm(messages)

        return result.split(",")

    async def async_call_llm(self, messages):
        def _resolve_think(content):
            import re
            start_tag = 'think'
            end_tag = '/think'
            # 使用正则表达式提取标签内的内容
            llm_think = ""
            match = re.search(
                rf"<{re.escape(start_tag)}(.*?)>(.|\n)*?<{re.escape(end_tag)}>",
                content,
                flags=re.DOTALL,
            )
            if match:
                llm_think = match.group(0).replace("<think>", "").replace("</think>", "")
            llm_result = re.sub(
                rf"<{re.escape(start_tag)}(.*?)>(.|\n)*?<{re.escape(end_tag)}>",
                "",
                content,
                flags=re.DOTALL,
            )
            return llm_think, llm_result

        result = await self.llm.ainvoke(input=messages)
        llm_think, llm_result = _resolve_think(result.content)
        return llm_result

    async def search_webpages(self, keywords, max_results):
        return await self.search_engine.async_batch_search(queries=keywords, max_results=max_results)

    async def gen_statement(self, topic, opinion, oppose_opinion, opponent_claim, history, search_results) -> str:
        search_results_content = ""
        for search_result in search_results:
            search_results_content += f"SearchQuery: {search_result['query']}"
            search_results_content += "\n\n".join([truncate_content(s['content'], 1000) for s in search_result['results']])

        unique_history = history
        # if len(history) >= 2:
        #     for i in range(len(history)):
        #         # Check if the current element is the same as the next one
        #         if i == len(history) - 1 or history[i] != history[i+1]:
        #             # Add the current element to the result list
        #             unique_history.append(history[i])


        affirmative_chat_history = ""
        negative_chat_history = ""
 
        if len(unique_history) >= 2:
            if self.stance == "affirmative":
                for speech in unique_history[:-1]:
                    if speech.stance == "affirmative":
                        affirmative_chat_history = affirmative_chat_history + "You: " + speech.content + "\n"
                    elif speech.stance == "negative":
                        affirmative_chat_history = affirmative_chat_history + "Your Opponent: " + speech.content + "\n"

            elif self.stance == "negative":
                for speech in unique_history[:-1]:
                    if speech.stance == "negative":
                        negative_chat_history = negative_chat_history + "You: " + speech.content + "\n"
                    elif speech.stance == "affirmative":
                        negative_chat_history = negative_chat_history + "Your Opponent: " + speech.content + "\n"

        few_shots = ""

        if self.stance == "affirmative":
            chat_history = affirmative_chat_history
            few_shots = affirmative_few_shots

        elif self.stance == "negative":
            chat_history = negative_chat_history
            few_shots = negative_few_shots
        
        human_prompt = user_debate_prompt.format(topic=topic,
                                                opinion=opinion,
                                                oppose_opinion=oppose_opinion,
                                                last_oppose_speech_content=opponent_claim,
                                                search_results_content=search_results_content,
                                                chat_history = chat_history,
                                                few_shots = few_shots 
                                                )

        messages = [
            SystemMessage(content=user_debate_system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        result = await self.async_call_llm(messages)
        return result

    def get_latest_speech(self, history: list[DebateSpeech]):
        """
        get the latest speech from history
        """
        if len(history) == 0:
            return None
        return history[-1]

    def set_workspace(self, workspace):
        self.workspace = workspace
