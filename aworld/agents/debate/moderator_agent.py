import json
import logging
from abc import ABC
from datetime import datetime
from typing import Dict, Any, Union, List, Literal

from langchain_core.messages import SystemMessage, HumanMessage

from aworld.agents.debate.prompts import user_assignment_system_prompt
from aworld.config import AgentConfig
from aworld.core.agent.base import Agent
from aworld.core.common import Observation, ActionModel
from aworld.output import CommonOutput


def truncate_content(raw_content, char_limit):
    if raw_content is None:
        raw_content = ''
    if len(raw_content) > char_limit:
        raw_content = raw_content[:char_limit] + "... [truncated]"
    return raw_content


class ModeratorAgent(Agent, ABC):
    stance: Literal["affirmative", "negative"]

    def __init__(self, conf: AgentConfig, **kwargs
                 ):
        super().__init__(conf)

    async def async_policy(self, observation: Observation, info: Dict[str, Any] = {}, **kwargs) -> Union[
        List[ActionModel], None]:
        ## step 1: params
        topic = observation.content


        ## step2: gen opinions
        opinions = await self.gen_opinions(topic)
        logging.info(f"gen opinions = {opinions}")

        if isinstance(opinions, str):
            opinions = json.loads(opinions)


        action = ActionModel(
            policy_info=CommonOutput(data=opinions)
        )

        return [action]

    async def gen_opinions(self, topic):

        current_time = datetime.now().strftime("%Y-%m-%d-%H")
        human_prompt = self.agent_prompt.format(topic=topic,
                                                     current_time=current_time,
                                                     )

        messages = [
            SystemMessage(content=user_assignment_system_prompt),
            HumanMessage(content=human_prompt)
        ]

        result = await self.async_call_llm(messages, json_parse= True)

        return result

    async def async_call_llm(self, messages, json_parse = False):
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
            if llm_result.__contains__("```json") and json_parse:
                llm_result = llm_result.replace("```json", "").replace("```", "")
                return llm_think, json.loads(llm_result)

            return llm_think, llm_result

        result = await self.llm.ainvoke(input=messages)
        llm_think, llm_result = _resolve_think(result.content)
        return llm_result
