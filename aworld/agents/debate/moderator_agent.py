from abc import ABC
from datetime import datetime
from typing import Dict, Any, Union, List

from aworld.agents.debate.prompts import user_assignment_system_prompt
from aworld.agents.debate.stream_output_agent import StreamOutputAgent
from aworld.config import AgentConfig
from aworld.core.common import Observation, ActionModel
from aworld.output import MessageOutput


def truncate_content(raw_content, char_limit):
    if raw_content is None:
        raw_content = ''
    if len(raw_content) > char_limit:
        raw_content = raw_content[:char_limit] + "... [truncated]"
    return raw_content


class ModeratorAgent(StreamOutputAgent, ABC):
    stance: str = "moderator"

    def __init__(self, conf: AgentConfig, **kwargs
                 ):
        super().__init__(conf)

    async def async_policy(self, observation: Observation, info: Dict[str, Any] = {}, **kwargs) -> Union[
        List[ActionModel], None]:
        ## step 1: params
        topic = observation.content


        ## step2: gen opinions
        output = await self.gen_opinions(topic)


        action = ActionModel(
            policy_info=output
        )

        return [action]

    async def gen_opinions(self, topic) -> MessageOutput:

        current_time = datetime.now().strftime("%Y-%m-%d-%H")
        human_prompt = self.agent_prompt.format(topic=topic,
                                                     current_time=current_time,
                                                     )

        messages = [
            {"role": "system", "content": user_assignment_system_prompt},
            {"role": "user", "content": human_prompt}
        ]

        output = await self.async_call_llm(messages, json_parse=True)

        return output