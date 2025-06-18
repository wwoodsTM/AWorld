import os
import uuid
import datetime
from typing import Dict, Any, List

from aworld.config.conf import AgentConfig
from aworld.agents.llm_agent import Agent
from aworld.core.agent.base import AgentFactory
from aworld.core.common import Observation, ActionModel
from aworld.core.task import Task
from aworld.config import ModelConfig, TaskConfig
from aworldspace.prompt.deepresearch_prompt import *
from aworld.runner import Runners
from aworld.models.llm import acall_llm_model


prompt = """Generate a high-quality answer to the user's question based on the provided summaries.

Instructions:
- The current date is {current_date}.
- You are the final step of a multi-step research process, don't mention that you are the final step. 
- You have access to all the information gathered from the previous steps.
- You have access to the user's question.
- Generate a high-quality answer to the user's question based on the provided summaries and the user's question.
- you MUST include all the citations from the summaries in the answer correctly.
- 输出中文结果
User Context:
- {research_topic}

Summaries:
{summaries}"""

# @AgentFactory.register(name='reporting_agent', desc="reporting_agent")
class ReportingAgent(Agent):

    async def async_policy(self, observation: Observation, info: Dict[str, Any] = {}, **kwargs) -> List[ActionModel]:
        print("[reporting_agent]receive from reasoning_loop_agent:", observation.content)

        # 1.构造 message
        messages = [{
            "role": "user",
            "content": prompt.format(
                current_date=datetime.datetime.now().strftime("%Y-%m-%d"),
                research_topic=observation.content,
                summaries=3,
            )
        }]

        # 2.call llm
        llm_response = await acall_llm_model(
            self.llm,
            messages=messages,
            model=self.model_name,
            temperature=self.conf.llm_config.llm_temperature,
            tools=self.tools if self.use_tools_in_prompt and self.tools else None
        )
        content = llm_response.get_message()['content']


        return [ActionModel(
            agent_name=self.name(),
            policy_info=content)]