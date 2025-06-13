import uuid
import datetime
from dataclasses import Field
from typing import Union, Dict, Any, List

from pydantic import BaseModel

from aworld.agents.llm_agent import Agent
from aworld.config import ConfigDict, AgentConfig
from aworld.core.agent.base import AgentFactory
from aworld.core.common import Observation, ActionModel
from aworld.models.llm import acall_llm_model


prompt = """Conduct targeted Google Searches to gather the most recent, credible information on "{research_topic}" and synthesize it into a verifiable text artifact.

Instructions:
- Query should ensure that the most current information is gathered. The current date is {current_date}.
- Conduct multiple, diverse searches to gather comprehensive information.
- Consolidate key findings while meticulously tracking the source(s) for each specific piece of information.
- The output should be a well-written summary or report based on your search findings. 
- Only include the information found in the search results, don't make up any information.

Research Topic:
{research_topic}
"""

@AgentFactory.register(name='web_search_agent', desc="web_search_agent")
class WebSearchAgent(Agent):
    def __init__(self, conf: Union[Dict[str, Any], ConfigDict, AgentConfig], **kwargs):
        super().__init__(conf, **kwargs)
        # 初始化一个 web_search_nums 变量来存储 search 的次数
        self.context.context_info['web_search_nums'] = 0


    async def async_policy(self, observation: Observation, info: Dict[str, Any] = {}, **kwargs) -> List[ActionModel]:
        print("receive from plan_agent:", observation.content)
        search_result = []

        # TODO: 并行提交任务处理, 先用for循环
        for research_topic in observation.content:
            self.context.context_info['web_search_nums'] += 1
            # 1.构造 message
            messages = [{
                "role": "user",  # 可以是 "system", "user", "assistant", "tool"
                "content": prompt.format(
                    current_date=datetime.datetime.now().strftime("%Y-%m-%d"),
                    research_topic=research_topic
                )
            }]

            # 2.TODO call llm and tools
            llm_response = await acall_llm_model(
                self.llm,
                messages=messages,
                model=self.model_name,
                temperature=self.conf.llm_config.llm_temperature,
                tools=self.tools if self.use_tools_in_prompt and self.tools else None
            )

        # 收集和汇总结果然后给 reasoning_loop_agent
        return [ActionModel(
            agent_name=self.name(),
            tool_name="reasoning_loop_agent",
            policy_info={
                "search_result": search_result,
                "search_summary": None,
                "search_topics": observation.content
            })]