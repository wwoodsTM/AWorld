import uuid
import datetime
from dataclasses import Field
from typing import Union, Dict, Any, List

from pydantic import BaseModel

from aworld.agents.llm_agent import Agent
from aworld.config import ConfigDict, AgentConfig
from aworld.core.agent.base import AgentFactory
from aworld.core.common import Observation, ActionModel
from aworld.core.memory import MemoryItem
from aworld.models.llm import acall_llm_model


prompt = """Conduct targeted aworld_search tools to gather the most recent, credible information on "{research_topic}" and synthesize it into a verifiable text artifact.

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

    def _messages_transform(self,
                           content: str,
                           image_urls: List[str] = None,
                           sys_prompt: str = None,
                           agent_prompt: str = None,
                           **kwargs):
        """Transform the original content to LLM messages of native format.

        Args:
            content: User content.
            image_urls: List of images encoded using base64.
            sys_prompt: Agent system prompt.
            max_step: The maximum list length obtained from memory.
        Returns:
            Message list for LLM.
        """
        # def _safe_template_substitute(template_str: str, **kwargs) -> str:
        #     """
        #     使用 string.Template 进行安全的模板替换
        #     """
        #     try:
        #         # 将 {variable} 格式转换为 $variable 格式
        #         template_str = template_str.replace('{task}', '$task')
        #         template = Template(template_str)
        #         return template.safe_substitute(**kwargs)
        #     except Exception as e:
        #         print(f"模板替换失败: {e}")
        #         return template_str

        messages = []
        if sys_prompt:
            if '$task' in sys_prompt:
                sys_prompt = Template(sys_prompt).safe_substitute(task=content)
                # sys_prompt = sys_prompt.format(task=content)

            messages.append({'role': 'system', 'content': sys_prompt if not self.use_tools_in_prompt else sys_prompt.format(
                tool_list=self.tools)})

        histories = self.memory.get_last_n(self.history_messages)
        user_content = content
        if not histories and  agent_prompt and '{task}' in agent_prompt:
            user_content = agent_prompt.format(task=content)

        cur_msg = {'role': 'user', 'content': user_content}
        # query from memory,
        # histories = self.memory.get_last_n(self.history_messages, filter={"session_id": self.context.session_id})

        if histories:
            # default use the first tool call
            for history in histories:
                if not self.use_tools_in_prompt and "tool_calls" in history.metadata and history.metadata['tool_calls']:
                    messages.append({'role': history.metadata['role'], 'content': history.content,
                                     'tool_calls': [history.metadata["tool_calls"][0]]})
                else:
                    messages.append({'role': history.metadata['role'], 'content': history.content,
                                     "tool_call_id": history.metadata.get("tool_call_id")})

            if not self.use_tools_in_prompt and "tool_calls" in histories[-1].metadata and histories[-1].metadata[
                'tool_calls']:
                tool_id = histories[-1].metadata["tool_calls"][0].id
                if tool_id:
                    cur_msg['role'] = 'tool'
                    cur_msg['tool_call_id'] = tool_id
            if self.use_tools_in_prompt and "is_use_tool_prompt" in histories[-1].metadata and "tool_calls" in \
                    histories[-1].metadata and agent_prompt:
                cur_msg['content'] = agent_prompt.format(action_list=histories[-1].metadata["tool_calls"],
                                                         result=content)

        if image_urls:
            urls = [{'type': 'text', 'text': content}]
            for image_url in image_urls:
                urls.append({'type': 'image_url', 'image_url': {"url": image_url}})

            cur_msg['content'] = urls
        messages.append(cur_msg)
        return messages

    async def async_policy(self, observation: Observation, info: Dict[str, Any] = {}, **kwargs) -> List[ActionModel]:
        print("receive from plan_agent:", observation.content)
        print("mcp config", self.mcp_config)
        search_result = []

        # TODO: 并行提交任务处理, 先用for循环
        for research_topic in observation.content:

            if hasattr(observation, 'context') and observation.context:
                self.task_histories = observation.context
            self._finished = False
            await self.async_desc_transform()
            images = observation.images if self.conf.use_vision else None
            if self.conf.use_vision and not images and observation.image:
                images = [observation.image]

            messages = [{
                "role": "user",  # 可以是 "system", "user", "assistant", "tool"
                "content": prompt.format(
                    current_date=datetime.datetime.now().strftime("%Y-%m-%d"),
                    research_topic=research_topic
                )
            }]

            self._log_messages(messages)
            self.memory.add(MemoryItem(
                content=messages[-1]['content'],
                metadata={
                    "role": messages[-1]['role'],
                    "agent_name": self.name(),
                    "tool_call_id": messages[-1].get("tool_call_id")
                }
            ))

            self.context.context_info['web_search_nums'] += 1

            # 2.call llm and tools
            llm_response = await self.llm_and_tool_execution(
                observation=observation,
                messages=messages
            )

            print("llm_response", llm_response)


        # 收集和汇总结果然后给 reasoning_loop_agent
        return [ActionModel(
            agent_name=self.name(),
            tool_name="reasoning_loop_agent",
            policy_info={
                "search_result": search_result,
                "search_summary": None,
                "search_topics": observation.content
            })]