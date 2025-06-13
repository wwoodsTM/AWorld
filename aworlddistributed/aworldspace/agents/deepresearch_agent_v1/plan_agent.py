import os
import uuid
from typing import Dict, Any, List
import json
import re
import datetime

from aworld.config.conf import AgentConfig
from aworld.agents.llm_agent import Agent
from aworld.core.agent.base import AgentFactory
from aworld.core.common import Observation, ActionModel
from aworld.core.task import Task
from aworld.config import ModelConfig, TaskConfig

from aworldspace.agents.deepresearch_agent_v1.tools_and_schemas import parse_json_to_model, SearchQueryList
from aworldspace.prompt.deepresearch_prompt import *
from aworld.runner import Runners

from aworld.models.llm import acall_llm_model

prompt = """Your goal is to generate sophisticated and diverse web search queries. These queries are intended for an advanced automated web research tool capable of analyzing complex results, following links, and synthesizing information.

Instructions:
- Always prefer a single search query, only add another query if the original question requests multiple aspects or elements and one query is not enough.
- Each query should focus on one specific aspect of the original question.
- Don't produce more than {number_queries} queries.
- Queries should be diverse, if the topic is broad, generate more than 1 query.
- Don't generate multiple similar queries, 1 is enough.
- Query should ensure that the most current information is gathered. The current date is {current_date}.

Format: 
- Format your response as a JSON object with ALL three of these exact keys:
   - "rationale": Brief explanation of why these queries are relevant
   - "query": A list of search queries

Example:

Topic: What revenue grew more last year apple stock or the number of people buying an iphone
```json
{{
    "rationale": "To answer this comparative growth question accurately, we need specific data points on Apple's stock performance and iPhone sales metrics. These queries target the precise financial information needed: company revenue trends, product-specific unit sales figures, and stock price movement over the same fiscal period for direct comparison.",
    "query": ["Apple total revenue growth fiscal year 2024", "iPhone unit sales growth fiscal year 2024", "Apple stock price growth fiscal year 2024"],
}}
```

Context: {research_topic}"""


@AgentFactory.register(name='plan_agent', desc="plan_agent")
class PlanAgent(Agent):

    def _parse_query_from_llm_response(self, content: str) -> List[str]:
        """
        从LLM响应中解析出query列表
        
        Args:
            content: LLM返回的字符串内容，包含JSON格式的数据
            
        Returns:
            List[str]: 解析出的query列表
        """
        try:
            # 方法1: 直接解析JSON
            try:
                data = json.loads(content)
                if isinstance(data, dict) and 'query' in data:
                    return data['query']
            except json.JSONDecodeError:
                pass

            # 方法2: 提取```json```包围的内容
            json_pattern = r'```json\s*\n(.*?)\n\s*```'
            match = re.search(json_pattern, content, re.DOTALL)

            if match:
                json_str = match.group(1)
                data = json.loads(json_str)
                if isinstance(data, dict) and 'query' in data:
                    return data['query']

            # 方法3: 查找任何JSON格式的内容
            brace_count = 0
            start_idx = -1

            for i, char in enumerate(content):
                if char == '{':
                    if brace_count == 0:
                        start_idx = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start_idx != -1:
                        try:
                            json_str = content[start_idx:i + 1]
                            data = json.loads(json_str)
                            if isinstance(data, dict) and 'query' in data:
                                return data['query']
                        except json.JSONDecodeError:
                            continue

            return []

        except Exception as e:
            print(f"解析LLM响应错误: {e}")
            return []

    async def async_policy(self, observation: Observation, info: Dict[str, Any] = {}, **kwargs) -> List[ActionModel]:
        # 1.构造 message
        messages = [{
            "role": "user",  # 可以是 "system", "user", "assistant", "tool"
            "content": prompt.format(
                current_date=datetime.datetime.now().strftime("%Y-%m-%d"),
                research_topic=observation.content,
                number_queries=3,
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

        # 3. 解析LLM响应，提取query列表
        content = llm_response.get_message()['content']
        # query_list = self._parse_query_from_llm_response(content)
        query_list = parse_json_to_model(content, SearchQueryList)
        print(f"query_list: {query_list}")

        # 4. 转交给 web_search_agent 处理
        return [ActionModel(
            agent_name=self.name(),
            tool_name="web_search_agent",
            policy_info=query_list)]
