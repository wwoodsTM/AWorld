# coding: utf-8
import os
import json
from typing import Dict, Any, List, Union

from aworld.agents.travel.prompts import search_prompt, search_sys_prompt
from aworld.config.common import Agents, Tools
from aworld.config.conf import AgentConfig
from aworld.core.agent.base import AgentFactory, BaseAgent
from aworld.core.common import Observation, ActionModel
from aworld.core.envs.tool_desc import get_tool_desc_by_name
from aworld.models.utils import tool_desc_transform
from aworld.core.envs.tool import ToolFactory
from aworld.config.conf import load_config


# Step1
@AgentFactory.register(name=Agents.SEARCH.value, desc="search agent")
class SearchAgent(BaseAgent):

    def __init__(self, conf: AgentConfig, **kwargs):
        super(SearchAgent, self).__init__(conf, **kwargs)
        # Step 2
        # Also can add other agent as tools (optional, it can be ignored if the interacting agent is deterministic.),
        # we only use search api tool for example.
        self.tool_desc = tool_desc_transform({Tools.SEARCH_API.value: get_tool_desc_by_name(Tools.SEARCH_API.value)})

    def policy(self, observation: Observation, info: Dict[str, Any] = {}, **kwargs) -> Union[
        List[ActionModel], None]:
        if observation.action_result is not None and len(observation.action_result) != 0 and observation.action_result[
            0].is_done:
            self._finished = True

            return [ActionModel(tool_name="[done]", policy_info=observation.content)]

        messages = [{'role': 'system', 'content': search_sys_prompt},
                    {'role': 'user',
                     'content': search_prompt.format(task=observation.content, tool_desc=self.tool_desc)}]

        llm_result = self.llm.invoke(
            input=messages,
        )
        tool_calls = llm_result.content
        print(tool_calls)
        return self._result(tool_calls)

    def _result(self, data):

        data = json.loads(data.replace("```json", "").replace("```", ""))
        actions = data.get("action", [])
        parsed_results = []

        for action in actions:
            for key, value in action.items():
                if "__" in key:
                    tool_name, action_name = key.split("__", 1)

                params = value
                parsed_results.append(ActionModel(tool_name=tool_name,
                                                  action_name=action_name,
                                                  params=params))
        return parsed_results


if __name__ == '__main__':
    agentConfig = AgentConfig(
        llm_provider="chatopenai",
        llm_model_name="gpt-4o",
        llm_base_url="http://localhost:5000",
        llm_api_key="dummy-key",
    )
    os.environ["GOOGLE_API_KEY"] = "AIzaSyAqNaFl2Ly-sBiLXcxA68ZLWwLF0Yc99V0"
    os.environ["GOOGLE_ENGINE_ID"] = "77bfce5ddc990489c"

    searchagent = SearchAgent(agentConfig)

    goal = "Best places in Japan for kendo, tea ceremony, and Zen meditation near Kyoto, Nara, or Kobe"
    observation = Observation(content=goal)
    while True:
        policy = searchagent.policy(observation=observation)

        print(policy)

        if policy[0].tool_name == '[done]':
            break

        tool = ToolFactory(policy[0].tool_name, conf=load_config(f"{policy[0].tool_name}.yaml"))

        observation, reward, terminated, _, info = tool.step(policy)

        print(observation)

    print(policy[0].policy_info)
