from aworld.core.task import GeneralTaskfrom aworld.core.common import Observationfrom aworld.config.conf import AgentConfig

# AI Agents

Intelligent agents that control devices or tools in env using AI models or policy.

Most of the time, we directly use existing tools to build different types of agents that use LLM, 
using frameworks makes it easy to write various agents.

we provide a simple example for writing an agent:
```python
import copy
import json
from typing import Dict, Any, List, Union

from aworld.config.conf import AgentConfig
from aworld.core.agent.base import AgentFactory, BaseAgent
from aworld.core.common import Observation, ActionModel, Tools
from aworld.core.envs.tool_desc import get_tool_desc_by_name

from aworld.models.utils import tool_desc_transform


sys_prompt = "You are a helpful agent."
prompt = """
your prompt description

Here are the task: {task}

"""
# Detailed steps for building an agent:
# 1. Register your agent to agent factory, and inherit `BaseAgent`
# 2. Build tools with the description of their actions in __init__, such as variable `self.tool_desc` represents.
# 3. Implement the `name` method as a name identifier for the agent
# 4. Build roles messages for LLM input, variable `messages` represents in policy method.
# 5. Call LLM to obtain its response, such as self.llm.chat.completions.create, or self.llm.invoke(langchain).
# 6. Distinguish whether to use function/tool calls or not.
# 7. You can use memory to improve performance during multiple rounds of interaction.
    
# Step1
@AgentFactory.register(name="your_agent_name", desc="agent description")
class YourAgent(BaseAgent):
    
    def __init__(self, conf: AgentConfig, **kwargs):
        super(YourAgent, self).__init__(conf, **kwargs)
        # Step2
        self.tool_desc = tool_desc_transform({Tools.SEARCH_API.value: get_tool_desc_by_name(Tools.SEARCH_API.value)})

    # Step3
    def name(self) -> str:
        return "your_agent_name"

    def policy(self, observation: Observation, info: Dict[str, Any] = {}, **kwargs) -> Union[
        List[ActionModel], None]:
        
        # Step4
        # messages = [ChatMessage(role='system', content=sys_prompt),
        #             ChatMessage(role='user', content=prompt.format(task=observation.content))]
        # or
        messages = [{'role': 'system', 'content': sys_prompt},
                    {'role': 'user', 'content': prompt.format(task=observation.content)}]
        
        # Step7.1 (use memory)
        histories = self._history_messages()
        if histories:
            histories.insert(0, messages[0])
            histories.append(messages[1])
            messages = histories
        
        # Step5
        llm_result = self.llm.chat.completions.create(
            messages=messages,
            model=self.model_name,
            **{'temperature': 0, 'tools': self.tool_desc},
        )
        content = llm_result.choices[0].message.content
        tool_calls = llm_result.choices[0].message.tool_calls
        
        # Step7.2 (use memory)
        self.trajectory.append((copy.deepcopy(observation), info, llm_result))
        
        # Step6
        # whether done
        self._finished = is_done?
        if tool_calls:
            return  self._result(tool_calls)
        else:
            # use variable `content` to do something if there is no need to call the tools
            pass
        
    def _result(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_action_name: str = tool_call.function.name
            if not tool_action_name:
                continue
            tool_name = tool_action_name.split("__")[0]
            action_name = tool_action_name.split("__")[1]
            params = json.loads(tool_call.function.arguments)
            results.append(ActionModel(tool_name=tool_name, action_name=action_name, params=params))
        return results

    def _history_messages(self):
        history = []
        for traj in self.trajectory:
            history.append(traj[0].content)
            if traj[-1].choices[0].message.tool_calls is not None:
                history.append(
                    {'role': 'assistant', 'content': '', 'tool_calls': traj[-1].choices[0].message.tool_calls})
            else:
                history.append({'role': 'assistant', 'content': traj[-1].choices[0].message.content})
        return history
```

It can also quickly develop multi-agent based on the framework.

On the basis of the above agent(YourAgent), we provide a multi-agent example:

```python
import copy
import json
from typing import Dict, Any, List, Union

from aworld.agents import agent_desc
from aworld.config.conf import AgentConfig
from aworld.core.agent.base import AgentFactory, BaseAgent
from aworld.core.common import Observation, ActionModel
from aworld.core.envs.tool_desc import get_tool_desc
from aworld.models.utils import agent_desc_transform, tool_desc_transform

sys_prompt = "You are a helpful agent, you must start to instruct me to solve the task step-by-step.."
prompt = """
Now, here is the overall task: <task>{task}</task>. Never forget the task!
"""


# The agent can be considered as a plan agent, the process is almost the same as YourAgent, we define as OtherAgent.

# Step 1
@AgentFactory.register(name="other_agent_name", desc="agent description")
class OtherAgent(BaseAgent):

    def __init__(self, conf: AgentConfig, **kwargs):
        super(OtherAgent, self).__init__(conf, **kwargs)
        # Step 2 (Optional, it can be ignored if the interacting agent is deterministic.)
        # agent as a tool
        self.agent_desc = agent_desc_transform(agent_desc, agents=['your_agent_name'])
        # also can add other tools, optional
        # self.agent_desc.extend(tool_desc_transform(get_tool_desc()))

    # Step 3
    def name(self) -> str:
        return "your_agent_name"

    def policy(self, observation: Observation, info: Dict[str, Any] = {}, **kwargs) -> Union[
        List[ActionModel], None]:

        # Step 4
        # messages = [ChatMessage(role='system', content=sys_prompt),
        #             ChatMessage(role='user', content=prompt.format(task=observation.content))]
        # or
        messages = [{'role': 'system', 'content': sys_prompt},
                    {'role': 'user', 'content': prompt.format(task=observation.content)}]

        # Step 7.1 (use memory)
        histories = self._history_messages()
        if histories:
            histories.insert(0, messages[0])
            histories.append(messages[1])
            messages = histories

        # Step 5
        llm_result = self.llm.chat.completions.create(
            messages=messages,
            model=self.model_name,
            # can be ignored if the agent_desc is None
            **{'temperature': 0, 'tools': self.agent_desc},
        )

        # Step 7.2 (use memory)
        self.trajectory.append((copy.deepcopy(observation), info, llm_result))

        content = llm_result.choices[0].message.content
        tool_calls = llm_result.choices[0].message.tool_calls
        self._finished = is_done?
        # Step 6, use tool call or content depends on the result and user
        if content:
            # if the interacting agent is deterministic
            return [ActionModel(agent_name='your_agent_name', policy_info=content)]
        elif tool_calls:
            # agent as a tool
            return self._result(tool_calls)

    def _result(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_action_name: str = tool_call.function.name
            if not tool_action_name:
                continue
            tool_name = tool_action_name.split("__")[0]
            action_name = tool_action_name.split("__")[1]
            params = json.loads(tool_call.function.arguments)
            results.append(ActionModel(agent_name=tool_name, action_name=action_name, params=params))
        return results

    def _history_messages(self):
        history = []
        for traj in self.trajectory:
            history.append(traj[0].content)
            if traj[-1].choices[0].message.tool_calls is not None:
                history.append(
                    {'role': 'assistant', 'content': '', 'tool_calls': traj[-1].choices[0].message.tool_calls})
            else:
                history.append({'role': 'assistant', 'content': traj[-1].choices[0].message.content})
        return history
```


You can run single-agent or multi-agent through Swarm.

```python

task = "your task"
your_agent = YourAgent(AgentConfig())
other_agent = OtherAgent(AgentConfig())
# build topology graph, the correct order is necessary
swarm = Swarm((other_agent, your_agent))
# single-agent
# swarm = Swarm( your_agent)
t = GeneralTask(input=task, swarm=swarm)
t.run()
```