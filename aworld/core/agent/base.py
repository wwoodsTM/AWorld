# coding: utf-8
# Copyright (c) 2025 inclusionAI.

import abc
import json
import uuid
from typing import Generic, TypeVar, Dict, Any, List, Tuple, Union

from openai import Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from aworld.core.agent.agent_desc import get_agent_desc
from aworld.core.envs.tool_desc import get_tool_desc
from aworld.logs.util import logger
from aworld.models.llm import get_llm_model
from pydantic import BaseModel

from aworld.config.conf import AgentConfig, load_config, ConfigDict
from aworld.core.common import Observation, ActionModel
from aworld.core.factory import Factory
from aworld.models.utils import tool_desc_transform, agent_desc_transform
from aworld.utils.name_transform import convert_to_snake

INPUT = TypeVar('INPUT')
OUTPUT = TypeVar('OUTPUT')


def is_agent_by_name(name: str) -> bool:
    return name in AgentFactory


class AgentStatus:
    # Init status
    START = 0
    # Agent is running for monitor or collection
    RUNNING = 1
    # Agent reject the task
    REJECT = 2
    # Agent is idle
    IDLE = 3
    # Agent meets exception
    ERROR = 4
    # End of one agent step
    DONE = 5
    # End of one task step
    FINISHED = 6


class Agent(Generic[INPUT, OUTPUT]):
    __metaclass__ = abc.ABCMeta

    def __init__(self, conf: Union[Dict[str, Any], ConfigDict, AgentConfig], **kwargs):
        self.conf = conf
        if isinstance(conf, ConfigDict):
            pass
        elif isinstance(conf, Dict):
            self.conf = ConfigDict(conf)
        elif isinstance(conf, AgentConfig):
            # To add flexibility
            self.conf = ConfigDict(conf.model_dump())
        else:
            logger.warning(f"Unknown conf type: {type(conf)}")

        self._name = self.conf.get("name", convert_to_snake(self.__class__.__name__))
        # Unique flag based agent name
        self.id = f"{self.name()}_{uuid.uuid1().hex[0:6]}"
        self.task = None
        # An agent can use the tool list
        self.tool_names: List[str] = kwargs.get("tool_names", [])
        # An agent can delegate tasks to other agent
        self.handoffs: List[str] = kwargs.get("agent_names", [])
        # Supported MCP server
        self.mcp_servers: List[str] = kwargs.get("mcp_servers", [])
        self.trajectory: List[Tuple[INPUT, Dict[str, Any], AgentResult]] = []
        # all tools that the agent can use. note: string name/id only
        self._tools = []
        self.state = AgentStatus.START
        self._finished = False

        for k, v in kwargs.items():
            setattr(self, k, v)

    def name(self) -> str:
        """Agent name that must be implemented in subclasses"""
        return self._name

    @abc.abstractmethod
    def policy(self, observation: INPUT, info: Dict[str, Any] = None, **kwargs) -> OUTPUT:
        """The strategy of an agent can be to decide which tools to use in the environment, or to delegate tasks to other agents.

        Args:
            observation: The state observed from tools in the environment.
            info: Extended information is used to assist the agent to decide a policy.
        """

    @abc.abstractmethod
    async def async_policy(self, observation: INPUT, info: Dict[str, Any] = None, **kwargs) -> OUTPUT:
        """The strategy of an agent can be to decide which tools to use in the environment, or to delegate tasks to other agents.

        Args:
            observation: The state observed from tools in the environment.
            info: Extended information is used to assist the agent to decide a policy.
        """

    def reset(self, options: Dict[str, Any]):
        """Clean agent instance state and reset."""
        if options is None:
            options = {}
        self.task = options.get("task")
        self.tool_names = options.get("tool_names")
        self.handoffs = options.get("agent_names", [])
        self.trajectory = []
        self._finished = False

    async def async_reset(self, options: Dict[str, Any]):
        """Clean agent instance state and reset."""
        self.task = options.get("task")

    @property
    def finished(self) -> bool:
        """Agent finished the thing, default is True."""
        return self._finished


class BaseAgent(Agent[Observation, Union[Observation, List[ActionModel]]]):
    """Basic agent for unified protocol within the framework."""

    def __init__(self, conf: Union[Dict[str, Any], ConfigDict, AgentConfig], **kwargs):
        super(BaseAgent, self).__init__(conf, **kwargs)
        self.model_name = conf.llm_model_name
        self._llm = None
        self.memory = []

    @property
    def llm(self):
        # lazy
        if self._llm is None:
            conf = self.conf.llm_config if self.conf.llm_config.llm_provider else self.conf
            self._llm = get_llm_model(conf)
        return self._llm

    def desc_transform(self):
        """Transform of descriptions of supported tools, agents, and MCP servers in the framework to support function calls of LLM."""

        # Stateless tool
        self.tools = tool_desc_transform(get_tool_desc(),
                                         tools=self.tool_names if self.tool_names else [])
        # Agents as tool
        agents_desc = agent_desc_transform(get_agent_desc(),
                                           agents=self.handoffs if self.handoffs else [])
        self.tools.extend(agents_desc)
        # MCP servers are tool

    def messages_transform(self,
                           content: str,
                           image_urls: List[str] = None,
                           sys_prompt: str = None,
                           max_step: int = 100,
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
        messages = []
        if sys_prompt:
            messages.append({'role': 'system', 'content': sys_prompt})

        cur_msg = {'role': 'user', 'content': content}
        # query from memory, TODO: memory.query()
        histories = self.memory[-max_step:]
        if histories:
            for history in histories:
                if history.tool_calls:
                    messages.append({'role': 'assistant', 'content': '', 'tool_calls': history.tool_calls})
                else:
                    messages.append({'role': 'assistant', 'content': history.content})

            if histories[-1].tool_calls:
                tool_id = histories[-1].tool_calls[0].id
                if tool_id:
                    cur_msg['tool_call_id'] = tool_id

        if image_urls:
            urls = [{'type': 'text', 'text': content}]
            for image_url in image_urls:
                urls.append({'type': 'image_url', 'image_url': {"url": image_url}})

            cur_msg['content'] = urls
        messages.append(cur_msg)
        return messages

    def response_parse(self, resp: ChatCompletion | Stream[ChatCompletionChunk]):
        """Default parse response by LLM."""
        results = []
        if not resp:
            return []

        content = resp.choices[0].message.content
        if resp.choices[0].message.tool_calls:
            for tool_call in resp:
                full_name: str = tool_call.function.name
                if not full_name:
                    logger.warning("tool call response no tool name.")
                    continue

                params = json.loads(tool_call.function.arguments)
                # format in framework
                names = full_name.split("__")
                tool_name = names[0]
                if is_agent_by_name(tool_name):
                    results.append(ActionModel(agent_name=tool_name, params=params, policy_info=content))
                else:
                    action_name = names[1] if len(names) > 1 else None
                    results.append(ActionModel(tool_name=tool_name,
                                               action_name=action_name,
                                               params=params,
                                               policy_info=content))
        return results

    @abc.abstractmethod
    def policy(self, observation: Observation, info: Dict[str, Any] = {}, **kwargs) -> Union[
        List[ActionModel], None]:
        """The strategy of an agent can be to decide which tools to use in the environment, or to delegate tasks to other agents.

        Args:
            observation: The state observed from tools in the environment.
            info: Extended information is used to assist the agent to decide a policy.

        Returns:
            ActionModel sequence from agent policy
        """

    @abc.abstractmethod
    async def async_policy(self, observation: Observation, info: Dict[str, Any] = {}, **kwargs) -> Union[
        List[ActionModel], None]:
        """The strategy of an agent can be to decide which tools to use in the environment, or to delegate tasks to other agents.

        Args:
            observation: The state observed from tools in the environment.
            info: Extended information is used to assist the agent to decide a policy.

        Returns:
            ActionModel sequence from agent policy
        """


class AgentManager(Factory):
    def __init__(self, type_name: str = None):
        super(AgentManager, self).__init__(type_name)
        self._agent_conf = {}

    def __call__(self, name: str = None, *args, **kwargs):
        if name is None:
            return self

        conf = self._agent_conf.get(name)
        if not conf:
            logger.warning(f"{name} not find conf in tool factory")
            conf = dict()
        elif isinstance(conf, BaseModel):
            conf = conf.model_dump()

        user_conf = kwargs.pop('conf', None)
        if user_conf:
            if isinstance(user_conf, BaseModel):
                conf.update(user_conf.model_dump())
            elif isinstance(user_conf, dict):
                conf.update(user_conf)
            else:
                logger.warning(f"Unknown conf type: {type(user_conf)}, ignored!")

        conf['name'] = name
        conf = ConfigDict(conf)
        if name in self._cls:
            agent = self._cls[name](conf=conf, **kwargs)
        else:
            raise ValueError(f"Can not find {name} agent!")
        return agent

    def register(self, name: str, desc: str, conf_file_name: str = None, **kwargs):
        """Register a tool to tool factory.

        Args:
            name: Tool name
            desc: Tool description
            supported_action: Tool abilities
            conf_file_name: Default tool config
        """
        res = super(AgentManager, self).register(name, desc, **kwargs)
        conf_file_name = conf_file_name if conf_file_name else f"{name}.yaml"
        conf = load_config(conf_file_name, kwargs.get("dir"))
        if not conf:
            logger.warning(f"{conf_file_name} not find, will use default")
            # use general tool config
            conf = AgentConfig().model_dump()
        self._agent_conf[name] = conf
        return res


AgentFactory = AgentManager("agent_type")


class AgentResult(BaseModel):
    current_state: Any
    actions: List[ActionModel]
