# coding: utf-8
# Copyright (c) 2025 inclusionAI.

import abc
from typing import Generic, TypeVar, Dict, Any, List, Tuple, Union

from aworld.agents.common import LlmResult
from aworld.config.conf import AgentConfig
from aworld.core.common import Observation, ActionModel
from aworld.core.factory import Factory

INPUT = TypeVar('INPUT')
OUTPUT = TypeVar('OUTPUT')


class Agent(Generic[INPUT, OUTPUT]):
    __metaclass__ = abc.ABCMeta

    def __init__(self, conf: AgentConfig, **kwargs):
        self.conf = conf
        self.handoffs = []
        self.trajectory: List[Tuple[INPUT, Dict[str, Any], LlmResult]] = []

        for k, v in kwargs.items():
            setattr(self, k, v)

    @abc.abstractmethod
    def name(self) -> str:
        """"""

    @abc.abstractmethod
    def policy(self, observation: INPUT, info: Dict[str, Any] = None, **kwargs) -> OUTPUT:
        """"""

    @abc.abstractmethod
    async def async_policy(self, observation: INPUT, info: Dict[str, Any] = None, **kwargs) -> OUTPUT:
        """"""

    def reset(self, options: Dict[str, Any]):
        """Clean agent instance state and reset."""

    @abc.abstractmethod
    async def async_reset(self):
        """"""

    @property
    def finished(self) -> bool:
        """Agent finished the thing, default is True."""
        return False


class BaseAgent(Agent[Observation, Union[Observation, List[ActionModel]]]):
    """"""

    @abc.abstractmethod
    def policy(self, observation: Observation, info: Dict[str, Any] = {}, **kwargs) -> Union[
        List[ActionModel], None]:
        """"""

    @abc.abstractmethod
    async def async_policy(self, observation: Observation, info: Dict[str, Any] = {}, **kwargs) -> Union[
        List[ActionModel], None]:
        """"""


class AgentManager(Factory):
    def __call__(self, name: str = None, *args, **kwargs):
        if name is None:
            return self

        if 'conf' not in kwargs:
            if not args:
                raise ValueError("params `conf` must in args or kwargs!")
            else:
                conf = args[0]
        else:
            conf = kwargs.pop('conf')
            if conf is None:
                raise ValueError("params `conf` must in args or kwargs!")

        if name in self._cls:
            agent = self._cls[name](conf=conf, **kwargs)
        else:
            raise ValueError(f"Can not find {name} agent!")
        return agent


AgentFactory = AgentManager("agent_type")
