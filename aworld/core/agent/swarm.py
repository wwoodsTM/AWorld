# coding: utf-8
# Copyright (c) 2025 inclusionAI.
import abc
from typing import Dict, List, Any, Callable

from aworld.core.agent.agent_desc import agent_handoffs_desc
from aworld.core.agent.base import AgentFactory
from aworld.agents.llm_agent import Agent
from aworld.core.common import ActionModel, Observation
from aworld.core.context.base import Context
from aworld.logs.util import logger
from aworld.utils.common import new_instance, convert_to_subclass

DETERMINACY = "Determinacy"
DYNAMIC = "Dynamicy"


class Swarm(object):
    """Multi-agent topology."""

    def __init__(self,
                 *args,  # agent
                 root_agent: Agent = None,
                 determinacy: bool = True,
                 max_steps: int = 0,
                 event_driven: bool = True,
                 builder_cls: str = None,
                 **kwargs):
        self._communicate_agent = root_agent
        if root_agent and root_agent not in args:
            self._topology: List[Agent] = [root_agent] + list(args)
        else:
            self._topology: List[Agent] = list(args)
        self.sequence = determinacy
        self.max_steps = max_steps
        self._cur_step = 0
        self.execute_type = DETERMINACY if determinacy else DYNAMIC
        self._event_driven = event_driven
        if builder_cls:
            self.builder = new_instance(builder_cls, self)
        else:
            self.builder = DeterminacyBuilder(self) if determinacy else DynamicyBuilder(self)
        for agent in self._topology:
            if isinstance(agent, Agent):
                agent = [agent]
            for a in agent:
                if a and a.event_driven:
                    self._event_driven = True
                    break
            if self._event_driven:
                break

        self.agents: Dict[str, Agent] = dict()
        self.ordered_agents: List[Agent] = []
        # global tools
        self.tools = []
        self.task = ''
        self.initialized = False
        self._finished = False

    def reset(self, content: Any, context: Context = None, tools: List[str] = []):
        """Resets the initial internal state, and init supported tools in agent in swarm.

        Args:
            tools: Tool names that all agents in the swarm can use.
        """
        # can use the tools in the agents in the swarm as a global
        if self.initialized:
            logger.warning(f"swarm {self} already init")
            return

        self.tools = tools
        self.task = content

        self.builder.build()
        if not self.agents:
            logger.warning("No valid agent in swarm.")
            return

        self.cur_agent = self.communicate_agent

        if context is None:
            context = Context.instance()

        for agent in self.agents.values():
            agent.event_driven = self.event_driven
            if agent.need_reset:
                agent.context = context
                agent.reset({"task": content,
                             "tool_names": agent.tool_names,
                             "agent_names": agent.handoffs,
                             "mcp_servers": agent.mcp_servers})
            # global tools
            agent.tool_names.extend(self.tools)

        self.cur_step = 1
        self.initialized = True

    def loop_agent(self,
                   agent: Agent,
                   max_run_times: int,
                   loop_point: str = None,
                   loop_point_finder: Callable[..., Any] = None,
                   stop_func: Callable[..., Any] = None):
        """Loop execution of the flow.

        Args:
            agent: The agent.
            max_run_times: Maximum number of loops.
            loop_point: Loop point of the desired execution.
            loop_point_finder: Strategy function for obtaining execution loop point.
            stop_func: Termination function.
        """
        from aworld.agents.loop_llm_agent import LoopableAgent

        if agent not in self.topology:
            raise RuntimeError(f"{agent.name()} not in swarm, agent instance {agent}.")

        loop_agent: LoopableAgent = convert_to_subclass(agent, LoopableAgent)
        # loop_agent: LoopableAgent = type(LoopableAgent)(agent)
        loop_agent.max_run_times = max_run_times
        loop_agent.loop_point = loop_point
        loop_agent.loop_point_finder = loop_point_finder
        loop_agent.stop_func = stop_func

        idx = self.topology.index(agent)
        self.topology[idx] = loop_agent

    def parallel_agent(self,
                       agent: Agent,
                       agents: List[Agent],
                       aggregate_func: Callable[..., Any] = None):
        """Parallel execution of agents.

        Args:
            agent: The agent.
            agents: Agents that require parallel execution.
            aggregate_func: Aggregate strategy function.
        """
        from aworld.agents.parallel_llm_agent import ParallelizableAgent

        if agent not in self.topology:
            raise RuntimeError(f"{agent.name()} not in swarm, agent instance {agent}.")
        for agent in agents:
            if agent not in self.topology:
                raise RuntimeError(f"{agent.name()} not in swarm, agent instance {agent}.")

        parallel_agent: ParallelizableAgent = convert_to_subclass(agent, ParallelizableAgent)
        parallel_agent.agents = agents
        parallel_agent.aggregate_func = aggregate_func

        idx = self.topology.index(agent)
        self.topology[idx] = parallel_agent

    def _check(self):
        if not self.initialized:
            self.reset('')

    def handoffs_desc(self, agent_name: str = None, use_all: bool = False):
        """Get agent description by name for handoffs.

        Args:
            agent_name: Agent unique name.
        Returns:
            Description of agent dict.
        """
        self._check()
        agent: Agent = self.agents.get(agent_name, None)
        return agent_handoffs_desc(agent, use_all)

    def action_to_observation(self, policy: List[ActionModel], observation: List[Observation], strategy: str = None):
        """Based on the strategy, transform the agent's policy into an observation, the case of the agent as a tool.

        Args:
            policy: Agent policy based some messages.
            observation: History of the current observable state in the environment.
            strategy: Transform strategy, default is None. enum?
        """
        self._check()

        if not policy:
            logger.warning("no agent policy, will return origin observation.")
            # get the latest one
            if not observation:
                raise RuntimeError("no observation and policy to transform in swarm, please check your params.")
            return observation[-1]

        if not strategy:
            # default use the first policy
            policy_info = policy[0].policy_info

            if not observation:
                res = Observation(content=policy_info)
            else:
                res = observation[-1]
                if res.content is None:
                    res.content = ''

                if policy_info:
                    res.content += policy_info
            return res
        else:
            logger.warning(f"{strategy} not supported now.")

    def supported_tools(self):
        """Tool names that can be used by all agents in Swarm."""
        self._check()
        return self.tools

    @property
    def topology(self):
        return self._topology

    @property
    def communicate_agent(self):
        return self._communicate_agent

    @communicate_agent.setter
    def communicate_agent(self, agent: Agent):
        self._communicate_agent = agent

    @property
    def event_driven(self):
        return self._event_driven

    @event_driven.setter
    def event_driven(self, event_driven):
        self._event_driven = event_driven

    @property
    def cur_step(self) -> int:
        return self._cur_step

    @cur_step.setter
    def cur_step(self, step):
        self._cur_step = step

    @property
    def finished(self) -> bool:
        """Need all agents in a finished state."""
        self._check()
        if not self._finished:
            self._finished = all([agent.finished for _, agent in self.agents.items()])
        return self._finished

    @finished.setter
    def finished(self, finished):
        self._finished = finished


class _Graph:
    def save(self, filepath: str):
        pass

    def load(self, filepath: str):
        pass


class TopologyBuilder:
    """Multi-agent topology base builder."""
    __metaclass__ = abc.ABCMeta

    def __init__(self, swarm: Swarm):
        self.swarm = swarm
        self.nodes = []
        self.edges: Dict[str, List[str]] = {}

    @abc.abstractmethod
    def build(self):
        """Build the agents' execution graph."""

    @staticmethod
    def register_agent(agent: Agent):
        if agent.name() not in AgentFactory:
            AgentFactory._cls[agent.name()] = agent.__class__
            AgentFactory._desc[agent.name()] = agent.desc()
            AgentFactory._agent_conf[agent.name()] = agent.conf
            AgentFactory._agent_instance[agent.name()] = agent
        else:
            if agent.name() not in AgentFactory._agent_instance:
                AgentFactory._agent_instance[agent.name()] = agent
            if agent.desc():
                AgentFactory._desc[agent.name()] = agent.desc()


class WorkflowBuilder(TopologyBuilder):
    """Workflow mechanism, workflow is a deterministic process orchestration where each node must execute.

    Only handle agent pairs based on the workflow mechanism, examples:
    >>> agent1 = Agent(name='agent1'); agent2 = Agent(name='agent2'); agent3 = Agent(name='agent3')
    >>> Swarm((agent1, agent2, agent3)
    """

    def build(self):
        valid_agents = []
        for agent in self.swarm.topology:
            if isinstance(agent, (list, tuple)):
                raise RuntimeError(f"agent in {agent} is not a agent, please check it.")

            if not isinstance(agent, Agent):
                raise RuntimeError(f"agent in {agent} is not a base agent instance, please check it.")

            valid_agents.append(agent)

        if not valid_agents:
            raise RuntimeError(f"no valid agent to build execution graph.")

        if self.swarm.communicate_agent is None:
            self.swarm.communicate_agent = valid_agents[0]

        for agent in valid_agents:
            self.swarm.ordered_agents.append(agent)

            if agent.name() not in self.swarm.agents:
                self.swarm.agents[agent.name()] = agent

            TopologyBuilder.register_agent(agent)


class GraphBuilder(TopologyBuilder):
    """Handoffs mechanism.

    Only handle agent pairs based on the handoffs mechanism, examples:
    >>> agent1 = Agent(name='agent1'); agent2 = Agent(name='agent2'); agent3 = Agent(name='agent3')
    >>> Swarm((agent1, agent2), (agent1, agent3), (agent2, agent3), determinacy=False)
    """

    def build(self):
        valid_agent_pair = []
        for pair in self.swarm.topology:
            if isinstance(pair, (list, tuple)):
                # (agent1, agent2)
                if len(pair) != 2:
                    raise RuntimeError(f"{pair} is not a pair value, please check it.")
                elif not isinstance(pair[0], Agent) or not isinstance(pair[1], Agent):
                    raise RuntimeError(f"agent in {pair} is not a base agent instance, please check it.")
                valid_agent_pair.append(pair)
            elif len(self.swarm.topology) == 1:
                # agent
                if not isinstance(pair, Agent):
                    raise RuntimeError(f"agent in {pair} is not a base agent instance, please check it.")
                valid_agent_pair.append((pair,))
            else:
                raise RuntimeError(f"{pair} is unsupported in handoffs mechanism of agent, please check it.")

        if not valid_agent_pair:
            raise RuntimeError(f"no valid agent pair to build execution graph.")

        # Agent that communicate with the outside world, the default is the first if the root agent is None.
        if self.swarm.communicate_agent is None:
            self.swarm.communicate_agent = valid_agent_pair[0][0]

        # agent handoffs build.
        for pair in valid_agent_pair:
            self.swarm.ordered_agents.append(pair[0])
            self.swarm.ordered_agents.append(pair[1])

            if pair[0].name() not in self.swarm.agents:
                self.swarm.agents[pair[0].name()] = pair[0]

            TopologyBuilder.register_agent(pair[0])

            # only one agent
            if len(pair) == 1:
                continue

            if pair[1].name() not in self.swarm.agents:
                self.swarm.agents[pair[1].name()] = pair[1]

                TopologyBuilder.register_agent(pair[1])

            # need to explicitly set handoffs in the agent
            pair[0].handoffs.append(pair[1].name())
