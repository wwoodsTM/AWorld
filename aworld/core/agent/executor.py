# coding: utf-8
# Copyright (c) 2025 inclusionAI.
import time
import traceback
from typing import Dict, Any, List

from aworld.config.conf import AgentConfig

from aworld.core.agent.base import BaseAgent, AgentResult, AgentFactory
from aworld.core.agent.swarm import Swarm
from aworld.core.common import Observation, ActionModel
from aworld.logs.util import logger, color_log


class AgentExecutor(object):
    """The default executor for agent execution can be used for sequential execution by the user."""

    def __init__(self, agent: BaseAgent = None):
        self.agent = agent
        self.agents: Dict[str, BaseAgent] = {}

    def register(self, name: str, agent: BaseAgent):
        self.agents[name] = agent

    def execute(self, observation: Observation, **kwargs) -> AgentResult:
        """"""
        return self.execute_agent(observation, self.agent, **kwargs)

    async def async_execute(self, observation: Observation, **kwargs) -> AgentResult:
        """"""
        return await self.async_execute_agent(observation, self.agent, **kwargs)

    def execute_agent(self,
                      observation: Observation,
                      agent: BaseAgent,
                      **kwargs) -> AgentResult:
        """The synchronous execution process of the agent with some hooks.

        Args:
            observation: Observation source from a tool or an agent.
            agent: The special agent instance.
        """
        agent = self._get_agent(observation.to_agent_name, agent, kwargs.get('conf'))
        try:
            actions = agent.policy(observation, kwargs)
            return AgentResult(actions=actions, current_state=None)
        except:
            logger.warning(traceback.format_exc())
            return AgentResult(actions=[ActionModel(tool_name="", action_name="")], current_state=None)

    async def async_execute_agent(self,
                                  observation: Observation,
                                  agent: BaseAgent,
                                  **kwargs) -> AgentResult:
        """The asynchronous execution process of the agent.

        Args:
            observation: Observation source from a tool or an agent.
            agent: The special agent instance.
        """
        agent = self._get_agent(observation.to_agent_name, agent, kwargs.get('conf'))
        try:
            actions = await agent.async_policy(observation, kwargs)
            return AgentResult(actions=actions, current_state=None)
        except:
            logger.warning(traceback.format_exc())
            return AgentResult(actions=[ActionModel(tool_name="", action_name="")], current_state=None)

    def _get_agent(self, name: str, agent: BaseAgent = None, conf=None):
        if agent is None:
            agent = self.agents.get(name)
            if agent is None:
                agent = AgentFactory(name, conf=conf if conf else AgentConfig())
                self.agents[name] = agent
        return agent


agent_executor = AgentExecutor()


class SwarmExecutor(object):
    """Default Executor for the agent execution."""

    def __init__(self, swarm: Swarm):
        self.swarm = swarm
        self.agents: Dict[str, BaseAgent] = {}

    def execute(self, observation: Observation, **kwargs) -> Dict[str, Any]:
        """The synchronous execution process of the swarm with some hooks.

        Args:
            observation: Observation source from a tool or an agent.
        """
        return self._process(observation, self.swarm, kwargs)

    async def async_execute(self, observation: Observation, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Default swarm `async_execute` not implemented.")

    def _process(self, observation: Observation, swarm: Swarm, info: Dict[str, Any]) -> Dict[str, Any]:
        """Default Multi-agent general process workflow.

        NOTE: Use the agent‘s finished state to control the loop, so the agent must carefully set finished state.
        Args:
            observation: Observation based on env
            info: Extend info by env
        """
        if not swarm:
            raise RuntimeError('no swarm instance to execute.')
        if not swarm.initialized:
            raise RuntimeError("swarm needs to use `reset` to init first.")

        start = time.time()
        step = 0
        max_steps = info.get("max_steps", 100)
        self.swarm.cur_agent = self.swarm.communicate_agent
        # use entry agent every time
        policy: List[ActionModel] = self.swarm.cur_agent.policy(observation, info)
        if not policy:
            logger.warning(f"current agent {self.swarm.cur_agent.name()} no policy to use.")
            return {"msg": f"current agent {self.swarm.cur_agent.name()} no policy to use.",
                    "steps": step,
                    "success": False,
                    "time_cost": (time.time() - start)}

        msg = None
        response = None
        return_entry = False
        cur_agent = None
        finished = False
        try:
            while step < max_steps:
                terminated = False
                if self.is_agent(policy[0]):
                    # only one agent, and get agent from policy
                    policy_for_agent = policy[0]
                    cur_agent: BaseAgent = self.swarm.agents.get(policy_for_agent.agent_name)
                    if not cur_agent:
                        raise RuntimeError(f"Can not find {policy_for_agent.agent_name} agent in swarm.")
                    if cur_agent.name() == self.swarm.communicate_agent.name():
                        # Current agent is entrance agent, means need to exit to the outer loop
                        logger.warning("Exit to the outer loop")
                        return_entry = True
                        break

                    if self.swarm.cur_agent.handoffs and policy_for_agent.agent_name not in self.swarm.cur_agent.handoffs:
                        # Unable to hand off, exit to the outer loop
                        return {"msg": f"Can not handoffs {policy_for_agent.agent_name} agent "
                                       f"by {cur_agent.name()} agent.",
                                "response": policy[0].policy_info if policy else "",
                                "steps": step,
                                "success": False}
                    # Check if current agent done
                    if cur_agent.finished:
                        cur_agent._finished = False
                        logger.info(f"{cur_agent.name()} agent be be handed off, so finished state reset to False.")

                    observation = Observation(content=policy_for_agent.policy_info)
                    agent_policy = cur_agent.policy(observation, info=info)
                    if not agent_policy:
                        logger.warning(
                            f"{observation} can not get the valid policy in {policy_for_agent.agent_name}, exit task!")
                        msg = f"{policy_for_agent.agent_name} invalid policy"
                        break
                    self.swarm.cur_agent = cur_agent
                    policy = agent_policy
                    # clear observation
                    observation = None
                else:
                    # group action by tool name
                    tool_mapping = dict()
                    # Directly use or use tools after creation.
                    for act in policy:
                        if not self.tools or (self.tools and act.tool_name not in self.tools):
                            # dynamic only use default config in module.
                            tool = ToolFactory(act.tool_name)
                            tool.reset()
                            tool_mapping[act.tool_name] = []
                            self.tools[act.tool_name] = tool
                        if act.tool_name not in tool_mapping:
                            tool_mapping[act.tool_name] = []
                        tool_mapping[act.tool_name].append(act)

                    for tool_name, action in tool_mapping.items():
                        # Execute action using browser tool and unpack all return values
                        observation, reward, terminated, _, info = self.tools[tool_name].step(action)

                        logger.info(f'{action} state: {observation}; reward: {reward}')
                        # Check if there's an exception in info
                        if info.get("exception"):
                            color_log(f"Step {step} failed with exception: {info['exception']}")
                            msg = info.get("exception")
                        logger.info(f"step: {step} finished by tool action.")

                    # The tool results give itself, exit; give to other agents, continue
                    tmp_name = policy[0].agent_name
                    if self.swarm.cur_agent.name() == self.swarm.communicate_agent.name() and (
                            len(self.swarm.agents) == 1 or tmp_name is None or self.swarm.cur_agent.name() == tmp_name):
                        return_entry = True
                        break
                    elif policy[0].agent_name:
                        policy_for_agent = policy[0]
                        cur_agent: BaseAgent = self.swarm.agents.get(policy_for_agent.agent_name)
                        if not cur_agent:
                            raise RuntimeError(f"Can not find {policy_for_agent.agent_name} agent in swarm.")
                        if self.swarm.cur_agent.handoffs and policy_for_agent.agent_name not in self.swarm.cur_agent.handoffs:
                            # Unable to hand off, exit to the outer loop
                            return {"msg": f"Can not handoffs {policy_for_agent.agent_name} agent "
                                           f"by {cur_agent.name()} agent.",
                                    "response": policy[0].policy_info if policy else "",
                                    "steps": step,
                                    "success": False}
                        # Check if current agent done
                        if cur_agent.finished:
                            cur_agent._finished = False
                            logger.info(f"{cur_agent.name()} agent be be handed off, so finished state reset to False.")

                step += 1
                if terminated and self.swarm.cur_agent.finished:
                    logger.info("swarm finished")
                    break

                if observation:
                    if cur_agent is None:
                        cur_agent = self.swarm.cur_agent
                    policy = cur_agent.policy(observation, info)

            if policy:
                response = policy[0].policy_info if policy[0].policy_info else policy[0].action_name

            # All agents or tools have completed their tasks
            if all(agent.finished for _, agent in self.swarm.agents.items()) or (all(
                    tool.finished for _, tool in self.tools.items()) and len(self.swarm.agents) == 1):
                logger.info("entry agent finished, swarm process finished.")
                finished = True

            if return_entry and not finished:
                # Return to the entrance, reset current agent finished state
                self.swarm.cur_agent._finished = False
            return {"steps": step,
                    "response": response,
                    "observation": observation,
                    "msg": msg,
                    "success": True if not msg else False}
        except Exception as e:
            logger.error(f"Task execution failed with error: {str(e)}\n{traceback.format_exc()}")
            return {
                "msg": str(e),
                "response": "",
                "traceback": traceback.format_exc(),
                "steps": step,
                "success": False
            }
