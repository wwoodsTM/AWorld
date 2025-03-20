# coding: utf-8
# Copyright (c) 2025 inclusionAI.

import time
import traceback
from typing import Dict, Any, List

from aworld.core.agent.base import Agent, BaseAgent
from aworld.core.envs.env_tool import ToolFactory
from aworld.logs.util import logger, color_log
from aworld.core.common import Observation, ActionModel
from aworld.config.conf import ToolConfig, load_config
from aworld.core.envs.tool_desc import get_actions_by_tools


class Swarm(object):
    """Implementation of interactive collaboration between multi-agent and interaction with env tools."""

    def __init__(self, *args, **kwargs):
        valid_agent_pair = []
        for pair in args:
            if len(pair) != 2:
                logger.warning(f"{pair} is not a pair value, ignore it.")
                continue
            if not isinstance(pair[0], BaseAgent) or not isinstance(pair[1], BaseAgent):
                logger.warning(f"agent in {pair} is not a agent instance, ignore it.")
                continue
            valid_agent_pair.append(pair)

        self.initialized = False
        if not valid_agent_pair:
            logger.warning("no valid agent pair to build graph.")
            return

        self.conf = kwargs
        # entrance agent
        self.entry_agent: Agent = valid_agent_pair[0][0]
        # agents in swarm
        self.agents = {}

        for pair in valid_agent_pair:
            if pair[0] not in self.agents:
                self.agents[pair[0].name()] = pair[0]
            if pair[1] not in self.agents:
                self.agents[pair[1].name()] = pair[1]

            pair[0].handoffs.append(pair[1].name())
        self.finished = False

    def reset(self, tools):
        if not self.agents:
            logger.warning("No valid agent in swarm.")
            return

        # can only use the special tools in the swarm as a global
        self.tools_actions = get_actions_by_tools(tools)
        self.tools = tools
        self.cur_agent = self.entry_agent
        self.initialized = True

    def is_agent(self, policy: ActionModel):
        return policy.tool_name is None and policy.action_name is None

    def process(self, observation, info) -> Dict[str, Any]:
        """Multi-agent general process workflow."""
        if not self.initialized:
            raise RuntimeError("swarm needs to use `reset` to init first.")

        start = time.time()
        step = 0
        max_steps = self.conf.get("max_steps", 100)

        policy: List[ActionModel] = self.cur_agent.policy(observation, info)
        if not policy:
            logger.warning(f"current agent {self.cur_agent.name()} no policy to use.")
            return {"msg": f"current agent {self.cur_agent.name()} no policy to use.",
                    "steps": step,
                    "success": False,
                    "time_cost": (time.time() - start)}

        if self.tools is None:
            self.tools = {}

        msg = None
        try:
            while step < max_steps:
                terminated = False
                if self.is_agent(policy[0]):
                    # only one agent, and get agent from policy
                    policy_for_agent = policy[0]
                    cur_agent: BaseAgent = self.agents.get(policy_for_agent.agent_name)
                    if not cur_agent:
                        raise RuntimeError(f"Can not find {policy_for_agent.agent_name} agent in swarm.")
                    if self.cur_agent.handoffs and policy_for_agent.agent_name not in self.cur_agent.handoffs:
                        return {"msg": f"Can not handoffs {policy_for_agent.agent_name} agent "
                                       f"by {cur_agent.name()} agent.",
                                "steps": step,
                                "success": False}

                    observation = Observation(content=policy_for_agent.policy_info)
                    agent_policy = cur_agent.policy(observation, info=info)
                    if not agent_policy:
                        logger.warning(
                            f"{observation} can not get the valid policy in {policy_for_agent.agent_name}, exit task!")
                        msg = f"{policy_for_agent.agent_name} invalid policy"
                        break
                    self.cur_agent = cur_agent
                    policy = agent_policy
                    # clear observation
                    observation = None
                else:
                    # group action by tool name
                    tool_mapping = dict()
                    # Directly use or use tools after creation.
                    for act in policy:
                        if not self.tools or (self.tools and act.tool_name not in self.tools):
                            # only use default config in module or XXConfig.
                            conf = load_config(f"{act.tool_name}.yaml")
                            if not conf:
                                conf = ToolConfig()
                            tool = ToolFactory(act.tool_name, conf=conf)
                            logger.info(f"Dynamic load config from {act.tool_name}.yaml, "
                                        f"conf is: {conf}")
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
                    # Check if current agent done
                    if self.cur_agent.finished:
                        logger.info(f"{self.cur_agent.name()} agent finished.")
                        terminated = True
                        break

                step += 1
                if terminated:
                    logger.info("swarm finished")
                    break

                if observation:
                    policy = self.cur_agent.policy(observation, info)

            if self.entry_agent.finished:
                self.finished = True
            return {"steps": step,
                    "msg": msg,
                    "success": True if not msg else False}
        except Exception as e:
            logger.error(f"Task execution failed with error: {str(e)}\n{traceback.format_exc()}")
            return {
                "msg": str(e),
                "traceback": traceback.format_exc(),
                "steps": step,
                "success": False
            }

    @property
    def is_finished(self) -> bool:
        return self.finished
