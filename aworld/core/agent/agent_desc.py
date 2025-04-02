# coding: utf-8
# Copyright (c) 2025 inclusionAI.
from typing import Dict, Any

from aworld.agents import agent_desc
from aworld.core.agent.base import AgentFactory, BaseAgent
from aworld.logs.util import logger


def get_agent_desc() -> Dict[str, dict]:
    return agent_desc


def get_agent_desc_by_name(name: str) -> Dict[str, Any]:
    return agent_desc.get(name, None)


def agent_handoffs_desc(agent: BaseAgent, use_all: bool = False) -> Dict[str, dict]:
    if not agent:
        if use_all:
            # use all agent description
            return agent_desc
        logger.warning(f"no agent to gen description!")
        return {}

    desc = {}
    # agent.handoffs never is None
    for reachable in agent.handoffs:
        res = get_agent_desc_by_name(reachable)
        if not res:
            logger.warning(f"{reachable} can not find in the agent factory, ignored it.")
            continue
        desc[reachable] = res

    return desc


def is_agent_by_name(name: str) -> bool:
    return name in AgentFactory
