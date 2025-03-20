# coding: utf-8
# Copyright (c) 2025 inclusionAI.
from aworld.agents import agent_desc
from aworld.core.agent.base import AgentFactory


def get_agent_desc():
    return agent_desc


def get_agent_desc_by_name(name: str):
    return getattr(agent_desc, name, None)


def is_agent_by_name(name: str) -> bool:
    return name in AgentFactory
