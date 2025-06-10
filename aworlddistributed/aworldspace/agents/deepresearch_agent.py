# coding: utf-8
# Copyright (c) 2025 inclusionAI.
import logging
import os
from typing import Optional

from anthropic import BaseModel
from aworld.agents.llm_agent import Agent
from aworld.config.conf import TaskConfig, ModelConfig

from aworld.core.agent.swarm import Swarm
from aworld.core.task import Task
from aworld.runner import Runners
from aworldspace.agents.sub_agents.plan_agent import plan_agent
from aworldspace.agents.sub_agents.reasoning_loop_agent import reasoning_loop_agent
from aworldspace.agents.sub_agents.reporting_agent import reporting_agent
from aworldspace.base_agent import AworldBaseAgent


class Pipeline(AworldBaseAgent):
    class Valves(BaseModel):
        pass

    def __init__(self):
        self.valves = self.Valves()
        logging.info("deepresearch_agent init success")

    async def build_swarm(self, body):
        return Swarm(plan_agent, reasoning_loop_agent, reporting_agent,
                      sequence=True)

    async def build_task(self, agent: Optional[Agent],swarm: Optional[Swarm], task_id, user_input, user_message, body):
        task = Task(
            id = task_id,
            swarm=swarm,
            input=user_input,
            endless_threshold=5,
            conf=TaskConfig(exit_on_failure=True),
            event_driven=False
        )
        return task

    def agent_name(self) -> str:
        return "DeepSearchAgent"
