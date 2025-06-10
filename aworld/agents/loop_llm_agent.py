# coding: utf-8
# Copyright (c) 2025 inclusionAI.
from typing import Any, Callable

from aworld.agents.llm_agent import Agent


class LoopableAgent(Agent):
    """Support for loop agents in the swarm.

    # NOTE: Can only be used for deterministic execution.
    """
    max_run_times: int = 1
    cur_run_times: int = 0
    # the loop agent only consider the starting and ending agent in the graph
    start_agent: Agent
    end_agent: Agent
    # todo: API to be determined
    transfer_condition: Callable[..., Any] = None
    stop_condition: Callable[..., Any] = None

    @property
    def finished(self) -> bool:
        if self.cur_run_times >= self.max_run_times or (self.stop_condition and self.stop_condition()):
            self._finished = True
            return True

        self._finished = False
        return False
