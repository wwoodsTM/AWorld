# coding: utf-8
# Copyright (c) 2025 inclusionAI.
from typing import Any, Callable

from aworld.agents.llm_agent import Agent


class LoopableAgent(Agent):
    max_run_times: int = 1
    cur_run_times: int = 0
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
