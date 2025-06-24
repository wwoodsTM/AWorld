# coding: utf-8
# Copyright (c) 2025 inclusionAI.
import abc
import uuid
from dataclasses import dataclass, field
from typing import Any, Union, List, Dict, Callable, Optional

from pydantic import BaseModel

from aworld.core.agent.llm_agent import Agent
from aworld.core.agent.swarm import Swarm
from aworld.core.common import Config
from aworld.core.context.base import Context
from aworld.core.tool.base import Tool, AsyncTool
from aworld.output.outputs import Outputs, StreamingOutputs, DefaultOutputs


@dataclass
class Task:
    id: str = uuid.uuid1().hex
    name: str = uuid.uuid1().hex
    user_id: str = None
    session_id: str = None
    input: Any = None
    # task config
    conf: Config = None
    # global tool instance
    tools: List[Union[Tool, AsyncTool]] = field(default_factory=list)
    # global tool names
    tool_names: List[str] = field(default_factory=list)
    # custom tool conf
    tools_conf: Config = field(default_factory=dict)
    # custom mcp servers conf
    mcp_servers_conf: Config = field(default_factory=dict)
    swarm: Optional[Swarm] = None
    agent: Optional[Agent] = None
    event_driven: bool = True
    # for loop detect
    endless_threshold: int = 3
    # task_outputs
    outputs: Outputs = field(default_factory=DefaultOutputs)
    # task special runner class, for example: package.XXRunner
    runner_cls: Optional[str] = None
    # such as: {"start": ["init_tool", "init_context", ...]}
    hooks: Dict[str, List[str]] = field(default_factory=dict)


class TaskResponse(BaseModel):
    id: str
    answer: str | None
    usage: Dict[str, Any] | None = None
    time_cost: float | None = None
    success: bool = False
    msg: str | None = None
