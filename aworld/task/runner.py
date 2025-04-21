# coding: utf-8
# Copyright (c) 2025 inclusionAI.
import abc
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Union, List, Dict, Callable

from pydantic import BaseModel

from aworld.config import ConfigDict, ToolConfig
from aworld.framework.agent.base import Agent, BaseAgent
from aworld.framework.agent.swarm import Swarm
from aworld.framework.envs.tool import Tool
from aworld.logs.util import logger

Config = Dict[str, Union[Dict[str, Any], ConfigDict, ToolConfig]]


@dataclass
class Task:
    name: str = uuid.uuid1().hex
    input: Any = None
    conf: Union[ConfigDict, Dict[str, Any], BaseModel] = None
    # global tool instance for
    tools: List[Tool] = field(default_factory=list)
    tool_names: List[str] = field(default_factory=list)
    tools_conf: Config = field(default_factory=dict)
    mcp_servers_conf: Config = field(default_factory=dict)
    swarm: Swarm = None
    agent: Agent = None
    endless_threshold: int = 3


class Runner(object):
    __metaclass__ = abc.ABCMeta

    _use_demon: bool = False
    daemon_target: Callable[..., Any] = None

    async def pre_run(self):
        pass

    async def post_run(self):
        pass

    @abc.abstractmethod
    async def do_run(self):
        """"""

    async def _daemon_run(self):
        if self._use_demon and self.daemon_target and callable(self.daemon_target):
            import threading
            t = threading.Thread(target=self.daemon_target, name="daemon", daemon=True)
            t.start()

    async def run(self) -> Any:
        try:
            await self.pre_run()
            await self._daemon_run()
            ret = await self.do_run()
            return 0 if ret is None else ret
        except BaseException as ex:
            self._exception = ex
            # do record or report
            raise ex
        finally:
            await self.post_run()


class Runners:
    @staticmethod
    async def run(agent: BaseAgent, task: Task | List[Task], parallel: bool = False):
        if agent and task:
            raise ValueError('`agent` and `task` can only choose one.')

        if parallel:
            pass
        else:
            # SequenceTaskRunner(task).run()
            pass

        res = {}
        if isinstance(task, Task):
            Runners._run_in_local(agent, task, res)
        else:
            if parallel:
                await Runners._parallel_run_in_local(task, res)
            else:
                for i, t in enumerate(task):
                    self._run_in_local(t, res, i, None)

        res['success'] = True
        res['time_cost'] = time.time() - start
        return res

    @staticmethod
    async def _parallel_run_in_local(task, res):
        with ProcessPoolExecutor() as pool:
            loop = self.loop()
            tasks = [loop.run_in_executor(pool, t.start) for t in task]

            results = await asyncio.gather(*tasks)
            for idx, t in enumerate(results):
                res[f'task_{idx}'] = t

    @staticmethod
    async def _run_in_local(task: Task, res: Dict[str, Any], idx: int = 0) -> None:
        try:
            # Execute the task
            result = task.start()
            res[f'task_{idx}'] = result
            return result
        except Exception as e:
            logger.error(traceback.format_exc())
            # Re-raise the exception to allow caller to handle it
            raise e
