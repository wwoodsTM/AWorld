# coding: utf-8
# Copyright (c) 2025 inclusionAI.
import time

import traceback

import abc
import uuid
from typing import Union, Dict, Any, List

from pydantic import BaseModel

from aworld.core.agent.base import BaseAgent
from aworld.config import ToolConfig, load_config, wipe_secret_info
from aworld.core.common import Observation
from aworld.core.envs.env_tool import EnvTool, ToolFactory
from aworld.core.swarm import Swarm
from aworld.logs.util import logger, color_log


class Task(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 agent: BaseAgent = None,
                 swarm: Swarm = None,
                 name: str = uuid.uuid1().hex,
                 input: Any = None,
                 conf: Union[Dict[str, Any], BaseModel] = {},
                 tools: List[EnvTool] = None,
                 *args,
                 **kwargs):
        """Task instance init.

        Args:
            agent(required): Agent instance which want to run.
            input: A query string or dataset.
            conf: Task config in process.
            tools: Special tools in task run.
        """
        if isinstance(conf, BaseModel):
            conf = conf.model_dump()
        if not agent and not swarm:
            raise ValueError("agent and swarm all is None.")
        check_input = conf.get("check_input", False)
        if check_input and not input:
            raise ValueError

        self.agent = agent
        self.swarm = swarm
        self.input = input
        self.name = name
        self.conf = conf
        self.tools = {tool.name(): tool for tool in tools} if tools else {}
        self.daemon_target = kwargs.pop('daemon_target', None)
        self._use_demon = False if not conf else conf.get('use_demon', False)
        self._exception = None
        for k, v in kwargs.items():
            setattr(self, k, v)

    def before_run(self):
        pass

    def after_run(self):
        pass

    @abc.abstractmethod
    def run(self):
        """Raise exception if not success."""

    def start(self) -> Any:
        try:
            self.before_run()
            self._daemon_run()
            ret = self.run()
            return 0 if ret is None else ret
        except BaseException as ex:
            self._exception = ex
            # do record or report
            raise ex
        finally:
            self.after_run()

    def _daemon_run(self) -> None:
        if self._use_demon and self.daemon_target and callable(self.daemon_target):
            import threading
            t = threading.Thread(target=self.daemon_target, name="daemon")
            t.setDaemon(True)
            t.start()


class GeneralTask(Task):
    def __init__(self,
                 agent: BaseAgent = None,
                 swarm: Swarm = None,
                 name: str = uuid.uuid1().hex,
                 input: Any = None,
                 conf: Union[Dict[str, Any], BaseModel] = {},
                 tools: List[EnvTool] = None,
                 *args,
                 **kwargs):
        super().__init__(agent, swarm, name, input, conf, tools, *args, **kwargs)

    def run(self):
        # init tool state by reset(), and ignore them observation
        observation = None
        info = dict()
        if self.tools:
            for _, tool in self.tools.items():
                observation, info = tool.reset()

        if observation:
            if not observation.content:
                observation.content = self.input
        else:
            observation = Observation(content=self.input)

        if self.agent:
            return self.agent_process(observation, info)
        elif self.swarm:
            # example now
            return self.swarm_process(observation, info)

    def swarm_process(self,
                      observation: Observation,
                      info: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()

        input = observation.content
        self.swarm.reset(self.tools)
        for agent in self.swarm.agents.values():
            agent.reset({"task": input})

        step = 0
        max_steps = self.conf.get("max_steps", 100)
        results = []
        try:
            while step < max_steps:
                # Loose protocol
                result_dict = self.swarm.process(observation=observation, info=info)
                results.append(result_dict)

                swarm_resp = result_dict.get("response")
                logger.info(f"Step: {step} response:\n {result_dict}")

                step += 1
                if self.swarm.finished:
                    logger.info("task done!")
                    break

                if not swarm_resp:
                    logger.warning(f"Step: {step} swarm no valid response")
                    break

                observation = Observation(content=swarm_resp)

            time_cost = time.time() - start
            if not results:
                logger.warning("task no result!")
                return {"answer": "",
                        "traceback": traceback.format_exc(),
                        "steps": step,
                        "success": False,
                        "total_time": time_cost}

            answer = results[-1].get('response')
            return {"answer": answer,
                    "steps": step,
                    "success": True,
                    "total_time": (time.time() - start)}
        except Exception as e:
            logger.error(f"Task execution failed with error: {str(e)}\n{traceback.format_exc()}")
            return {"msg": str(e),
                    "traceback": traceback.format_exc(),
                    "steps": step,
                    "success": False,
                    "total_time": (time.time() - start)}

    def agent_process(self,
                      observation: Observation,
                      info: Dict[str, Any]) -> Dict[str, Any]:
        agent = self.agent
        agent.reset({"task": input})
        step = 0
        max_steps = self.conf.get("max_steps", 100)
        excep = None
        start = time.time()
        try:
            while step < max_steps:
                terminated = False
                # Get policy from agent, policy must is an action list or None
                policy = agent.policy(observation=observation, info=info)

                if not policy:
                    logger.warning(f"{observation} can not get the valid policy, exit task!")
                    excep = "invalid policy"
                    break

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
                                    f"conf is: {wipe_secret_info(conf, ['api_key'])}")
                        tool.reset()
                        tool_mapping[act.tool_name] = []
                        self.tools[act.tool_name] = tool
                    if act.tool_name not in tool_mapping:
                        tool_mapping[act.tool_name] = []
                    tool_mapping[act.tool_name].append(act)

                for tool_name, action in tool_mapping.items():
                    # Execute action using browser tool and unpack all return values
                    observation, reward, terminated, _, info = self.tools[tool_name].step(action)
                    # need merge?

                    logger.info(f'{action} state: {observation}; reward: {reward}')
                    # Check if there's an exception in info
                    if info.get("exception"):
                        color_log(f"Step {step} failed with exception: {info['exception']}")
                        excep = info.get("exception")

                    step += 1
                    logger.info(f"step: {step} finished by tool action.")
                    # Check if task should end (either terminated or truncated)
                    if agent.finished:
                        logger.info("agent or tools finished")
                        break

                if terminated:
                    logger.info(f"{self.name} task finished")
                    break

            time_cost = time.time() - start
            return {"steps": step,
                    "msg": excep,
                    "success": True if not excep else False,
                    "total_time": time_cost}
        except Exception as e:
            logger.error(f"Task execution failed with error: {str(e)}\n{traceback.format_exc()}")
            time_cost = time.time() - start
            return {
                "msg": str(e),
                "traceback": traceback.format_exc(),
                "steps": step,
                "success": False,
                "total_time": time_cost
            }
        finally:
            # Cleanup if not keeping open
            if self.tools:
                for _, tool in self.tools.items():
                    if not tool.dict_conf.get("keep_open", False):
                        tool.close()
