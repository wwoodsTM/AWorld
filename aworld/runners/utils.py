# coding: utf-8
# Copyright (c) 2025 inclusionAI.
from typing import List, Dict

from aworld.config import ConfigDict
from aworld.core.common import Config
from aworld.core.runtime_engine import LOCAL

from aworld.core.task import Task, Runner, TaskResponse
from aworld.logs.util import logger
from aworld.utils.common import new_instance, snake_to_camel


class TaskType:
    START = "__start"
    FINISHED = "__finished"
    OUTPUT = "__output"
    ERROR = "__error"
    RERUN = "__rerun"
    HUMAN_CONFIRM = "__human_confirm"
    # for dynamic subscribe
    SUBSCRIBE_TOOL = "__subscribe_tool"
    SUBSCRIBE_AGENT = "__subscribe_agent"


async def choose_runners(tasks: List[Task]) -> List[Runner]:
    """Choose the correct runner to run the task.

    Args:
        task: A task that contains agents, tools and datas.

    Returns:
        Runner instance or exception.
    """
    runners = []
    for task in tasks:
        # user custom runner class
        runner_cls = task.runner_cls
        if runner_cls:
            return new_instance(runner_cls, task)
        else:
            # user runner class in the framework
            if task.swarm:
                task.swarm.event_driven = task.event_driven
                task.swarm.reset(task.input)
                topology = task.swarm.topology_type
            else:
                topology = "sequence"

            if task.event_driven:
                runner = new_instance("aworld.runners.event_runner.TaskEventRunner", task)
            else:
                runner = new_instance(
                    f"aworld.runners.call_driven_runner.{snake_to_camel(topology)}Runner",
                    task
                )
        runners.append(runner)
    return runners


async def execute_runner(runners: List[Runner], run_conf: Config) -> Dict[str, TaskResponse]:
    """Execute runner in the runtime engine.

    Args:
        runners: The task processing flow.
        run_conf: Runtime config, can choose the special computing engine to execute the runner.
    """
    if not run_conf:
        run_conf = ConfigDict({"name": LOCAL})

    name = run_conf.name
    if run_conf.get('cls'):
        runtime_backend = new_instance(run_conf.cls, run_conf)
    else:
        runtime_backend = new_instance(
            f"aworld.core.runtime_engine.{snake_to_camel(name)}Runtime", run_conf)
    runtime_engine = runtime_backend.build_engine()
    return await runtime_engine.execute([runner.run for runner in runners])


def endless_detect(records: List[str], endless_threshold: int, root_agent_name: str):
    """A very simple implementation of endless loop detection.

    Args:
        records: Call sequence of agent.
        endless_threshold: Threshold for the number of repetitions.
        root_agent_name: Name of the entrance agent.
    """
    if not records:
        return False

    threshold = endless_threshold
    last_agent_name = root_agent_name
    count = 1
    for i in range(len(records) - 2, -1, -1):
        if last_agent_name == records[i]:
            count += 1
        else:
            last_agent_name = records[i]
            count = 1

        if count >= threshold:
            logger.warning("detect loop, will exit the loop.")
            return True

    if len(records) > 6:
        last_agent_name = None
        # latest
        for j in range(1, 3):
            for i in range(len(records) - j, 0, -2):
                if last_agent_name and last_agent_name == (records[i], records[i - 1]):
                    count += 1
                elif last_agent_name is None:
                    last_agent_name = (records[i], records[i - 1])
                    count = 1
                else:
                    last_agent_name = None
                    break

                if count >= threshold:
                    logger.warning(f"detect loop: {last_agent_name}, will exit the loop.")
                    return True

    return False
