# coding: utf-8
# Copyright (c) 2025 inclusionAI.

import time
import traceback
from threading import Thread

from typing import Any, List, Dict, Union

from aworld.core.singleton import InheritanceSingleton
from aworld.core.task import Task
from aworld.logs.util import logger


class ReturnableThread(Thread):
    """This class is a subclass of `Thread` that allows the thread to return value."""

    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}):
        Thread.__init__(self, group, target, name, args, kwargs)
        self.result = None

    def run(self):
        if self._target is not None:
            self.result = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self.result


class Client(InheritanceSingleton):
    """Client class for executing tasks."""

    def __init__(self):
        pass

    def submit(self, task: Union[Task, List[Task]] = None, parallel: bool = False, **kwargs) -> Dict[str, Any]:
        """Run the task in local.

        Returns:
            The result of the task execution.
        """
        start = time.time()
        res = {}

        if isinstance(task, Task):
            self._run(task, res)
        else:
            if parallel:
                run_list = []
                for tas in task:
                    process = ReturnableThread(name=tas.name, target=tas.start)
                    run_list.append(process)

                for t in run_list:
                    t.start()
                for idx, t in enumerate(run_list):
                    res[f"task_{str(idx)}"] = t.join()
            else:
                for t in task:
                    self._run(t, res)

        res['success'] = True
        res['time_cost'] = time.time() - start
        return res

    def _run(self, task: Task, res: Dict[str, Any]) -> None:
        try:
            # Execute the task
            result = task.start()
            res['task_0'] = result
        except Exception as e:
            logger.error(traceback.format_exc())
            # Re-raise the exception to allow caller to handle it
            raise e
