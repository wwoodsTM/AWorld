# coding: utf-8
# Copyright (c) 2025 inclusionAI.

from aworld.core.envs.tool_action import PythonToolAction
from aworld.core.envs.action_factory import ActionFactory
from aworld.core.common import Tools, ActionModel, ActionResult
from aworld.virtual_environments.action import ExecutableAction


@ActionFactory.register(name=PythonToolAction.EXECUTE.value.name,
                        desc=PythonToolAction.EXECUTE.value.desc,
                        tool_name=Tools.PYTHON_EXECUTE.value)
class ExecuteAction(ExecutableAction):
    """Only one action, define it, implemented can be omitted."""