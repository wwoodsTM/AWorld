# coding: utf-8
from typing import Tuple, Any

from aworld.core.action import ShellAction
from aworld.core.action_factory import ActionFactory
from aworld.core.common import Tools, ActionModel, ActionResult
from aworld.virtual_environments.action import ExecutableAction


@ActionFactory.register(name=ShellAction.EXECUTE_SCRIPT.value.name,
                        desc=ShellAction.EXECUTE_SCRIPT.value.desc,
                        tool_name=Tools.SHELL.value)
class ShellAction(ExecutableAction):
    def act(self, action: ActionModel, **kwargs) -> Tuple[ActionResult, Any]:
        pass
