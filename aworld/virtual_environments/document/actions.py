# coding: utf-8
from typing import Tuple, Any

from aworld.core.envs.tool_action import DocumentExecuteAction
from aworld.core.envs.action_factory import ActionFactory
from aworld.core.common import ActionModel, ActionResult, Tools
from aworld.virtual_environments.action import ExecutableAction


@ActionFactory.register(name=DocumentExecuteAction.DOCUMENT_ANALYSIS.value.name,
                        desc=DocumentExecuteAction.DOCUMENT_ANALYSIS.value.desc,
                        tool_name=Tools.DOCUMENT_ANALYSIS.value)
class ExecuteAction(ExecutableAction):
    def act(self, action: ActionModel, **kwargs) -> Tuple[ActionResult, Any]:
        pass
