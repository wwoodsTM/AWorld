# coding: utf-8
from typing import Tuple, Any

from aworld.core.action import DocumentExecuteAction
from aworld.core.action_factory import ActionFactory
from aworld.core.common import ToolActionModel, ActionResult, Tools
from aworld.virtual_environments import ExecutableAction


@ActionFactory.register(name=DocumentExecuteAction.DOCUMENT_ANALYSIS.value.name,
                        desc=DocumentExecuteAction.DOCUMENT_ANALYSIS.value.desc,
                        tool_name=Tools.DOCUMENT_ANALYSIS.value)
class ExecuteAction(ExecutableAction):
    def act(self, action: ToolActionModel, **kwargs) -> Tuple[ActionResult, Any]:
        pass
