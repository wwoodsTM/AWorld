# coding: utf-8
# Copyright (c) 2025 inclusionAI.

from typing import List, Dict

from pydantic import BaseModel

from aworld.core.env.action_factory import ActionFactory
from aworld.core.common import ParamInfo
from aworld.logs.util import logger
from aworld.core.env.env_tool import ToolFactory


class ActionDescriptionProtocol(BaseModel):
    name: str
    desc: str
    params: Dict[str, ParamInfo]


class ToolDescriptionProtocol(BaseModel):
    name: str
    desc: str
    actions: List[ActionDescriptionProtocol] = []


def tool_desc():
    """Utility method of generate description of tools and their actions.

    The standard protocol can be transformed based on the API of different llm.
    Define as follows:
    ```
    {
        "tool_name": {
            "desc": "A toolkit description.",
            "actions": [
                {
                    "name": "action name",
                    "desc": "action description.",
                    "params": {
                        "param_name": {
                            "desc": "param description.",
                            "type": "param type, such as int, str, etc.",
                            "required": True | False
                        }
                    }
                }
            ]
        }
    }
    ```
    """

    def process(action_info):
        action_dict = dict()
        action_dict["name"] = action_info.name
        action_dict["desc"] = action_info.desc
        action_dict["params"] = dict()

        for k, v in action_info.input_params.items():
            params_dict = v.model_dump()
            params_dict.pop("name")
            action_dict["params"][k] = params_dict
        return action_dict

    descs = dict()
    for tool in ToolFactory:
        tool_val_dict = dict()
        descs[tool] = tool_val_dict

        tool_val_dict["desc"] = ToolFactory.desc(tool)
        tool_action = ToolFactory.get_tool_action(tool)
        actions = []
        action_names = ActionFactory.get_actions_by_tool(tool_name=tool)
        if action_names:
            for action_name in action_names:
                info = tool_action.get_value_by_name(action_name)
                if not info:
                    logger.warning(f"{action_name} can not find in {tool}, please check it.")
                    continue
                try:
                    action_dict = process(info)
                except:
                    logger.warning(f"{action_name} process fail.")
                    action_dict = dict()
                actions.append(action_dict)
        elif tool_action:
            for k, info in tool_action.__members__.items():
                action_dict = process(info.value)
                actions.append(action_dict)
        else:
            logger.warning(f"{tool} no action!")
        tool_val_dict["actions"] = actions
    return descs


tool_action_desc_dict = tool_desc()


def get_actions() -> List[str]:
    res = []
    for _, tool_info in tool_action_desc_dict.items():
        actions = tool_info.get("actions")
        if not actions:
            continue

        for action in actions:
            res.append(action['name'])
    return res


def get_actions_by_tools(tool_names: Dict = None) -> List[str]:
    if not tool_names:
        return get_actions()

    res = []
    for tool_name, tool_info in tool_action_desc_dict.items():
        if tool_name not in tool_names:
            continue

        actions = tool_info.get("actions")
        if not actions:
            continue

        for action in actions:
            res.append(action['name'])
    return res


def get_desc_by_tool(name: str):
    return getattr(tool_action_desc_dict, name, None)


def is_tool_by_name(name: str) -> bool:
    return name in ToolFactory
