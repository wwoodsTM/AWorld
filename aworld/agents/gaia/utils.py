# coding: utf-8
# Copyright (c) 2025 inclusionAI.
from typing import Dict, Any, List

from aworld.logs.util import logger


def openai_tool_desc_transform(tool_dict: Dict[str, Any], tools: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Default implement transform framework standard protocol to openai protocol of tool description."""
    openai_tools = []
    for tool_name, tool_info in tool_dict.items():
        if tools and tool_name not in tools:
            logger.info(f"{tool_name} can not supported, you can set `tools` params to support it.")
            continue

        for action in tool_info["actions"]:
            # 构建参数属性
            properties = {}
            required = []
            for param_name, param_info in action["params"].items():
                properties[param_name] = {
                    "description": param_info["desc"],
                    "type": param_info["type"] if param_info["type"] != "str" else "string"
                }
                if param_info.get("required", False):
                    required.append(param_name)

            openai_function_schema = {
                "name": f'{tool_name}__{action["name"]}',
                "description": action["desc"],
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }

            openai_tools.append({
                "type": "function",
                "function": openai_function_schema
            })
    return openai_tools
