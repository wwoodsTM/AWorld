# coding: utf-8
# Copyright (c) 2025 inclusionAI.
import os
import traceback
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any

import yaml
from pydantic import BaseModel

from aworld.logs.util import logger


def load_config(file_name: str, dir_name: str = None) -> Dict[str, Any]:
    """Dynamically load config file form current path.

    Args:
        file_name: Config file name.
        dir_name: Config file directory.

    Returns:
        Config dict.
    """

    if dir_name:
        file_path = os.path.join(dir_name, file_name)
    else:
        # load conf form current path
        current_dir = Path(__file__).parent.absolute()
        file_path = os.path.join(current_dir, file_name)
    if not os.path.exists(file_path):
        logger.warning(f"{file_path} not exists, please check it.")

    configs = dict()
    try:
        with open(file_path, "r") as file:
            yaml_data = yaml.safe_load(file)
        configs.update(yaml_data)
    except FileNotFoundError:
        logger.warning(f"Can not find the file: {file_path}")
    except Exception as e:
        logger.warning(f"{file_name} read fail.\n", traceback.format_exc())
    return configs


def wipe_secret_info(config: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """Return a deep copy of this config as a plain Dict as well ass wipe up secret info, used to log."""

    def _wipe_secret(conf):
        def _wipe_secret_plain_value(v):
            if isinstance(v, List):
                return [_wipe_secret_plain_value(e) for e in v]
            elif isinstance(v, Dict):
                return _wipe_secret(v)
            else:
                return v

        key_list = []
        for key in conf.keys():
            key_list.append(key)
        for key in key_list:
            if key.strip('"') in keys:
                conf[key] = '-^_^-'
            else:
                _wipe_secret_plain_value(conf[key])
        return conf

    if not config:
        return config
    return _wipe_secret(config)


class ModelConfig(BaseModel):
    llm_provider: str = None
    llm_model_name: str | None = None
    llm_temperature: float | None = None
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    max_input_tokens: int = 128000

class AgentConfig(BaseModel):
    agent_name: str = None
    max_steps: int = 10
    llm_provider: str = None
    llm_model_name: str | None = None
    llm_num_ctx: int | None = None
    llm_temperature: float | None = None
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    max_input_tokens: int = 128000
    max_actions_per_step: int = 10
    include_attributes: List[str] = [
        'title',
        'type',
        'name',
        'role',
        'aria-label',
        'placeholder',
        'value',
        'alt',
        'aria-expanded',
        'data-date-format',
    ]
    message_context: Optional[str] = None
    available_file_paths: Optional[List[str]] = None
    ext: dict | None = None


class TaskConfig(BaseModel):
    task_id: str | None = str(uuid.uuid4())
    task_name: str | None = None
    task: str | None = None
    max_steps: int = 100
    max_actions_per_step: int = 10
    ext: dict | None = None


class ToolConfig(BaseModel):
    use_vision: bool | None = None
    custom_executor: bool = True
    tool_calling_method: str | None = None
    headless: bool | None = False
    disable_security: bool | None = None
    enable_recording: bool | None = None
    window_w: int | None = None
    window_h: int | None = None
    save_recording_path: str | None = None
    save_trace_path: str | None = None
    save_agent_history_path: str | None = None
    task: str | None = None
    config_data: dict | None = None
    max_retry: int | None = None
    ext: dict | None = None
