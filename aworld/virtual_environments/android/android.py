# coding: utf-8
# Copyright (c) 2025 inclusionAI.

import traceback
from typing import Any, Tuple, List, Dict

from aworld.core.envs.tool_action import AndroidAction
from aworld.core.common import ActionModel, Observation, ActionResult, Tools
from aworld.virtual_environments.android.action.adb_controller import ADBController
from aworld.virtual_environments.android.action.executor import AndroidToolActionExecutor
from aworld.virtual_environments.conf import AndroidToolConfig
from aworld.core.envs.env_tool import ToolFactory, EnvTool

ALL_UNICODE_CHARS = frozenset(chr(i) for i in range(0x10FFFF + 1))


@ToolFactory.register(name=Tools.ANDROID.value, desc="android", supported_action=AndroidAction)
class AndroidTool(EnvTool[Observation, List[ActionModel]]):

    def __init__(self, conf: AndroidToolConfig, **kwargs):
        super(AndroidTool, self).__init__(conf, **kwargs)
        self.controller = ADBController(avd_name=conf.avd_name,
                                        adb_path=conf.adb_path,
                                        emulator_path=conf.emulator_path)

        self.action_executor = AndroidToolActionExecutor(self.controller)

    def name(self):
        return Tools.ANDROID.value

    def reset(self, *, seed: int | None = None, options: Dict[str, str] | None = None) -> Tuple[
        Observation, Dict[str, Any]]:
        # self.controller.stop_emulator()
        self.controller.start_emulator()
        print("start emulator successfully...")
        # snapshot screen and annotate
        xml, pic_base64 = self.get_observation()
        action_result_list = [ActionResult(content='start', keep=True)]
        return Observation(dom_tree=xml, image=pic_base64, action_result=action_result_list), {}

    def step(self, action_list: List[ActionModel], **kwargs) -> Tuple[
        Observation, float, bool, bool, Dict[str, Any]]:

        exec_state = 0
        fail_error = ""
        action_result_list = None
        try:
            action_result_list = self.action_executor.execute_action(action_list, **kwargs)
            exec_state = 1
        except Exception as e:
            traceback.print_exc()
            fail_error = str(e)

        info = {"exception": fail_error}
        xml, pic_base64 = self.get_observation()

        return (Observation(dom_tree=xml, image=pic_base64, action_result=action_result_list),
                exec_state,
                kwargs.get("terminated", False),
                kwargs.get("truncated", False),
                info)

    def close(self):
        self.controller.stop_emulator()

    def get_controller(self):
        return self.controller

    def get_observation(self) -> Observation:
        return self.controller.screenshot_and_annotate()
