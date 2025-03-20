# coding: utf-8
# Copyright (c) 2025 inclusionAI.

import time
import json
import traceback
from typing import Dict, Any, Optional, List, Union

from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage

from aworld.core.agent.base import AgentFactory, BaseAgent, AgentResult

from aworld.agents.browser.common import AgentStepInfo
from aworld.config.conf import AgentConfig
from aworld.core.common import Observation, ActionModel, Tools, ToolActionInfo, Agents
from aworld.core.envs.tool_action import AndroidAction
from aworld.logs.util import logger
from aworld.models.llm import get_llm_model
from aworld.agents.android.common import (
    AgentState,
    AgentSettings,
    AgentHistory,
    AgentHistoryList,
    ActionResult,
    PolicyMetadata,
    AgentBrain
)

SYSTEM_PROMPT = """
你是一个Android设备自动化助手。你的任务是帮助用户在Android设备上执行各种操作。
你可以执行以下操作：
1. 点击元素 (tap) - 需要参数: index (元素编号)
2. 输入文本 (text) - 需要参数: params.text (要输入的文本内容)
3. 长按元素 (long_press) - 需要参数: index (元素编号)
4. 滑动元素 (swipe) - 需要参数: index (元素编号), params.direction (方向: "up", "down", "left", "right"), params.dist (距离: "short", "medium", "long", 可选，默认为"medium")
5. 任务完成 (done) - 需要参数: success (是否成功完成任务,取值true\false)

每个可交互元素都有一个编号。你需要根据界面上显示的元素编号来执行操作。
元素编号从1开始，0不是有效的元素编号。
当前界面的XML和截图会作为你的输入。请仔细分析界面元素，选择正确的操作。

重要提示：请直接返回JSON格式的响应，不要包含任何其他文本、解释或代码块标记。
响应必须是一个有效的JSON对象，格式如下：

{
    "current_state": {
        "evaluation_previous_goal": "分析上一步的执行结果",
        "memory": "记住重要的上下文信息",
        "next_goal": "下一步要执行的具体目标"
    },
    "action": [
        {
            "type": "tap",
            "index": "元素编号(从1开始的整数),不要每次都返回index等于1"
        },
        {
            "type": "text",
            "params": {
                "text": "要输入的文本内容"
            }
        },
        {
            "type": "long_press",
            "index": "元素编号(从1开始的整数),不要每次都返回index等于1"
        },
        {
            "type": "swipe",
            "index": "元素编号(从1开始的整数),不要每次都返回index等于1",
            "params": {
                "direction": "滑动方向(up/down/left/right)",
                "dist": "滑动距离(short/medium/long, 可选)"
            }
        },
        {
            "type": "done",
            "success": "是否成功完成任务(true/false)"
        }
    ]
}

注意：
- index必须是一个从1开始的有效整数
- 不要在JSON前后添加任何其他文本或标记
- 确保JSON格式完全正确
- 每种操作类型必须包含其所需的所有必要参数
"""


@AgentFactory.register(name=Agents.ANDROID.value, desc="browser agent")
class AndroidAgent(BaseAgent):
    def __init__(self, input: str, conf: AgentConfig, android_tool, observation, **kwargs):
        super(AndroidAgent, self).__init__(conf, **kwargs)
        self._build_prompt()
        self.task = input
        self.available_actions_desc = self._build_action_prompt()
        # Settings
        self.settings = AgentSettings(**conf.model_dump())
        self.model_name = conf.llm_model_name
        self.llm = get_llm_model(conf)
        self.android_tool = android_tool
        self.observation = observation
        # State
        self.state = AgentState()
        # History
        self.history = AgentHistoryList(history=[])

    def name(self) -> str:
        return Agents.ANDROID.value

    def _build_action_prompt(self) -> str:
        def _prompt(info: ToolActionInfo) -> str:
            s = f'{info.desc}:\n'
            s += '{' + str(info.name) + ': '
            if info.input_params:
                s += str({k: {"title": k, "type": v} for k, v in info.input_params.items()})
            s += '}'
            return s

        # Iterate over all android actions
        val = "\n".join([_prompt(v.value) for k, v in AndroidAction.__members__.items()])
        return val

    def _build_prompt(self):
        # If additional system prompt building is required for the android agent, implement here.
        pass

    def policy(self,
               observation: Observation,
               info: Dict[str, Any] = None,
               **kwargs) -> Union[List[ActionModel], None]:

        step_info = AgentStepInfo(number=self.state.n_steps, max_steps=self.conf.max_steps)
        last_step_msg = None
        if step_info and step_info.is_last_step():
            # Add last step warning if needed
            last_step_msg = HumanMessage(
                content='Now comes your last step. Use only the "done" action now. No other actions - so here your action sequence must have length 1.\n'
                        'If the task is not yet fully finished as requested by the user, set success in "done" to false! E.g. if not all steps are fully completed.\n'
                        'If the task is fully finished, set success in "done" to true.\n'
                        'Include everything you found out for the ultimate task in the done text.')
            logger.info('Last step finishing up')

        logger.info(f'[agent] 📍 Step {self.state.n_steps}')
        step_start_time = time.time()

        try:

            xml_content, base64_img = self.observation["dom_tree"], self.observation["image"]

            if xml_content is None:
                logger.error("[agent] ⚠ Failed to get UI state, stopping task")
                self.stop()
                return None

            # 记录状态
            self.state.last_result = (xml_content, base64_img if base64_img else "")

            # 使用LLM分析当前状态并决定下一步操作
            logger.info("[agent] 🤖 Analyzing current state with LLM...")
            # 界面XML:
            # {xml_content}
            a_step_msg = HumanMessage(content=[
                {
                    "type": "text",
                    "text": f"""
                        任务: {self.task}
                        当前步骤: {self.state.n_steps}
                        
                        请分析当前界面并决定下一步操作。请直接返回JSON格式的响应，不要包含任何其他文本或代码块标记。
                    """
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_img}"
                }
            ])

            messages = [SystemMessage(content=SYSTEM_PROMPT)]
            if last_step_msg:
                messages.append(last_step_msg)
            messages.append(a_step_msg)

            # 打印messages最近一条
            logger.info(f"[agent] VLM Input last message: {messages[-1]}")
            llm_result = None
            try:
                llm_result = self._do_policy(messages)

                if self.state.stopped or self.state.paused:
                    logger.info('Android agent paused after getting state')
                    return [ActionModel(tool_name=Tools.ANDROID.value, action_name="stop")]

                tool_action = llm_result.actions

                # 创建历史记录
                step_metadata = PolicyMetadata(
                    start_time=step_start_time,
                    end_time=time.time(),
                    number=self.state.n_steps,
                    input_tokens=1
                )

                history_item = AgentHistory(
                    result=[ActionResult(success=True)],
                    metadata=step_metadata,
                    content=xml_content,
                    base64_img=base64_img
                )
                self.history.history.append(history_item)

                # 保存历史
                if self.settings.save_history and self.settings.history_path:
                    self.history.save_to_file(self.settings.history_path)

                logger.info(f'📍 步骤 {self.state.n_steps} 执行完成')
                # 增加步数
                self.state.n_steps += 1
                self.state.consecutive_failures = 0
                return tool_action

            except Exception as e:
                logger.warning(traceback.format_exc())
                raise e
            finally:
                if llm_result:
                    self.trajectory.append((observation, info, llm_result))
                    metadata = PolicyMetadata(
                        number=self.state.n_steps,
                        start_time=step_start_time,
                        end_time=time.time(),
                        input_tokens=1
                    )
                    self._make_history_item(llm_result, observation, metadata)
                else:
                    logger.warning("no result to record!")

        except json.JSONDecodeError as e:
            logger.error("[agent] ❌ JSON parsing error")
            raise
        except Exception as e:
            logger.error(f"[agent] ❌ Action execution error: {str(e)}")
            raise

    def _do_policy(self, input_messages: list[BaseMessage]) -> AgentResult:
        response = self.llm.invoke(input_messages)  # 使用同步版本
        # 清理响应内容
        content = response.content
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        # 解析响应内容
        action_data = json.loads(content)
        brain_state = AgentBrain(**action_data["current_state"])

        # 记录状态信息
        logger.info(f"[agent] ⚠ Eval: {brain_state.evaluation_previous_goal}")
        logger.info(f"[agent] 🧠 Memory: {brain_state.memory}")
        logger.info(f"[agent] 🎯 Next goal: {brain_state.next_goal}")

        actions = action_data.get('action')
        result = []
        if not actions:
            actions = action_data.get("actions")

        # 打印actions
        logger.info(f"[agent] VLM Output actions: {actions}")
        for action in actions:
            action_type = action.get('type')
            if not action_type:
                logger.warning(f"Action missing type: {action}")
                continue

            params = {}
            if 'params' in action:
                params = action['params']
            if 'index' in action:
                params['index'] = action['index']

            action_model = ActionModel(
                tool_name=Tools.ANDROID.value,
                action_name=action_type,
                params=params
            )
            result.append(action_model)

        return AgentResult(current_state=brain_state, actions=result)

    def _make_history_item(self,
                           model_output: AgentResult | None,
                           state: Observation,
                           metadata: Optional[PolicyMetadata] = None) -> None:
        if isinstance(state, dict):
            # 如果是字典，转换为 Observation 对象
            state = Observation(**state)

        history_item = AgentHistory(
            model_output=model_output,
            result=state.action_result,
            metadata=metadata,
            content=state.dom_tree,  # 这个地方android和browser不一样
            base64_img=state.image
        )
        self.state.history.history.append(history_item)

    def pause(self) -> None:
        """Pause the agent"""
        logger.info('🔄 Pausing Agent')
        self.state.paused = True

    def resume(self) -> None:
        """Resume the agent"""
        logger.info('▶️ Agent resuming')
        self.state.paused = False

    def stop(self) -> None:
        """Stop the agent"""
        logger.info('⏹️ Agent stopping')
        self.state.stopped = True
