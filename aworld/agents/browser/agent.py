# coding: utf-8
# Copyright (c) 2025 inclusionAI.

import re
import time
import traceback
from typing import Dict, Any, Optional, List, Union

from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage, AIMessage
from pydantic import ValidationError

from aworld.core.agent.base import AgentFactory, BaseAgent, AgentResult
from aworld.agents.browser.memory import MessageManager, MessageManagerSettings
from aworld.agents.browser.prompts import SystemPrompt
from aworld.agents.browser.utils import convert_input_messages, extract_json_from_model_output
from aworld.agents.browser.common import AgentState, AgentStepInfo, AgentHistory, PolicyMetadata, AgentBrain
from aworld.config.conf import AgentConfig
from aworld.core.envs.tool_action import BrowserAction
from aworld.core.common import Observation, ActionModel, Tools, ToolActionInfo, Agents, ActionResult
from aworld.logs.util import logger


@AgentFactory.register(name=Agents.BROWSER.value, desc="browser agent")
class BrowserAgent(BaseAgent):
    def __init__(self, conf: AgentConfig, **kwargs):
        super(BrowserAgent, self).__init__(conf, **kwargs)
        self.state = AgentState()
        self.available_actions_desc = self._build_action_prompt()
        self.settings = conf.model_dump()
        if conf.llm_provider == 'openai':
            conf.llm_provider = 'chatopenai'

        # Initialize message manager
        self._message_manager = None
        self._init = False

    def reset(self, options: Dict[str, Any]):
        super(BrowserAgent, self).reset(options)
        self._message_manager = MessageManager(
            task=self.task,
            system_message=SystemPrompt(
                action_description=self.available_actions_desc,
                max_actions_per_step=self.settings.get('max_actions_per_step'),
            ).get_system_message(),
            settings=MessageManagerSettings(
                max_input_tokens=self.settings.get('max_input_tokens'),
                include_attributes=self.settings.get('include_attributes'),
                message_context=self.settings.get('message_context'),
                available_file_paths=self.settings.get('available_file_paths'),
            ),
            state=self.state.message_manager_state,
        )
        self._init = True

    def name(self) -> str:
        return Agents.BROWSER.value

    def _build_action_prompt(self) -> str:
        def _prompt(info: ToolActionInfo) -> str:
            s = f'{info.desc}: \n'
            s += '{' + str(info.name) + ': '
            if info.input_params:
                s += str({k: {"title": k, "type": v.type} for k, v in info.input_params.items()})
            s += '}'
            return s

        val = "\n".join([_prompt(v.value) for k, v in BrowserAction.__members__.items()])
        return val

    def add_new_task(self, new_task: str) -> None:
        self._message_manager.add_new_task(new_task)

    def policy(self,
               observation: Observation,
               info: Dict[str, Any] = None, **kwargs) -> Union[List[ActionModel], None]:
        start_time = time.time()

        if self._init is False:
            self.reset({"task": observation.content})

        step_info = AgentStepInfo(number=self.state.n_steps, max_steps=self.conf.max_steps)
        self._message_manager.add_state_message(observation, self.state.last_result, step_info,
                                                self.settings.get('use_vision'))
        self.state.last_result = observation.action_result

        if self.conf.max_steps <= self.state.n_steps:
            logger.info('Last step finishing up')
            human_msg = HumanMessage(
                content=[
                {
                    'type': 'text',
                    'text': f"""
                        Now comes your last step. Use only the "done" action now. No other actions - so here your action sequence must have length 1.
                        \nIf the task is not yet fully finished as requested by the user, set success in "done" to false! E.g. if not all steps are fully completed.
                        \nIf the task is fully finished, set success in "done" to true.
                        \nInclude everything you found out for the ultimate task in the done text.
                    """
                }
            ])
            self._message_manager._add_message_with_tokens(human_msg)

        logger.info(f'[agent] step {self.state.n_steps}')

        input_messages = self._message_manager.get_messages()
        tokens = self._message_manager.state.history.current_tokens
        llm_result = None
        logger.info(f"[agent] 🔍 Invoking LLM with {len(input_messages)} messages")
        logger.info("[agent] 📝 Messages sequence:")
        try:
            for i, msg in enumerate(input_messages):
                prefix = msg.type
                logger.info(f"[agent] Message {i + 1}: {prefix} ===================================")
                if isinstance(msg.content, list):
                    for item in msg.content:
                        if item.get('type') == 'text':
                            logger.info(f"[agent] Text content: {item.get('text')}")
                        elif item.get('type') == 'image_url':
                            # 只打印图片URL的前30个字符，避免打印整个base64
                            image_url = item.get('image_url', {}).get('url', '')
                            if image_url.startswith('data:image'):
                                logger.info(f"[agent] Image: [Base64 image data]")
                            else:
                                logger.info(f"[agent] Image URL: {image_url[:30]}...")
                else:
                    content = str(msg.content)
                    chunk_size = 500
                    for j in range(0, len(content), chunk_size):
                        chunk = content[j:j + chunk_size]
                        if j == 0:
                            logger.info(f"[agent] Content: {chunk}")
                        else:
                            logger.info(f"[agent] Content (continued): {chunk}")

                if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        logger.info(f"[agent] Tool call: {tool_call.get('name')} - ID: {tool_call.get('id')}")
                        args = str(tool_call.get('args', {}))[:100]
                        logger.info(f"[agent] Tool args: {args}...")

            llm_result = self._do_policy(input_messages)

            if not llm_result:
                logger.error("[agent] ❌ Failed to parse LLM response")
                return [ActionModel(tool_name=Tools.BROWSER.value, action_name="stop")]

            self.state.n_steps += 1

            self._message_manager._remove_last_state_message()

            if self.state.stopped or self.state.paused:
                logger.info('Browser gent paused after getting state')
                return [ActionModel(tool_name=Tools.BROWSER.value, action_name="stop")]

            tool_action = llm_result.actions

            self._message_manager.add_model_output(llm_result)
        except Exception as e:
            logger.warning(traceback.format_exc())
            # model call failed, remove last state message from history
            self._message_manager._remove_last_state_message()
            logger.error(f"[agent] ❌ Error parsing LLM response: {str(e)}")

            return [ActionModel(tool_name=Tools.BROWSER.value, action_name="stop")]
        finally:
            if llm_result:
                self.trajectory.append((observation, info, llm_result))
                metadata = PolicyMetadata(
                    number=self.state.n_steps,
                    start_time=start_time,
                    end_time=time.time(),
                    input_tokens=tokens,
                )
                self._make_history_item(llm_result, observation, observation.action_result, metadata)
            else:
                logger.warning("no result to record!")

        return tool_action

    def _do_policy(self, input_messages: list[BaseMessage]) -> AgentResult:
        THINK_TAGS = re.compile(r'<think>.*?</think>', re.DOTALL)

        def _remove_think_tags(text: str) -> str:
            """Remove think tags from text"""
            return re.sub(THINK_TAGS, '', text)

        input_messages = self._convert_input_messages(input_messages)
        output_message = None
        try:
            output_message = self.llm.invoke(input_messages)

            if not output_message or not output_message.content:
                logger.warning("[agent] LLM returned empty response")
                return AgentResult(current_state=AgentBrain(evaluation_previous_goal="", memory="", next_goal=""),
                                   actions=[ActionModel(tool_name=Tools.BROWSER.value, action_name="stop")])
        except:
            logger.error(f"[agent] Response content: {output_message}")

        if self.model_name == 'deepseek-reasoner':
            output_message.content = _remove_think_tags(output_message.content)
        try:
            parsed_json = extract_json_from_model_output(output_message.content)
            logger.info((f"llm response: {parsed_json}"))
            agent_brain = AgentBrain(**parsed_json['current_state'])
            actions = parsed_json.get('action')
            result = []
            if not actions:
                actions = parsed_json.get("actions")
            if not actions:
                logger.warning("agent not policy  an action.")
                return AgentResult(current_state=agent_brain,
                                   actions=[ActionModel(tool_name=Tools.BROWSER.value,
                                                        action_name="done")])

            for action in actions:
                if "action_name" in action:
                    action_name = action['action_name']
                    browser_action = BrowserAction.get_value_by_name(action_name)
                    if not browser_action:
                        logger.warning(f"Unsupported action: {action_name}")
                    action_model = ActionModel(tool_name=Tools.BROWSER.value,
                                               action_name=action_name,
                                               params=action.get('params', {}))
                    result.append(action_model)
                else:
                    for k, v in action.items():
                        browser_action = BrowserAction.get_value_by_name(k)
                        if not browser_action:
                            logger.warning(f"Unsupported action: {k}")

                        action_model = ActionModel(tool_name=Tools.BROWSER.value, action_name=k, params=v)
                        result.append(action_model)
            return AgentResult(current_state=agent_brain, actions=result)
        except (ValueError, ValidationError) as e:
            logger.warning(f'Failed to parse model output: {output_message} {str(e)}')
            raise ValueError('Could not parse response.')

    def _convert_input_messages(self, input_messages: list[BaseMessage]) -> list[BaseMessage]:
        """Convert input messages to the correct format"""
        if self.model_name == 'deepseek-reasoner' or self.model_name.startswith('deepseek-r1'):
            return convert_input_messages(input_messages, self.model_name)
        else:
            return input_messages

    def _make_history_item(self,
                           model_output: AgentResult | None,
                           state: Observation,
                           result: list[ActionResult],
                           metadata: Optional[PolicyMetadata] = None) -> None:
        content = ""
        if hasattr(state, 'dom_tree') and state.dom_tree is not None:
            if hasattr(state.dom_tree, 'element_tree'):
                content = state.dom_tree.element_tree.__repr__()
            else:
                content = str(state.dom_tree)

        history_item = AgentHistory(model_output=model_output,
                                    result=state.action_result,
                                    metadata=metadata,
                                    content=content,
                                    base64_img=state.image if hasattr(state, 'image') else None)

        self.state.history.history.append(history_item)
