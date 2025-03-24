# coding: utf-8
# Copyright (c) 2025 inclusionAI.

import re
import time
import traceback
from typing import Dict, Any, Optional, List, Union

from langchain_core.messages import HumanMessage, BaseMessage
from pydantic import ValidationError

from aworld.core.agent.base import AgentFactory, BaseAgent, AgentResult
from aworld.agents.browser.memory import MessageManager, MessageManagerSettings
from aworld.agents.browser.prompts import SystemPrompt
from aworld.agents.browser.utils import convert_input_messages, extract_json_from_model_output
from aworld.agents.browser.common import AgentState, AgentStepInfo, AgentHistory, PolicyMetadata, AgentBrain
from aworld.config.conf import AgentConfig
from aworld.core.envs.tool_action import BrowserAction
from aworld.core.common import Observation, ActionModel, Tools, ToolActionInfo, Agents
from aworld.logs.util import logger
from aworld.models.llm import get_llm_model


@AgentFactory.register(name=Agents.BROWSER.value, desc="browser agent")
class BrowserAgent(BaseAgent):
    def __init__(self, conf: AgentConfig, **kwargs):
        super(BrowserAgent, self).__init__(conf, **kwargs)
        self._build_prompt()
        self.state = AgentState()
        self.available_actions_desc = self._build_action_prompt()
        self.settings = conf.model_dump()
        if self.conf.llm_provider == 'openai':
            self.conf.llm_provider = 'chatopenai'

        # Initialize message manager
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

    def _build_prompt(self):
        pass

    def add_new_task(self, new_task: str) -> None:
        self._message_manager.add_new_task(new_task)

    def policy(self,
               observation: Observation,
               info: Dict[str, Any] = None, **kwargs) -> Union[List[ActionModel], None]:
        start_time = time.time()
        step_info = AgentStepInfo(number=self.state.n_steps, max_steps=self.conf.max_steps)
        self._message_manager.add_state_message(observation, self.state.last_result, step_info,
                                                self.settings.get('use_vision'))

        if step_info and step_info.is_last_step():
            # Add last step warning if needed
            msg = 'Now comes your last step. Use only the "done" action now. No other actions - so here your action sequence must have length 1.'
            msg += '\nIf the task is not yet fully finished as requested by the user, set success in "done" to false! E.g. if not all steps are fully completed.'
            msg += '\nIf the task is fully finished, set success in "done" to true.'
            msg += '\nInclude everything you found out for the ultimate task in the done text.'
            logger.info('Last step finishing up')
            self._message_manager._add_message_with_tokens(HumanMessage(content=msg))
            # self.AgentOutput = self.DoneAgentOutput

        input_messages = self._message_manager.get_messages()
        tokens = self._message_manager.state.history.current_tokens
        llm_result = None
        try:
            llm_result = self._do_policy(input_messages)

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
            raise e
        finally:
            if llm_result:
                self.trajectory.append((observation, info, llm_result))
                metadata = PolicyMetadata(
                    number=self.state.n_steps,
                    start_time=start_time,
                    end_time=time.time(),
                    input_tokens=tokens,
                )
                self._make_history_item(llm_result, observation, metadata)
            else:
                logger.warning("no result to record!")

        return tool_action

    def _do_policy(self, input_messages: list[BaseMessage]) -> AgentResult:
        THINK_TAGS = re.compile(r'<think>.*?</think>', re.DOTALL)

        def _remove_think_tags(text: str) -> str:
            """Remove think tags from text"""
            return re.sub(THINK_TAGS, '', text)

        input_messages = self._convert_input_messages(input_messages)
        output_message = self.llm.invoke(input_messages)

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
                           metadata: Optional[PolicyMetadata] = None) -> None:
        history_item = AgentHistory(model_output=model_output,
                                    result=state.action_result,
                                    metadata=metadata,
                                    content=state.dom_tree.element_tree.__repr__(),
                                    base64_img=state.image)

        self.state.history.history.append(history_item)
