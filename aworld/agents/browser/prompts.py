# coding: utf-8

import importlib.resources
from datetime import datetime
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from aworld.agents.common import AgentStepInfo
from aworld.core.common import Observation, ActionResult


class SystemPrompt:
    def __init__(self,
                 action_description: str,
                 max_actions_per_step: int = 10,
                 override_system_message: Optional[str] = None,
                 extend_system_message: Optional[str] = None):
        self.default_action_description = action_description
        self.max_actions_per_step = max_actions_per_step
        if override_system_message:
            prompt = override_system_message
        else:
            self._load_prompt_template()
            prompt = self.prompt_template.format(max_actions=self.max_actions_per_step)

        if extend_system_message:
            prompt += f'\n{extend_system_message}'

        self.system_message = SystemMessage(content=prompt)

    def _load_prompt_template(self) -> None:
        """Load the prompt template from the markdown file."""
        try:
            # This works both in development and when installed as a package
            with importlib.resources.files('aworld.agents.browser').joinpath('system_prompt.md').open('r') as f:
                self.prompt_template = f.read()
        except Exception as e:
            raise RuntimeError(f'Failed to load system prompt template: {e}')

    def get_system_message(self) -> SystemMessage:
        """
        Get the system prompt for the agent.

        Returns:
            SystemMessage: Formatted system prompt
        """
        return self.system_message


class AgentMessagePrompt:
    def __init__(
            self,
            state: Observation,
            result: Optional[List[ActionResult]] = None,
            include_attributes: list[str] = [],
            step_info: Optional[AgentStepInfo] = None,
    ):
        self.state = state
        self.result = result
        self.include_attributes = include_attributes
        self.step_info = step_info

    def get_user_message(self, use_vision: bool = True) -> HumanMessage:
        elements_text = self.state.dom_tree.element_tree.clickable_elements_to_string(
            include_attributes=self.include_attributes)

        pixels_above = self.state.info.get('pixels_above', 0)
        pixels_below = self.state.info.get('pixels_below', 0)

        if elements_text != '':
            if pixels_above > 0:
                elements_text = (
                    f'... {pixels_above} pixels above - scroll or extract content to see more ...\n{elements_text}'
                )
            else:
                elements_text = f'[Start of page]\n{elements_text}'
            if pixels_below > 0:
                elements_text = (
                    f'{elements_text}\n... {pixels_below} pixels below - scroll or extract content to see more ...'
                )
            else:
                elements_text = f'{elements_text}\n[End of page]'
        else:
            elements_text = 'empty page'

        if self.step_info:
            step_info_description = f'Current step: {self.step_info.number + 1}/{self.step_info.max_steps}'
        else:
            step_info_description = ''
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M')
        step_info_description += f'Current date and time: {time_str}'

        state_description = f"""
[Task history memory ends]
[Current state starts here]
The following is one-time information - if you need to remember it write it to memory:
Current url: {self.state.info.get("url")}
Interactive elements from top layer of the current page inside the viewport:
{elements_text}
{step_info_description}
"""

        if self.result:
            for i, result in enumerate(self.result):
                if result.content:
                    state_description += f'\nAction result {i + 1}/{len(self.result)}: {result.content}'
                if result.error:
                    # only use last line of error
                    error = result.error.split('\n')[-1]
                    state_description += f'\nAction error {i + 1}/{len(self.result)}: ...{error}'

        if self.state.image and use_vision == True:
            # Format message for vision model
            return HumanMessage(
                content=[
                    {'type': 'text', 'text': state_description},
                    {
                        'type': 'image_url',
                        'image_url': {'url': f'data:image/png;base64,{self.state.image}'},  # , 'detail': 'low'
                    },
                ]
            )

        return HumanMessage(content=state_description)


class PlannerPrompt(SystemPrompt):
    def get_system_message(self) -> SystemMessage:
        return SystemMessage(
            content="""You are a planning agent that helps break down tasks into smaller steps and reason about the current state.
Your role is to:
1. Analyze the current state and history
2. Evaluate progress towards the ultimate goal
3. Identify potential challenges or roadblocks
4. Suggest the next high-level steps to take

Inside your messages, there will be AI messages from different agents with different formats.

Your output format should be always a JSON object with the following fields:
{
    "state_analysis": "Brief analysis of the current state and what has been done so far",
    "progress_evaluation": "Evaluation of progress towards the ultimate goal (as percentage and description)",
    "challenges": "List any potential challenges or roadblocks",
    "next_steps": "List 2-3 concrete next steps to take",
    "reasoning": "Explain your reasoning for the suggested next steps"
}

Ignore the other AI messages output structures.

Keep your responses concise and focused on actionable insights."""
        )
