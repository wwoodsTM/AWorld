# coding: utf-8
# Copyright (c) 2025 inclusionAI.
import re
import os
import time
import traceback
from typing import Dict, Any, Optional, List, Union

from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, ToolMessage, SystemMessage
from pydantic import ValidationError

from aworld.config.common import Agents, Tools
from aworld.core.agent.base import AgentFactory, BaseAgent, AgentResult
from aworld.agents.browser.utils import convert_input_messages, extract_json_from_model_output, estimate_messages_tokens
from aworld.agents.browser.common import AgentState, AgentStepInfo, AgentHistory, PolicyMetadata, AgentBrain
from aworld.config.conf import AgentConfig
from aworld.core.common import Observation, ActionModel, ActionResult
from aworld.logs.util import logger
from aworld.agents.browser.prompts import AgentMessagePrompt
from langchain_openai import ChatOpenAI
from aworld.agents.browser.agent import Trajectory

from aworld.agents.travel.search_agent import SearchAgent
from aworld.agents.browser.agent import BrowserAgent
from aworld.agents.travel.write_agent import WriteAgent
from aworld.core.envs.tool import ToolFactory
from aworld.config import ToolConfig, load_config, wipe_secret_info
from aworld.virtual_environments.conf import BrowserToolConfig


PROMPT_TEMPLATE = """
You are an AI agent designed to automate tasks. Your goal is to accomplish the ultimate task following the rules.

# Input Format
Task
Previous steps

# Response Rules
1. RESPONSE FORMAT: You must ALWAYS respond with valid JSON in this exact format:
{"current_state": {"evaluation_previous_goal": "Success|Failed|Unknown - Analyze and check if the previous goals/actions are successful like intended by the task. Mention if something unexpected happened. Shortly state why/why not",
"memory": "Description of what has been done and what you need to remember. Be very specific. Count here ALWAYS how many times you have done something and how many remain. E.g. 0 out of 10 subtask completed Continue with abc and xyz",
"next_goal": "What needs to be done with the next immediate action"},
"action":[{{"one_action_name": {{// action-specific parameter}}}}, // ... more actions in sequence]}

2. ACTIONS: You can specify one actions in the list to be executed in sequence. 

3. REQUIREMENTS:
- If you want to extract some information, you can use search_agent gets related info and url, then you can use browser_agent extract info from specific url.
- If you want to search, you need use search_agent and give the specific task. 
- If you want to extract, you need use broswer_agent and give the task contains specific url. you can give two url once for browser agent, and tell browser agent only need extract from one url. if one url is invalid, use another url for replace.
- If you want to write, you need use write_agent and give the task and refer, the task needs be very detailed and contains all requirements.

4. Pipeline:
- If you have many information to search. you should excute search - extract loop many times.

5. TASK COMPLETION:
- Use the done action as the last action as soon as the ultimate task is complete
- Dont use "done" before you are done with everything the user asked you, except you reach the last step of max_steps. 
- If you reach your last step, use the done action even if the task is not fully finished. Provide all the information you have gathered so far. If the ultimate task is completly finished set success to true. If not everything the user asked for is completed set success in done to false!
- If you have to do something repeatedly for example the task says for "each", or "for all", or "x times", count always inside "memory" how many times you have done it and how many remain. Don't stop until you have completed like the task asked you. Only call done after the last step.
- Don't hallucinate actions
- Make sure you include everything you found out for the ultimate task in the done text parameter. Do not just say you are done, but include the requested information of the task. 

there are actions you can use：
[{'type': 'function', 'function': {'name': 'search_agent', 'description': 'Search the abstract and url from search api for the specific task.', 'parameters': {'type': 'object', 'properties': {'task': {'description': 'the search query input.', 'type': 'string'}}, 'required': ['query']}}}, {'type': 'function', 'function': {'name': 'browser_agent', 'description': 'use browser agent execute specific task like extract content from url.', 'parameters': {'type': 'object', 'properties': {'task': {'description': 'the task you want browser agent to do, if extract, please give extract goal and complete url', 'type': 'string'}, 'required': ['task']}}}, {'type': 'function', 'function': {'name': 'write_agent', 'description': 'Use write agent to write file about specific tasks.', 'parameters': {'type': 'object', 'properties': {'task': {'description': 'the specific writing task description.', 'type': 'string'}, 'refer': {'description': 'the related information write agent need refer.', 'type': 'string'}}, 'required': ['task', 'refer']}}}]
"""


@AgentFactory.register(name="travel_plan_agent", desc="travel plan agent")
class TravelPlanAgent(BaseAgent):
    def __init__(self, conf: AgentConfig, **kwargs):
        super(TravelPlanAgent, self).__init__(conf, **kwargs)
        self.state = AgentState()
        self.settings = conf.model_dump()
        if conf.llm_provider == 'openai':
            conf.llm_provider = 'chatopenai'

        # Note: Removed _message_manager initialization as it's no longer used
        # Initialize trajectory
        self.trajectory = Trajectory()
        self._init = False

    def reset(self, options: Dict[str, Any]):
        super(TravelPlanAgent, self).reset(options)

        # Reset trajectory
        self.trajectory = Trajectory()

        # Note: Removed _message_manager initialization as it's no longer used
        # _estimate_tokens_for_messages method now directly uses functions from utils.py

        self._init = True

    @property
    def llm(self):
        # lazy
        if self._llm is None:
            self._llm = ChatOpenAI(
                model_name="gpt-4o",
                openai_api_base="http://localhost:5000",
                openai_api_key="dummy-key",
            )
        return self._llm

    def name(self) -> str:
        return "travel_plan_agent"

    def _log_message_sequence(self, input_messages: List[BaseMessage]) -> None:
        """Log the sequence of messages for debugging purposes"""
        logger.info(f"[agent] 🔍 Invoking LLM with {len(input_messages)} messages")
        logger.info("[agent] 📝 Messages sequence:")
        for i, msg in enumerate(input_messages):
            prefix = msg.type
            logger.info(f"[agent] Message {i + 1}: {prefix} ===================================")
            if isinstance(msg.content, list):
                for item in msg.content:
                    if item.get('type') == 'text':
                        logger.info(f"[agent] Text content: {item.get('text')}")
                    elif item.get('type') == 'image_url':
                        # Only print the first 30 characters of image URL to avoid printing entire base64
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
                    args = str(tool_call.get('args', {}))[:1000]
                    logger.info(f"[agent] Tool args: {args}...")

    def policy(self,
               observation: Observation,
               info: Dict[str, Any] = None, **kwargs) -> Union[List[ActionModel], None]:
        start_time = time.time()

        if self._init is False:
            self.reset({"task": observation.content})

        # Save current observation to state for message construction
        self.state.last_result = observation.action_result

        if self.conf.max_steps <= self.state.n_steps:
            logger.info('Last step finishing up')

        logger.info(f'[agent] step {self.state.n_steps}')

        # Use the new method to build messages, passing the current observation
        input_messages = self.build_messages_from_trajectory_and_observation(observation=observation)

        # Note: Special message addition has been moved to build_messages_from_trajectory_and_observation

        # Estimate token count
        tokens = self._estimate_tokens_for_messages(input_messages)

        llm_result = None
        try:
            # Log the message sequence
            self._log_message_sequence(input_messages)

            llm_result = self._do_policy(input_messages)

            if not llm_result:
                logger.error("[agent] ❌ Failed to parse LLM response")
                return [ActionModel(tool_name=None, action_name="stop")]  ## tmp stop

            self.state.n_steps += 1

            # No longer need to remove the last state message
            # self._message_manager._remove_last_state_message()

            if self.state.stopped or self.state.paused:
                logger.info('Browser gent paused after getting state')
                return [ActionModel(tool_name=None, action_name="stop")] ## tmp stop

            tool_action = llm_result.actions

            # Add the current step to the trajectory
            self.trajectory.add_step(observation, info, llm_result)

        except Exception as e:
            logger.warning(traceback.format_exc())
            # No longer need to remove the last state message
            # self._message_manager._remove_last_state_message()
            logger.error(f"[agent] ❌ Error parsing LLM response: {str(e)}")

            # Create an AgentResult object with an empty actions list
            error_result = AgentResult(
                current_state=AgentBrain(
                    evaluation_previous_goal="Failed due to error",
                    memory=f"Error occurred: {str(e)}",
                    next_goal="Recover from error"
                ),
                actions=[]  # Empty actions list
            )

            # Add the error state to the trajectory
            self.trajectory.add_step(observation, info, error_result)

            return [ActionModel(tool_name=Tools.BROWSER.value, action_name="stop")]
        finally:
            if llm_result:
                # Remove duplicate trajectory addition, as it's already been added in the try block
                # self.trajectory.add_step(observation, info, llm_result) - Already executed in try block

                # Only keep the history_item creation part
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
            # print(input_messages)
            output_message = self.llm.invoke(input_messages)
            # print(output_message)
            if not output_message or not output_message.content:
                logger.warning("[agent] LLM returned empty response")
                # return AgentResult(current_state=AgentBrain(evaluation_previous_goal="", memory="", next_goal=""),
                #                    actions=[ActionModel(tool_name=Tools.BROWSER.value, action_name="stop")])
                return AgentResult(current_state=AgentBrain(evaluation_previous_goal="", memory="", next_goal=""),
                                   actions=[ActionModel(tool_name=None, action_name="stop")])
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
                logger.warning("agent not policy an action.")
                return AgentResult(current_state=agent_brain,
                                   actions=[ActionModel(tool_name=None,
                                                        action_name="done")])

            for action in actions:
                if "action_name" in action:
                    action_name = action['action_name']
                    # browser_action = BrowserAction.get_value_by_name(action_name)
                    # if not browser_action:
                    if action_name not in ("", "", "", ""):
                        logger.warning(f"Unsupported action: {action_name}")
                    action_model = ActionModel(
                                               agent_name=action_name,
                                               params=action.get('params', {}))
                    result.append(action_model)
                else:
                    for k, v in action.items():
                        # browser_action = BrowserAction.get_value_by_name(k)
                        # if not browser_action:
                        #     logger.warning(f"Unsupported action: {k}")
                        action_model = ActionModel(agent_name=k, params=v)
                        # action_model = ActionModel(tool_name=Tools.BROWSER.value, action_name=k, params=v)
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

    def build_messages_from_trajectory_and_observation(self, observation: Optional[Observation] = None) -> List[
        BaseMessage]:
        """
        Build complete message history from trajectory and current observation

        Args:
            observation: Current observation object, if None current observation won't be added
        """
        messages = []

        system_prompt = PROMPT_TEMPLATE + AGENT_CAN_USE
        # Add system message
        system_message = SystemMessage(content=system_prompt)
        if isinstance(system_message, tuple):
            system_message = system_message[0]
        messages.append(system_message)

        # Add task context (if any)
        if self.settings.get('message_context'):
            context_message = HumanMessage(content='Context for the task' + self.settings.message_context)
            messages.append(context_message)

        # Add task message
        task_message = HumanMessage(
            content=f'Your ultimate task is: """{self.task}""". If you achieved your ultimate task, stop everything and use the done action in the next step to complete the task. If not, continue as usual.'
        )
        messages.append(task_message)

        # Add example output
        placeholder_message = HumanMessage(content='Example output:')
        messages.append(placeholder_message)

        # Add example tool call
        tool_calls = [
            {
                'name': 'AgentOutput',
                'args': {
                    'current_state': {
                        'evaluation_previous_goal': 'Success - I completed search and gets the url',
                        'memory': 'search compeleted and gets url',
                        'next_goal': 'extract information from the related url',
                    },
                    'action': [{'browser_agent': {"task":"goto xxx(url) to extract content about xxx."}}],
                },
                'id': '1',
                'type': 'tool_call',
            }
        ]
        example_tool_call = AIMessage(
            content='',
            tool_calls=tool_calls,
        )
        messages.append(example_tool_call)

        # Add first tool message with "Plan agent started" content
        messages.append(ToolMessage(content='Plan agent started', tool_call_id='1'))

        # Add task history marker
        messages.append(HumanMessage(content='[Your task history memory starts here]'))

        # Add available file paths (if any)
        if self.settings.get('available_file_paths'):
            filepaths_msg = HumanMessage(
                content=f'Here are file paths you can use: {self.settings.get("available_file_paths")}')
            messages.append(filepaths_msg)

        # Add messages from the history trajectory
        for obs, info, result in self.trajectory.get_history():
            # Add observation message
            step_info = AgentStepInfo(number=len(messages) - 7, max_steps=self.conf.max_steps)

            # Build observation message
            if hasattr(obs, 'dom_tree') and obs.dom_tree:
                state_message = AgentMessagePrompt(
                    obs,
                    obs.action_result,
                    include_attributes=self.settings.get('include_attributes'),
                    step_info=step_info,
                ).get_user_message(True)
                messages.append(state_message)
            elif obs.content:
                messages.append(HumanMessage(content=str(obs.content)))

            # Add agent response
            if result:
                # Create AI message
                output_data = result.model_dump(mode='json', exclude_unset=True)
                output_data["action"] = [{action.action_name: action.params} for action in result.actions]
                if "actions" in output_data:
                    del output_data["actions"]

                # Calculate tool_id based on trajectory history. If no actions yet, start with ID 1
                tool_id = 1 if len(self.trajectory.get_history()) == 0 else len(self.trajectory.get_history()) + 1
                tool_calls = [
                    {
                        'name': 'AgentOutput',
                        'args': output_data,
                        'id': str(tool_id),
                        'type': 'tool_call',
                    }
                ]

                ai_message = AIMessage(
                    content='',
                    tool_calls=tool_calls,
                )
                messages.append(ai_message)

                # Add empty tool message after each AIMessage
                messages.append(ToolMessage(content='', tool_call_id=str(tool_id)))

        # Add current observation - using the passed observation parameter instead of self.state.current_observation
        if observation:
            step_info = AgentStepInfo(number=self.state.n_steps, max_steps=self.conf.max_steps)
            if hasattr(observation, 'dom_tree') and observation.dom_tree:
                state_message = AgentMessagePrompt(
                    observation,
                    self.state.last_result,
                    include_attributes=self.settings.get('include_attributes'),
                    step_info=step_info,
                ).get_user_message(self.settings.get('use_vision'))
                messages.append(state_message)
            if observation.content:
                messages.append(HumanMessage(content=observation.content))

        # Add special message for the last step
        # Note: Moved here from policy method to centralize all message building logic
        if self.conf.max_steps <= self.state.n_steps:
            last_step_message = f"""
                   Now comes your last step. Use only the "done" action now. No other actions - so here your action sequence must have length 1.
                   \nIf the task is not yet fully finished as requested by the user, set success in "done" to false! E.g. if not all steps are fully completed.
                   \nIf the task is fully finished, set success in "done" to true.
                   \nInclude everything you found out for the ultimate task in the done text.
               """
            messages.append(HumanMessage(content=[{'type': 'text', 'text': last_step_message}]))

        return messages

    def _estimate_tokens_for_messages(self, messages: List[BaseMessage]) -> int:
        """Roughly estimate token count for message list"""
        # Note: Using estimate_messages_tokens function from utils.py instead of calling _message_manager
        # This decouples the dependency on MessageManager
        return estimate_messages_tokens(
            messages,
            image_tokens=self.settings.get('image_tokens', 800),
            estimated_characters_per_token=self.settings.get('estimated_characters_per_token', 3)
        )


if __name__ == '__main__':
    agentConfig = AgentConfig(
        llm_provider="chatopenai",
        llm_model_name="gpt-4o",
        llm_base_url="http://localhost:5000",
        llm_api_key="dummy-key",
        max_steps=100,
    )

    travelPlanAgent = TravelPlanAgent(agentConfig)

    search_agent = SearchAgent(agentConfig)
    browser_agent = BrowserAgent(agentConfig)
    write_agent = WriteAgent(agentConfig)
    browser_tool_config = BrowserToolConfig(width=800, height=1100, keep_browser_open=True)
    browser_tool = ToolFactory(Tools.BROWSER.value, browser_tool_config)

    goal = (
        "I need a 7-day Japan itinerary from April 2 to April 8 2025, departing from Hangzhou, We want to see beautiful cherry blossoms and experience traditional Japanese culture (kendo, tea ceremonies, Zen meditation). We would like to taste matcha in Uji and enjoy the hot springs in Kobe. I am planning to propose during this trip, so I need a special location recommendation. Please provide a detailed itinerary and create a simple HTML travel handbook that includes a 7-day Japan itinerary, an updated cherry blossom table, attraction descriptions, essential Japanese phrases, and travel tips for us to reference throughout our journey."
        "you need search and extract different info 3 times, and then write, at last use browser agent goto the html url and then, complete the task.")
    observation = Observation(content=goal, action_result=[ActionResult(content='start', keep=True)])

    step = 0
    extract_memorys = []

    while True:
        policy = travelPlanAgent.policy(observation=observation)
        print(policy)

        if policy[0].agent_name == "search_agent":
            ## mock
            # policy_info = [{"result_id": 1, "title": "Japan Cherry Blossom Forecast 2025 - January 23rd Update : r ...", "description": "Jan 23, 2025 ... Tokyo's full bloom is still forecasted to occur on March 31, while Kyoto's is pushed back by a day to April 5!", "long_description": "Posted by u/Markotan - 72 votes and 44 comments", "url": "https://www.reddit.com/r/JapanTravelTips/comments/1i8fixo/japan_cherry_blossom_forecast_2025_january_23rd/"}, {"result_id": 2, "title": "Release of 2025 Cherry Blossom Forecast (10th forecast) | \u30cb\u30e5\u30fc\u30b9 ...", "description": "On March 27, 2025, JMC released its 10th forecast of the dates when cherry blossoms will start to flower (kaika) and reach full bloom (mankai).", "long_description": "Earth Communication Provider\u3068\u3057\u3066\u3001\u6c17\u8c61\u4e88\u6e2c\u3001\u5927\u6c17\u74b0\u5883\u8abf\u67fb\u3001\u30b7\u30df\u30e5\u30ec\u30fc\u30b7\u30e7\u30f3\u306a\u3069\u69d8\u3005\u306a\u6280\u8853\u3092\u3082\u3063\u3066\u3001\u5168\u56fd\u306b\u5e45\u5e83\u304f\u6d3b\u52d5\u3092\u884c\u3063\u3066\u3044\u307e\u3059\u300224\u6642\u9593365\u65e5\u4f11\u3080\u3053\u3068\u306a\u304f\u7a7a\u30fb\u6d77\u30fb\u5927\u5730\u306e\u5909\u5316\u306b\u6ce8\u610f\u3092\u6255\u3044\u3001\u4eba\u3005\u306e\u304b\u3051\u304c\u3048\u306e\u306a\u3044\u547d\u3068\u5927\u5207\u306a\u8ca1\u7523\u306e\u4fdd\u5168\u3001\u307e\u305f\u3001\u81ea\u7136\u30a8\u30cd\u30eb\u30ae\u30fc\u6d3b\u7528\u578b\u793e\u4f1a\u306e\u63a8\u9032\u3092\u306f\u3058\u3081\u3001\u5730\u7403\u74b0\u5883\u3084\u793e\u4f1a\u306e\u5b89\u5168\u30fb\u5b89\u5fc3\u306b\u8ca2\u732e\u3057\u3066\u307e\u3044\u308a\u307e\u3059\u3002", "url": "https://n-kishou.com/corp/news-contents/sakura/?lang=en"}, {"result_id": 3, "title": "Spring in Japan: Cherry Blossom Forecast 2025", "description": "Spring in Japan: Cherry Blossom Forecast 2025. Where & when to enjoy sakura in Japan. Aomori. Springtime in Japan is a tableau of dreamlike scenes.", "long_description": "N/A", "url": "https://www.japan.travel/en/see-and-do/cherry-blossom-forecast-2025/"}, {"result_id": 4, "title": "Japan Cherry Blossom Forecast 2025 - February 27th Update : r ...", "description": "Feb 28, 2025 ... Predictions for Tokyo's full bloom is April 1, one day earlier compared to the 5th forecast (April 2). Compared to the original forecast (March 31), it is one\u00a0...", "long_description": "Posted by u/Markotan - 66 votes and 58 comments", "url": "https://www.reddit.com/r/JapanTravelTips/comments/1izypka/japan_cherry_blossom_forecast_2025_february_27th/"}, {"result_id": 5, "title": "2025 Cherry Blossom Forecast", "description": "A forecast of the cherry blossom blooming season in Japan for 2025 ... April, before swiftly moving into northern Japan during mid to late April.", "long_description": "A forecast of the cherry blossom blooming season in Japan for 2025.", "url": "https://www.japan-guide.com/sakura/"}, {"result_id": 6, "title": "Here's the official Japan cherry blossom forecast for 2025 \u2013 updated ...", "description": "Mar 21, 2025 ... In Kyoto, the blossoms are expected to open up on March 28 with full bloom by April 6. Nearby, Osaka is also looking at March 29 for its initial\u00a0...", "long_description": "Get an idea of when you can expect to see this year\u2019s blooms across Japan in Tokyo, Osaka, Kyoto, Sapporo and more", "url": "https://www.timeout.com/tokyo/news/heres-the-official-japan-cherry-blossom-forecast-for-2025-011725"}, "If the search result does not contain the information you want, please make reflection on your query: what went well, what didn't, then refine your search plan."]
            #
            # observation = Observation(content = str(policy_info), action_result=[ActionResult(content=str(policy_info), keep=True)])
            goal = policy[0].params['task']
            observation = Observation(content=goal)
            while True:
                policy = search_agent.policy(observation=observation)
                print(policy)

                if policy[0].tool_name == '[done]':
                    # print(policy[0].policy_info)
                    # observation = Observation(action_result=[ActionResult(content=policy[0].policy_info, keep=True)])
                    observation = Observation(content=str(policy[0].policy_info),
                                              action_result=[ActionResult(content=str(policy[0].policy_info), keep=True)])
                    break

                tool = ToolFactory(policy[0].tool_name, conf=load_config(f"{policy[0].tool_name}.yaml"))

                observation, reward, terminated, _, info = tool.step(policy)

                print(observation)

        elif policy[0].agent_name == "browser_agent":
            task = policy[0].params['task']
            goal = policy
            observation, info = browser_tool.reset()
            results=[]
            observation.content = task
            extract_flag = False
            browser_agent.reset({"task": observation.content})
            while True:
                policy = browser_agent.policy(observation=observation)
                print(policy)
                if "done" in policy[0].action_name:
                    result = str(policy[0].policy_info) + "extract_result:" + str(results)
                    observation = Observation(content = result, action_result=[ActionResult(content=result, keep=True)])
                    break
                if "extract" in policy[0].action_name:
                    extract_flag = True
                observation, reward, terminated, _, info = browser_tool.step(policy)
                if extract_flag:
                    extract_flag = False
                    results.append(observation.action_result)
                    extract_memorys.append(observation.action_result)
                print(observation)

        elif policy[0].agent_name == "write_agent":

            observation.content = {
                "task": policy[0].params['task'],
                "refer": str(extract_memorys)
            }

            while True:
                policy = write_agent.policy(observation=observation)

                if policy[0].tool_name == '[done]':
                    observation = Observation(content=str(policy[0].policy_info), action_result=[ActionResult(content=str(policy[0].policy_info), keep=True)])
                    break
                # policy[0].params['information'] += str(extract_memorys)
                tool = ToolFactory(policy[0].tool_name, conf=load_config)
                observation, reward, terminated, _, info = tool.step(policy)
                print(observation)
        elif "done" in policy[0].agent_name:
            print("now is done.")
            logger.info(policy[0].params['text'])
            logger.info("task is done.")
            break
        else:
            print("invalid agent name.")
            observation = Observation(content="invalid agent name, please try again", action_result=[ActionResult(content="invalid agent name, please try again.", keep=True)])
            continue

