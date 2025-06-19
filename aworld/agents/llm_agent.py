# coding: utf-8
# Copyright (c) 2025 inclusionAI.
import abc
import json
import time
import traceback
import uuid
from collections import OrderedDict
from typing import AsyncGenerator, Dict, Any, List, Union, Callable

import aworld.trace as trace
from aworld.config import ToolConfig
from aworld.config.conf import AgentConfig, ConfigDict, ContextRuleConfig
from aworld.core.agent.agent_desc import get_agent_desc
from aworld.core.agent.base import BaseAgent, AgentResult, is_agent_by_name, is_agent
from aworld.core.common import Observation, ActionModel
from aworld.core.context.base import AgentContext
from aworld.core.context.base import Context
from aworld.core.context.base import ContextUsage
from aworld.core.contextprocessor.context_processor import ContextProcessor
from aworld.core.event.base import Message, ToolMessage, Constants
from aworld.core.event.event_bus import InMemoryEventbus
from aworld.core.memory import MemoryItem
from aworld.core.tool.base import ToolFactory, AsyncTool, Tool
from aworld.core.tool.tool_desc import get_tool_desc
from aworld.logs.util import logger, color_log, Color, trace_logger
from aworld.mcp_client.utils import mcp_tool_desc_transform
from aworld.memory.main import Memory
from aworld.models.llm import get_llm_model, call_llm_model, acall_llm_model, acall_llm_model_stream
from aworld.models.model_response import ModelResponse, ToolCall
from aworld.models.utils import tool_desc_transform, agent_desc_transform
from aworld.output import Outputs
from aworld.output.base import StepOutput, MessageOutput
from aworld.runners.hook.hook_factory import HookFactory
from aworld.runners.hook.hooks import HookPoint, PreLLMCallHook
from aworld.utils.common import convert_to_snake, sync_exec, nest_dict_counter


class Agent(BaseAgent[Observation, List[ActionModel]]):
    """Basic agent for unified protocol within the framework."""

    def __init__(self,
                 conf: Union[Dict[str, Any], ConfigDict, AgentConfig],
                 resp_parse_func: Callable[..., Any] = None,
                 **kwargs):
        """A api class implementation of agent, using the `Observation` and `List[ActionModel]` protocols.

        Args:
            conf: Agent config, supported AgentConfig, ConfigDict or dict.
            resp_parse_func: Response parse function for the agent standard output, transform llm response.
        """
        super(Agent, self).__init__(conf, **kwargs)
        conf = self.conf
        self.model_name = conf.llm_config.llm_model_name if conf.llm_config.llm_model_name else conf.llm_model_name
        self._llm = None
        self.memory = Memory.from_config(
            {"memory_store": kwargs.pop("memory_store") if kwargs.get("memory_store") else "inmemory"})
        self.system_prompt: str = kwargs.pop("system_prompt") if kwargs.get("system_prompt") else conf.system_prompt
        self.agent_prompt: str = kwargs.get("agent_prompt") if kwargs.get("agent_prompt") else conf.agent_prompt

        self.event_driven = kwargs.pop('event_driven', conf.get('event_driven', False))
        self.handler: Callable[..., Any] = kwargs.get('handler')

        self.need_reset = kwargs.get('need_reset') if kwargs.get('need_reset') else conf.need_reset
        # whether to keep contextual information, False means keep, True means reset in every step by the agent call
        self.step_reset = kwargs.get('step_reset') if kwargs.get('step_reset') else True
        # tool_name: [tool_action1, tool_action2, ...]
        self.black_tool_actions: Dict[str, List[str]] = kwargs.get("black_tool_actions") if kwargs.get(
            "black_tool_actions") else conf.get('black_tool_actions', {})
        self.resp_parse_func = resp_parse_func if resp_parse_func else self.response_parse
        self.history_messages = kwargs.get("history_messages") if kwargs.get("history_messages") else 100
        self.use_tools_in_prompt = kwargs.get('use_tools_in_prompt', conf.use_tools_in_prompt)
        # init agent context
        context_rule = kwargs.get("context_rule") if kwargs.get("context_rule") else conf.context_rule
        self.update_current_agent_context(context_rule)
        self.tools_instances = {}
        self.tools_conf = {}

    def reset(self, options: Dict[str, Any]):
        super().reset(options)
        self.memory = Memory.from_config(
            {"memory_store": options.pop("memory_store") if options.get("memory_store") else "inmemory"})

    def set_tools_instances(self, tools, tools_conf):
        self.tools_instances = tools
        self.tools_conf = tools_conf

    @property
    def llm(self):
        # lazy
        if self._llm is None:
            llm_config = self.conf.llm_config or None
            conf = llm_config if llm_config and (
                    llm_config.llm_provider or llm_config.llm_base_url or llm_config.llm_api_key or llm_config.llm_model_name) else self.conf
            self._llm = get_llm_model(conf)
        return self._llm

    def _env_tool(self):
        """Description of agent as tool."""
        return tool_desc_transform(get_tool_desc(),
                                   tools=self.tool_names if self.tool_names else [],
                                   black_tool_actions=self.black_tool_actions)

    def _handoffs_agent_as_tool(self):
        """Description of agent as tool."""
        return agent_desc_transform(get_agent_desc(),
                                    agents=self.handoffs if self.handoffs else [])

    def _mcp_is_tool(self):
        """Description of mcp servers are tools."""
        try:
            return sync_exec(mcp_tool_desc_transform, self.mcp_servers, self.mcp_config)
        except Exception as e:
            logger.error(f"mcp_is_tool error: {e}")
            return []

    def desc_transform(self):
        """Transform of descriptions of supported tools, agents, and MCP servers in the framework to support function calls of LLM."""

        # Stateless tool
        self.tools = self._env_tool()
        # Agents as tool
        self.tools.extend(self._handoffs_agent_as_tool())
        # MCP servers are tools
        self.tools.extend(self._mcp_is_tool())
        return self.tools

    async def async_desc_transform(self):
        """Transform of descriptions of supported tools, agents, and MCP servers in the framework to support function calls of LLM."""

        # Stateless tool
        self.tools = self._env_tool()
        # Agents as tool
        self.tools.extend(self._handoffs_agent_as_tool())
        # MCP servers are tools
        # todo sandbox
        if self.sandbox:
            sand_box = self.sandbox
            mcp_tools = await sand_box.mcpservers.list_tools()
            self.tools.extend(mcp_tools)
        else:
            self.tools.extend(await mcp_tool_desc_transform(self.mcp_servers, self.mcp_config))

    def _messages_transform(
            self,
            observation: Observation,
            sys_prompt: str = None,
            agent_prompt: str = None,
    ):
        messages = []
        if sys_prompt:
            messages.append({'role': 'system', 'content': sys_prompt if not self.use_tools_in_prompt else sys_prompt.format(
                tool_list=self.tools)})

        content = observation.content
        if agent_prompt and '{task}' in agent_prompt:
            content = agent_prompt.format(task=observation.content)

        cur_msg = {'role': 'user', 'content': content}
        # query from memory,
        # histories = self.memory.get_last_n(self.history_messages, filter={"session_id": self.context.session_id})
        histories = self.memory.get_last_n(self.history_messages)
        messages.extend(histories)

        action_results = observation.action_result
        if action_results:
            for action_result in action_results:
                cur_msg['role'] = 'tool'
                cur_msg['tool_call_id'] = action_result.tool_id

        agent_info = self.context.context_info.get(self.name())
        if (self.use_tools_in_prompt and "is_use_tool_prompt" in agent_info and "tool_calls"
                in agent_info and agent_prompt):
            cur_msg['content'] = agent_prompt.format(action_list=agent_info["tool_calls"],
                                                     result=content)

        if observation.images:
            urls = [{'type': 'text', 'text': content}]
            for image_url in observation.images:
                urls.append({'type': 'image_url', 'image_url': {"url": image_url}})

            cur_msg['content'] = urls
        messages.append(cur_msg)
        return messages

    def messages_transform(self,
                           content: str,
                           image_urls: List[str] = None,
                           sys_prompt: str = None,
                           agent_prompt: str = None,
                           **kwargs):
        """Transform the original content to LLM messages of native format.

        Args:
            content: User content.
            image_urls: List of images encoded using base64.
            sys_prompt: Agent system prompt.
            max_step: The maximum list length obtained from memory.
        Returns:
            Message list for LLM.
        """
        messages = []
        if sys_prompt:
            messages.append({'role': 'system', 'content': sys_prompt if not self.use_tools_in_prompt else sys_prompt.format(
                tool_list=self.tools)})

        histories = self.memory.get_last_n(self.history_messages)
        user_content = content
        if not histories and agent_prompt and '{task}' in agent_prompt:
            user_content = agent_prompt.format(task=content)

        cur_msg = {'role': 'user', 'content': user_content}
        # query from memory,
        # histories = self.memory.get_last_n(self.history_messages, filter={"session_id": self.context.session_id})

        if histories:
            # default use the first tool call
            for history in histories:
                if not self.use_tools_in_prompt and "tool_calls" in history.metadata and history.metadata['tool_calls']:
                    messages.append({'role': history.metadata['role'], 'content': history.content,
                                     'tool_calls': [history.metadata["tool_calls"][0]]})
                else:
                    messages.append({'role': history.metadata['role'], 'content': history.content,
                                     "tool_call_id": history.metadata.get("tool_call_id")})

            if not self.use_tools_in_prompt and "tool_calls" in histories[-1].metadata and histories[-1].metadata[
                'tool_calls']:
                tool_id = histories[-1].metadata["tool_calls"][0].id
                if tool_id:
                    cur_msg['role'] = 'tool'
                    cur_msg['tool_call_id'] = tool_id
            if self.use_tools_in_prompt and "is_use_tool_prompt" in histories[-1].metadata and "tool_calls" in \
                    histories[-1].metadata and agent_prompt:
                cur_msg['content'] = agent_prompt.format(action_list=histories[-1].metadata["tool_calls"],
                                                         result=content)

        if image_urls:
            urls = [{'type': 'text', 'text': content}]
            for image_url in image_urls:
                urls.append({'type': 'image_url', 'image_url': {"url": image_url}})

            cur_msg['content'] = urls
        messages.append(cur_msg)
        return messages

    def use_tool_list(self, resp: ModelResponse) -> List[Dict[str, Any]]:
        tool_list = []
        try:
            if resp and hasattr(resp, 'content') and resp.content:
                content = resp.content.strip()
            else:
                return tool_list
            content = content.replace('\n', '').replace('\r', '')
            response_json = json.loads(content)
            if "use_tool_list" in response_json:
                use_tool_list = response_json["use_tool_list"]
                if use_tool_list:
                    for use_tool in use_tool_list:
                        tool_name = use_tool["tool"]
                        arguments = use_tool["arguments"]
                        if tool_name and arguments:
                            tool_list.append(use_tool)

            return tool_list
        except Exception as e:
            logger.debug(f"tool_parse error, content: {resp.content}, \nerror msg: {e}")
            return tool_list

    def response_parse(self, resp: ModelResponse) -> AgentResult:
        """Default parse response by LLM."""
        results = []
        if not resp:
            logger.warning("LLM no valid response!")
            return AgentResult(actions=[], current_state=None)

        use_tool_list = self.use_tool_list(resp)
        is_call_tool = False
        content = '' if resp.content is None else resp.content
        if resp.tool_calls:
            is_call_tool = True
            for tool_call in resp.tool_calls:
                full_name: str = tool_call.function.name
                if not full_name:
                    logger.warning("tool call response no tool name.")
                    continue
                try:
                    params = json.loads(tool_call.function.arguments)
                except:
                    logger.warning(f"{tool_call.function.arguments} parse to json fail.")
                    params = {}
                # format in framework
                names = full_name.split("__")
                tool_name = names[0]
                if is_agent_by_name(tool_name):
                    param_info = params.get('content', "") + ' ' + params.get('info', '')
                    results.append(ActionModel(tool_name=tool_name,
                                               tool_id=tool_call.id,
                                               agent_name=self.name(),
                                               params=params,
                                               policy_info=content + param_info))
                else:
                    action_name = '__'.join(names[1:]) if len(names) > 1 else ''
                    results.append(ActionModel(tool_name=tool_name,
                                               tool_id=tool_call.id,
                                               action_name=action_name,
                                               agent_name=self.name(),
                                               params=params,
                                               policy_info=content))
        elif use_tool_list and len(use_tool_list) > 0:
            is_call_tool = True
            for use_tool in use_tool_list:
                full_name = use_tool["tool"]
                if not full_name:
                    logger.warning("tool call response no tool name.")
                    continue
                params = use_tool["arguments"]
                if not params:
                    logger.warning("tool call response no tool params.")
                    continue
                names = full_name.split("__")
                tool_name = names[0]
                if is_agent_by_name(tool_name):
                    param_info = params.get('content', "") + ' ' + params.get('info', '')
                    results.append(ActionModel(tool_name=tool_name,
                                               tool_id=use_tool.get('id'),
                                               agent_name=self.name(),
                                               params=params,
                                               policy_info=content + param_info))
                else:
                    action_name = '__'.join(names[1:]) if len(names) > 1 else ''
                    results.append(ActionModel(tool_name=tool_name,
                                               tool_id=use_tool.get('id'),
                                               action_name=action_name,
                                               agent_name=self.name(),
                                               params=params,
                                               policy_info=content))
        else:
            if content:
                content = content.replace("```json", "").replace("```", "")
            # no tool call, agent name is itself.
            results.append(ActionModel(agent_name=self.name(), policy_info=content))
        return AgentResult(actions=results, current_state=None, is_call_tool=is_call_tool)

    def _log_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Log the sequence of messages for debugging purposes"""
        logger.info(f"[agent] Invoking LLM with {len(messages)} messages:")
        for i, msg in enumerate(messages):
            prefix = msg.get('role')
            logger.info(f"[agent] Message {i + 1}: {prefix} ===================================")
            if isinstance(msg['content'], list):
                for item in msg['content']:
                    if item.get('type') == 'text':
                        logger.info(f"[agent] Text content: {item.get('text')}")
                    elif item.get('type') == 'image_url':
                        image_url = item.get('image_url', {}).get('url', '')
                        if image_url.startswith('data:image'):
                            logger.info(f"[agent] Image: [Base64 image data]")
                        else:
                            logger.info(f"[agent] Image URL: {image_url[:30]}...")
            else:
                content = str(msg['content'])
                chunk_size = 500
                for j in range(0, len(content), chunk_size):
                    chunk = content[j:j + chunk_size]
                    if j == 0:
                        logger.info(f"[agent] Content: {chunk}")
                    else:
                        logger.info(f"[agent] Content (continued): {chunk}")

            if 'tool_calls' in msg and msg['tool_calls']:
                for tool_call in msg.get('tool_calls'):
                    if isinstance(tool_call, dict):
                        logger.info(f"[agent] Tool call: {tool_call.get('name')} - ID: {tool_call.get('id')}")
                        args = str(tool_call.get('args', {}))[:1000]
                        logger.info(f"[agent] Tool args: {args}...")
                    elif isinstance(tool_call, ToolCall):
                        logger.info(f"[agent] Tool call: {tool_call.function.name} - ID: {tool_call.id}")
                        args = str(tool_call.function.arguments)[:1000]
                        logger.info(f"[agent] Tool args: {args}...")

    def _agent_result(self, actions: List[ActionModel], caller: str):
        if not actions:
            raise Exception(f'{self.name()} no action decision has been made.')

        tools = OrderedDict()
        agents = []
        for action in actions:
            if is_agent(action):
                agents.append(action)
            else:
                if action.tool_name not in tools:
                    tools[action.tool_name] = []
                tools[action.tool_name].append(action)

        _group_name = None
        # agents and tools exist simultaneously, more than one agent/tool name
        if (agents and tools) or len(agents) > 1 or len(tools) > 1:
            _group_name = f"{self.name()}_{uuid.uuid1().hex}"

        # complex processing
        if _group_name:
            logger.warning(f"more than one agent an tool causing confusion, will choose the first one. {agents}")
            agents = [agents[0]] if agents else []
            for _, v in tools.items():
                actions = v
                break

        if agents:
            return Message(payload=actions,
                           caller=caller,
                           sender=self.name(),
                           receiver=actions[0].tool_name,
                           session_id=self.context.session_id,
                           category=Constants.AGENT)
        else:
            return ToolMessage(payload=actions,
                               caller=caller,
                               sender=self.name(),
                               receiver=actions[0].tool_name,
                               session_id=self.context.session_id)

    def post_run(self, policy_result: List[ActionModel], policy_input: Observation) -> Message:
        return self._agent_result(
            policy_result,
            policy_input.from_agent_name if policy_input.from_agent_name else policy_input.observer
        )

    async def async_post_run(self, policy_result: List[ActionModel], policy_input: Observation) -> Message:
        return self._agent_result(
            policy_result,
            policy_input.from_agent_name if policy_input.from_agent_name else policy_input.observer
        )

    def policy(self, observation: Observation, info: Dict[str, Any] = {}, **kwargs) -> List[ActionModel]:
        """The strategy of an agent can be to decide which tools to use in the environment, or to delegate tasks to other agents.

        Args:
            observation: The state observed from tools in the environment.
            info: Extended information is used to assist the agent to decide a policy.

        Returns:
            ActionModel sequence from agent policy
        """
        output = None
        if kwargs.get("output") and isinstance(kwargs.get("output"), StepOutput):
            output = kwargs["output"]

        # Get current step information for trace recording
        step = kwargs.get("step", 0)
        exp_id = kwargs.get("exp_id", None)
        source_span = trace.get_current_span()

        if hasattr(observation, 'context') and observation.context:
            self.task_histories = observation.context

        self._finished = False
        self.desc_transform()
        images = observation.images if self.conf.use_vision else None
        if self.conf.use_vision and not images and observation.image:
            images = [observation.image]
            observation.images = images
        messages = self.messages_transform(content=observation.content,
                                           image_urls=observation.images,
                                           sys_prompt=self.system_prompt,
                                           agent_prompt=self.agent_prompt)

        self._log_messages(messages)
        self.memory.add(MemoryItem(
            content=messages[-1]['content'],
            metadata={
                "role": messages[-1]['role'],
                "agent_name": self.name(),
                "tool_call_id": messages[-1].get("tool_call_id")
            }
        ))

        llm_response = None
        span_name = f"llm_call_{exp_id}"
        serializable_messages = self._to_serializable(messages)
        with trace.span(span_name) as llm_span:
            llm_span.set_attributes({
                "exp_id": exp_id,
                "step": step,
                "messages": json.dumps(serializable_messages, ensure_ascii=False)
            })
            if source_span:
                source_span.set_attribute("messages", json.dumps(serializable_messages, ensure_ascii=False))

            try:
                llm_response = call_llm_model(
                    self.llm,
                    messages=messages,
                    model=self.model_name,
                    temperature=self.conf.llm_config.llm_temperature,
                    tools=self.tools if not self.use_tools_in_prompt and self.tools else None
                )

                logger.info(f"Execute response: {llm_response.message}")
            except Exception as e:
                logger.warn(traceback.format_exc())
                raise e
            finally:
                if llm_response:
                    use_tools = self.use_tool_list(llm_response)
                    is_use_tool_prompt = len(use_tools) > 0
                    if llm_response.error:
                        logger.info(f"llm result error: {llm_response.error}")
                    else:
                        info = {
                            "role": "assistant",
                            "agent_name": self.name(),
                            "tool_calls": llm_response.tool_calls if not self.use_tools_in_prompt else use_tools,
                            "is_use_tool_prompt": is_use_tool_prompt if not self.use_tools_in_prompt else False
                        }
                        self.memory.add(MemoryItem(
                            content=llm_response.content,
                            metadata=info
                        ))
                        # rewrite
                        self.context.context_info[self.name()] = info
                else:
                    logger.error(f"{self.name()} failed to get LLM response")
                    raise RuntimeError(f"{self.name()} failed to get LLM response")

        agent_result = sync_exec(self.resp_parse_func, llm_response)
        if not agent_result.is_call_tool:
            self._finished = True

        if output:
            output.add_part(MessageOutput(source=llm_response, json_parse=False))
            output.mark_finished()
        return agent_result.actions

    async def async_policy(self, observation: Observation, info: Dict[str, Any] = {}, **kwargs) -> List[ActionModel]:
        """The strategy of an agent can be to decide which tools to use in the environment, or to delegate tasks to other agents.

        Args:
            observation: The state observed from tools in the environment.
            info: Extended information is used to assist the agent to decide a policy.

        Returns:
            ActionModel sequence from agent policy
        """
        outputs = None
        if kwargs.get("outputs") and isinstance(kwargs.get("outputs"), Outputs):
            outputs = kwargs.get("outputs")

        # Get current step information for trace recording
        source_span = trace.get_current_span()

        if hasattr(observation, 'context') and observation.context:
            self.task_histories = observation.context

        self._finished = False
        messages = await self._prepare_llm_input(observation, info, **kwargs)

        serializable_messages = self._to_serializable(messages)
        llm_response = None
        if source_span:
            source_span.set_attribute("messages", json.dumps(serializable_messages, ensure_ascii=False))
        try:
            # update messages to agent context
            self.update_current_agent_messages(messages)
            async for event in self.run_hooks(self.context, self.current_agent_context, HookPoint.PRE_LLM_CALL):
                await event
            # restore messages
            messages = self.restore_current_agent_context()

            llm_response = await self._call_llm_model(observation, messages, info, **kwargs)

        except Exception as e:
            logger.warn(traceback.format_exc())
            raise e
        finally:
            if llm_response:
                # update usage
                self.context.update_current_agent_context_usage(self.id, llm_response.usage['total_tokens'])

                use_tools = self.use_tool_list(llm_response)
                is_use_tool_prompt = len(use_tools) > 0
                if llm_response.error:
                    logger.info(f"llm result error: {llm_response.error}")
                else:
                    self.memory.add(MemoryItem(
                        content=llm_response.content,
                        metadata={
                            "role": "assistant",
                            "agent_name": self.name(),
                            "tool_calls": llm_response.tool_calls if not self.use_tools_in_prompt else use_tools,
                            "is_use_tool_prompt": is_use_tool_prompt if not self.use_tools_in_prompt else False
                        }
                    ))
            else:
                logger.error(f"{self.name()} failed to get LLM response")
                raise RuntimeError(f"{self.name()} failed to get LLM response")

        agent_result = sync_exec(self.resp_parse_func, llm_response)
        if not agent_result.is_call_tool:
            self._finished = True
        return agent_result.actions

    def _to_serializable(self, obj):
        if isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._to_serializable(i) for i in obj]
        elif hasattr(obj, "to_dict"):
            return obj.to_dict()
        elif hasattr(obj, "model_dump"):
            return obj.model_dump()
        elif hasattr(obj, "dict"):
            return obj.dict()
        else:
            return obj
    def update_current_agent_context(self, context_rule: ContextRuleConfig):
        current_agent_context = AgentContext(
            agent_id=self.id,
            agent_name=self.name(),
            agent_desc=self._desc,
            system_prompt=self.system_prompt,
            agent_prompt=self.agent_prompt,
            tool_names=self.tool_names,
            context_rule=context_rule,
            context_usage=ContextUsage(total_context_length=self.conf.llm_config.max_model_len)
        )
        self.context.set_current_agent_context(self.id, current_agent_context)
        self.current_agent_context = current_agent_context

    def update_current_agent_messages(self, messages: List[Message]):
        self.current_agent_context.set_messages(messages)

    def restore_current_agent_context(self) -> List[Message]:
        return self.current_agent_context.messages

    async def run_hooks(self, context: Context, current_agent_context: AgentContext, hook_point: str) -> AsyncGenerator[Message, None]:
        hooks = HookFactory.hooks(hook_point).get(hook_point)
        print('hook_point|', hook_point, '|', HookFactory.hooks(hook_point), '|', context, '|', current_agent_context)
        for hook in hooks:
            try:
                print('current hook_point|', hook_point, '|', hook, '|', context, '|', current_agent_context)
                msg = hook.exec(message=None, current_agent_context=current_agent_context, context=context)
                if msg:
                    yield msg
            except:
                logger.warning(f"{hook.point()} {hook.name()} execute fail.")

    async def llm_and_tool_execution(self, observation: Observation, messages: List[Dict[str, str]] = [], info: Dict[str, Any] = {}, **kwargs) -> List[ActionModel]:
        """Perform combined LLM call and tool execution operations.

        Args:
            observation: The state observed from the environment
            info: Extended information to assist the agent in decision-making
            **kwargs: Other parameters

        Returns:
            ActionModel sequence. If a tool is executed, includes the tool execution result.
        """
        # Get current step information for trace recording
        llm_response = await self._call_llm_model(observation, messages, info, **kwargs)
        if llm_response:
            use_tools = self.use_tool_list(llm_response)
            is_use_tool_prompt = len(use_tools) > 0
            if llm_response.error:
                logger.info(f"llm result error: {llm_response.error}")
            else:
                self.memory.add(MemoryItem(
                    content=llm_response.content,
                    metadata={
                        "role": "assistant",
                        "agent_name": self.name(),
                        "tool_calls": llm_response.tool_calls if not self.use_tools_in_prompt else use_tools,
                        "is_use_tool_prompt": is_use_tool_prompt if not self.use_tools_in_prompt else False
                    }
                ))
        else:
            logger.error(f"{self.name()} failed to get LLM response")
            raise RuntimeError(f"{self.name()} failed to get LLM response")

        agent_result = sync_exec(self.resp_parse_func, llm_response)
        if not agent_result.is_call_tool:
            self._finished = True
            return agent_result.actions
        else:
            result = await self._execute_tool(agent_result.actions)
            return result

    async def _prepare_llm_input(self, observation: Observation, info: Dict[str, Any] = {}, **kwargs):
        """Prepare LLM input
        Args:
            observation: The state observed from the environment
            info: Extended information to assist the agent in decision-making
            **kwargs: Other parameters
        """
        await self.async_desc_transform()
        images = observation.images if self.conf.use_vision else None
        if self.conf.use_vision and not images and observation.image:
            images = [observation.image]
        messages = self.messages_transform(content=observation.content,
                                           image_urls=images,
                                           sys_prompt=self.system_prompt,
                                           agent_prompt=self.agent_prompt)

        self._log_messages(messages)
        self.memory.add(MemoryItem(
            content=messages[-1]['content'],
            metadata={
                "role": messages[-1]['role'],
                "agent_name": self.name(),
                "tool_call_id": messages[-1].get("tool_call_id")
            }
        ))
        return messages


    async def _call_llm_model(self, observation: Observation, messages: List[Dict[str, str]] = [], info: Dict[str, Any] = {}, **kwargs) -> ModelResponse:
        """Perform LLM call
        Args:
            observation: The state observed from the environment
            info: Extended information to assist the agent in decision-making
            **kwargs: Other parameters
        Returns:
            LLM response
        """
        outputs = None
        if kwargs.get("outputs") and isinstance(kwargs.get("outputs"), Outputs):
            outputs = kwargs.get("outputs")
        if not messages:
            messages = await self._prepare_llm_input(observation, info, **kwargs)

        llm_response = None
        source_span = trace.get_current_span()
        serializable_messages = self._to_serializable(messages)

        if source_span:
            source_span.set_attribute("messages", json.dumps(serializable_messages, ensure_ascii=False))

        try:
            stream_mode = kwargs.get("stream", False)
            if stream_mode:
                llm_response = ModelResponse(id="", model="", content="", tool_calls=[])
                resp_stream = acall_llm_model_stream(
                    self.llm,
                    messages=messages,
                    model=self.model_name,
                    temperature=self.conf.llm_config.llm_temperature,
                    tools=self.tools if not self.use_tools_in_prompt and self.tools else None,
                    stream=True
                )

                async def async_call_llm(resp_stream, json_parse=False):
                    llm_resp = ModelResponse(id="", model="", content="", tool_calls=[])

                    # Async streaming with acall_llm_model
                    async def async_generator():
                        async for chunk in resp_stream:
                            if chunk.content:
                                llm_resp.content += chunk.content
                                yield chunk.content
                            if chunk.tool_calls:
                                llm_resp.tool_calls.extend(chunk.tool_calls)
                            if chunk.error:
                                llm_resp.error = chunk.error
                            llm_resp.id = chunk.id
                            llm_resp.model = chunk.model
                            llm_resp.usage = nest_dict_counter(llm_resp.usage, chunk.usage)

                    return MessageOutput(source=async_generator(), json_parse=json_parse), llm_resp


                output, response = await async_call_llm(resp_stream)
                llm_response = response

                if InMemoryEventbus.instance() and resp_stream:
                    output_message = Message(
                        category=Constants.OUTPUT,
                        payload=output,
                        sender=self.name(),
                        session_id=Context.instance().session_id
                    )
                    await InMemoryEventbus.instance().publish(output_message)
                elif not self.event_driven and outputs:
                    outputs.append(output)

            else:
                llm_response = await acall_llm_model(
                    self.llm,
                    messages=messages,
                    model=self.model_name,
                    temperature=self.conf.llm_config.llm_temperature,
                    tools=self.tools if not self.use_tools_in_prompt and self.tools else None,
                    stream=kwargs.get("stream", False)
                )
                if InMemoryEventbus.instance() and llm_response:
                    await InMemoryEventbus.instance().publish(Message(
                        category=Constants.OUTPUT,
                        payload=llm_response,
                        sender=self.name(),
                        session_id=Context.instance().session_id
                    ))
                elif not self.event_driven and outputs:
                    outputs.append(llm_response)

            logger.info(f"Execute response: {json.dumps(llm_response.to_dict(), ensure_ascii=False)}")


        except Exception as e:
            logger.warn(traceback.format_exc())
            raise e
        finally:
            return llm_response

    async def _execute_tool(self, actions: List[ActionModel]) -> Any:
        """Execute tool calls

        Args:
            action: The action(s) to execute

        Returns:
            The result of tool execution
        """
        tool_actions = []
        for act in actions:
            if is_agent(act):
                continue
            else:
                tool_actions.append(act)

        msg = None
        terminated = False
        # group action by tool name
        tool_mapping = dict()
        reward = 0.0
        # Directly use or use tools after creation.
        for act in tool_actions:
            if not self.tools_instances or (self.tools_instances and act.tool_name not in self.tools):
                # Dynamically only use default config in module.
                conf = self.tools_conf.get(act.tool_name)
                if not conf:
                    conf = ToolConfig(exit_on_failure=self.task.conf.get('exit_on_failure'))
                tool = ToolFactory(act.tool_name, conf=conf, asyn=conf.use_async if conf else False)
                if isinstance(tool, Tool):
                    tool.reset()
                elif isinstance(tool, AsyncTool):
                    await tool.reset()
                tool_mapping[act.tool_name] = []
                self.tools_instances[act.tool_name] = tool
            if act.tool_name not in tool_mapping:
                tool_mapping[act.tool_name] = []
            tool_mapping[act.tool_name].append(act)

        observation = None

        for tool_name, action in tool_mapping.items():
            # Execute action using browser tool and unpack all return values
            if isinstance(self.tools_instances[tool_name], Tool):
                message = self.tools_instances[tool_name].step(action)
            elif isinstance(self.tools_instances[tool_name], AsyncTool):
                # todo sandbox
                message = await self.tools_instances[tool_name].step(action, agent=self)
            else:
                logger.warning(f"Unsupported tool type: {self.tools_instances[tool_name]}")
                continue

            observation, reward, terminated, _, info = message.payload


            # Check if there's an exception in info
            if info.get("exception"):
                color_log(f"Agent {self.name()} _execute_tool failed with exception: {info['exception']}", color=Color.red)
                msg = f"Agent {self.name()} _execute_tool failed with exception: {info['exception']}"
            logger.info(f"Agent {self.name()} _execute_tool finished by tool action: {action}.")
            log_ob = Observation(content='' if observation.content is None else observation.content,
                                 action_result=observation.action_result)
            trace_logger.info(f"{tool_name} observation: {log_ob}", color=Color.green)
            self.memory.add(MemoryItem(
                content=observation.content,
                metadata={
                    "role": "tool",
                    "agent_name": self.name(),
                    "tool_call_id": action[0].tool_id
                }
            ))
        return [ActionModel(agent_name=self.name(), policy_info=observation.content)]


@HookFactory.register(name="PreLLMCallContextProcessHook",
                      desc="PreLLMCallContextProcessHook")
class PreLLMCallContextProcessHook(PreLLMCallHook):
    """Process in the hook point of the pre_llm_call."""
    __metaclass__ = abc.ABCMeta

    def name(self):
        return convert_to_snake("PreLLMCallContextProcessHook")

    def process_messages(self, messages: List[Dict[str, Any]],
                         context: Context,
                         current_agent_context: AgentContext,
                         ) -> List[Dict[str, Any]]:
        context_rule = current_agent_context.context_rule
        if context_rule is None:
            logger.debug('debug|skip PreLLMCallContextProcessHook context_rule is None')
            return messages

        context_processor = ContextProcessor(context_rule, current_agent_context)
        result = context_processor.process_context(messages, context)

        return result.processed_messages

    async def exec(self, message: Message, current_agent_context: AgentContext = None, context: Context = None) -> Message:
        messages = origin_messages = current_agent_context.messages
        st = time.time()
        with trace.span(f"llm_context_pre_hook", attributes={
            "start_time": st
        }) as compress_span:
            origin_len = compressed_len = len(str(messages))
            origin_messages_count = truncated_messages_count = len(messages)
            try:
                messages = self.process_messages(messages, context, current_agent_context)
                compressed_len = len(str(messages))
                truncated_messages_count = len(messages)
                logger.debug(f'debug|llm_context_compress|{origin_len}|{compressed_len}|{origin_messages_count}|{truncated_messages_count}|\n|{origin_messages}\n|{messages}')
            finally:
                compress_span.set_attributes({
                    "end_time": time.time(),
                    "duration": time.time() - st,
                    # messages length
                    "origin_messages_count": origin_messages_count,
                    "truncated_messages_count": truncated_messages_count,
                    "truncated_ratio": round(truncated_messages_count / origin_messages_count, 2),
                    # token length
                    "origin_len": origin_len,
                    "compressed_len": compressed_len,
                    "compress_ratio": round(compressed_len / origin_len, 2)
                })