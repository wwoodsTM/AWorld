# coding: utf-8
# Copyright (c) 2025 inclusionAI.
import copy
import json
import time
import traceback
from typing import Dict, Any, List

from aworld.core.agent.base import BaseAgent, AgentFactory
from aworld.models.utils import tool_desc_transform
from aworld.config.conf import AgentConfig
from aworld.core.common import Observation, ActionModel, Agents
from aworld.logs.util import logger
from aworld.models.llm import get_llm_model
from aworld.core.envs.tools_desc import tool_action_desc_dict

init_prompt = f"""
Now please give me instructions to solve over overall task step by step. If the task requires some specific knowledge, please instruct me to use tools to complete the task.
"""

system_prompt = """
===== RULES OF ASSISTANT =====
Never forget you are a assistant and I am a user. Never flip roles! Never instruct me! You have to utilize your available tools to solve the task I assigned.
We share a common interest in collaborating to successfully complete a complex task.
You must help me to complete the task.

Here is our overall task: {task}. Never forget our task!

I must instruct you based on your expertise and my needs to complete the task. An instruction is typically a sub-task or question.

You must leverage your available tools, try your best to solve the problem, and explain your solutions.
Unless I say the task is completed, you should always start with:
Solution: [YOUR_SOLUTION]
[YOUR_SOLUTION] should be specific, including detailed explanations and provide preferable detailed implementations and examples and lists for task-solving.

Please note that our overall task may be very complicated. Here are some tips that may help you solve the task:
<tips>
- If one way fails to provide an answer, try other ways or methods. The answer does exists.
- If the search snippet is unhelpful but the URL comes from an authoritative source, try visit the website for more details.  
- When looking for specific numerical values (e.g., dollar amounts), prioritize reliable sources and avoid relying only on search snippets.  
- When solving tasks that require web searches, check Wikipedia first before exploring other websites.  
- When trying to solve math problems, you can try to write python code and use sympy library to solve the problem.
- Always verify the accuracy of your final answers! Try cross-checking the answers by other ways. (e.g., screenshots, webpage analysis, etc.).  
- Do not be overly confident in your own knowledge. Searching can provide a broader perspective and help validate existing knowledge.  
- After writing codes, do not forget to run the code and get the result. If it encounters an error, try to debug it.
- When a tool fails to run, or the code does not run correctly, never assume that it returns the correct result and continue to reason based on the assumption, because the assumed result cannot lead you to the correct answer. The right way is to think about the reason for the error and try again.
- Search results typically do not provide precise answers. It is not likely to find the answer directly using search toolkit only, the search query should be concise and focuses on finding sources rather than direct answers, as it always need to use other tools to further process the url, e.g. interact with the webpage, extract webpage content, etc. 
- For downloading files, you can either use the web browser simulation toolkit or write codes.
</tips>
"""


@AgentFactory.register(name=Agents.EXECUTE.value, desc="execute agent")
class ExcuteAgent(BaseAgent):
    def __init__(self, conf: AgentConfig, **kwargs):
        super(ExcuteAgent, self).__init__(conf, **kwargs)
        self.model_name = conf.llm_model_name
        self.llm = get_llm_model(conf)
        self.tools = tool_desc_transform(tool_action_desc_dict)

    def name(self) -> str:
        return Agents.EXECUTE.value

    def reset(self, options: Dict[str, Any]):
        """Execute agent reset need query task as input."""
        self.task = options.get("task")
        self.trajectory = []
        self.system_prompt = system_prompt.format(task=self.task)

    def policy(self,
               observation: Observation,
               info: Dict[str, Any] = None,
               **kwargs) -> List[ActionModel] | None:
        start_time = time.time()
        content = observation.content

        llm_result = None
        ## 构建json
        input_content = [
            {'role': 'system', 'content': self.system_prompt},
        ]
        for traj in self.trajectory:
            input_content.append(traj[0].content)
            if traj[-1].choices[0].message.tool_calls is not None:
                input_content.append(
                    {'role': 'assistant', 'content': '', 'tool_calls': traj[-1].choices[0].message.tool_calls})
            else:
                input_content.append({'role': 'assistant', 'content': traj[-1].choices[0].message.content})

        message = {'role': 'user', 'content': content}
        input_content.append(message)

        tool_calls = []
        try:
            llm_result = self.llm.chat.completions.create(
                messages=input_content,
                model=self.model_name,
                **{'temperature': 0, 'tools': self.tools},
            )
            logger.info(f"llm response: {llm_result.choices[0].message}")
            content = llm_result.choices[0].message.content
            tool_calls = llm_result.choices[0].message.tool_calls
        except Exception as e:
            logger.warn(traceback.format_exc())
            raise e
        finally:
            if llm_result:
                if llm_result.choices is None:
                    logger.info(f"llm result is None, info: {llm_result.model_extra}")
                ob = copy.deepcopy(observation)
                ob.content = message
                self.trajectory.append((ob, info, llm_result))
            else:
                logger.warn("no result to record!")

        res = []
        if tool_calls:
            for tool_call in tool_calls:
                tool_action_name: str = tool_call.function.name
                if not tool_action_name:
                    continue
                tool_name = tool_action_name.split("__")[0]
                action_name = tool_action_name.split("__")[1]
                params = json.loads(tool_call.function.arguments)
                res.append(ActionModel(tool_name=tool_name, action_name=action_name, params=params))
        else:
            self._finished = True
        if res:
            res[0].policy_info = content
        elif content:
            res.append(ActionModel(agent_name=self.name(), policy_info=content))

        print(f">>> execute result: {res}")
        return res


@AgentFactory.register(name=Agents.PLAN.value, desc="plan agent")
class PlanAgent(BaseAgent):
    def __init__(self, conf: AgentConfig, **kwargs):
        super(PlanAgent, self).__init__(conf, **kwargs)
        self.model_name = conf.llm_model_name
        self.llm = get_llm_model(conf)

    def name(self) -> str:
        return Agents.PLAN.value

    def reset(self, options: Dict[str, Any]):
        """Execute agent reset need query task as input."""
        self.task = options.get("task")
        self._build_prompt(input)
        self.trajectory = []
        self.done_prompt = f"""\n
        Here are auxiliary information about the overall task, which may help you understand the intent of the current task:
        <auxiliary_information>
        {self.task}
        </auxiliary_information>
        If there are available tools and you want to call them, never say 'I will ...', but first call the tool and reply based on tool call's result, and tell me which tool you have called.
        """
        self.postfix_prompt = f"""\n
        Now please make a final answer of the original task based on our conversation : <task>{self.task}</task>
        Please pay special attention to the format in which the answer is presented.
        You should first analyze the answer format required by the question and then output the final answer that meets the format requirements. 
        Your response should include the following content:
        - `analysis`: enclosed by <analysis> </analysis>, a detailed analysis of the reasoning result.
        - `final_answer`: enclosed by <final_answer> </final_answer>, the final answer to the question.
        Here are some hint about the final answer:
        <hint>
        Your final answer must be output exactly in the format specified by the question. It should be a number OR as few words as possible OR a comma separated list of numbers and/or strings:
        - If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. 
        - If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. 
        - If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
        </hint>
        """

    def _build_prompt(self, task_prompt):
        self.system_prompt = f"""
===== RULES OF USER =====
Never forget you are a user and I am a assistant. Never flip roles! You will always instruct me. We share a common interest in collaborating to successfully complete a task.
I must help you to complete a difficult task.
You must instruct me based on my expertise and your needs to solve the task step by step. The format of your instruction is: `Instruction: [YOUR INSTRUCTION]`, where "Instruction" describes a sub-task or question.
You must give me one instruction at a time.
I must write a response that appropriately solves the requested instruction.
You should instruct me not ask me questions.

Please note that the task may be very complicated. Do not attempt to solve the task by single step. You must instruct me to find the answer step by step.
Here are some tips that will help you to give more valuable instructions about our task to me:
<tips>
- I have various tools to use, such as search toolkit, web browser simulation toolkit, document relevant toolkit, code execution toolkit, etc. Thus, You must think how human will solve the task step-by-step, and give me instructions just like that. For example, one may first use google search to get some initial information and the target url, then retrieve the content of the url, or do some web browser interaction to find the answer.
- Although the task is complex, the answer does exist. If you can’t find the answer using the current scheme, try to re-plan and use other ways to find the answer, e.g. using other tools or methods that can achieve similar results.
- Always remind me to verify my final answer about the overall task. This work can be done by using multiple tools(e.g., screenshots, webpage analysis, etc.), or something else.
- If I have written code, please remind me to run the code and get the result.
- Search results typically do not provide precise answers. It is not likely to find the answer directly using search toolkit only, the search query should be concise and focuses on finding sources rather than direct answers, as it always need to use other tools to further process the url, e.g. interact with the webpage, extract webpage content, etc. 
- If the question mentions youtube video, in most cases you have to process the content of the mentioned video.
- For downloading files, you can either use the web browser simulation toolkit or write codes (for example, the github content can be downloaded via https://raw.githubusercontent.com/...).
- Flexibly write codes to solve some problems, such as excel relevant tasks.
</tips>

Now, here is the overall task: <task>{task_prompt}</task>. Never forget our task!

Now you must start to instruct me to solve the task step-by-step. Do not add anything else other than your instruction!
Keep giving me instructions until you think the task is completed.
When the task is completed, you must only reply with a single word <TASK_DONE>.
Never say <TASK_DONE> unless my responses have solved your task.
        """

    def policy(self,
               observation: Observation,
               info: Dict[str, Any] = None,
               **kwargs) -> List[ActionModel] | None:
        llm_result = None
        # 根据历史，构建input
        input_content = [
            {'role': 'system', 'content': self.system_prompt},
        ]
        for traj in self.trajectory:
            input_content.append(traj[0].content)
            if traj[-1].choices[0].message.tool_calls is not None:
                input_content.append(
                    {'role': 'assistant', 'content': '', 'tool_calls': traj[-1].choices[0].message.tool_calls})
            else:
                input_content.append({'role': 'assistant', 'content': traj[-1].choices[0].message.content})

        message = observation.content
        input_content.append({'role': 'user', 'content': message})
        try:
            llm_result = self.llm.chat.completions.create(
                messages=input_content,
                model=self.model_name,
            )
        except Exception as e:
            logger.warn(traceback.format_exc())
            raise e
        finally:
            if llm_result:
                if llm_result.choices is None:
                    logger.info(f"llm result is None, info: {llm_result.model_extra}")
                ob = copy.deepcopy(observation)
                ob.content = message
                self.trajectory.append((ob, info, llm_result))
            else:
                logger.warn("no result to record!")
        content = llm_result.choices[0].message.content
        if "TASK_DONE" not in content:
            # The task is done, and the assistant agent need to give the final answer about the original task
            content += self.done_prompt
            self._finished = True
        else:
            content += self.postfix_prompt

        print(">>> plan result: " + content)
        return [ActionModel(agent_name="execute",
                            policy_info=content)]
