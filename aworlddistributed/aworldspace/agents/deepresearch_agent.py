# coding: utf-8
# Copyright (c) 2025 inclusionAI.
import os
import sys
LLM_BASE_URL = "https://agi-pre.alipay.com/api"
LLM_API_KEY = "sk-9329256ff1394003b6761615361a8f0f"
LLM_MODEL_NAME = "shangshu.claude-3.7-sonnet"
os.environ["LLM_API_KEY"] = LLM_API_KEY
os.environ["LLM_BASE_URL"] = LLM_BASE_URL
os.environ["LLM_MODEL_NAME"] = LLM_MODEL_NAME
os.environ['GOOGLE_API_KEY'] = "AIzaSyDl7Axs2CyS0nvBJ47QL30t84N2-azuFNQ"
os.environ['TAVILY_API_KEY'] = "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"

from aworld.config.conf import TaskConfig

from aworld.core.agent.swarm import Swarm
from aworld.core.task import Task
from aworld.runner import Runners
from plan_agent import plan_agent
from reasoning_loop_agent import reasoning_loop_agent
from reporting_agent import reporting_agent

def main():
    goal = """ 帮我做一个国庆节去北京旅游的7天计划 """

    swarm = Swarm(plan_agent,
                  #reporting_agent,
                  sequence=True)

    task = Task(
        swarm=swarm,
        input=goal,
        endless_threshold=5,
        conf=TaskConfig(exit_on_failure=True),
        event_driven=False
    )
    result = Runners.sync_run_task(task)
    print("finalResult:", result)

if __name__ == '__main__':
    main()
