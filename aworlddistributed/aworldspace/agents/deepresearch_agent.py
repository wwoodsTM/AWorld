# coding: utf-8
# Copyright (c) 2025 inclusionAI.
from aworld.core.agent.swarm import Swarm
from aworld.core.task import Task
from aworld.runner import Runners
from plan_agent import plan_agent
from reasoning_loop_agent import reasoning_loop_agent
from reporting_agent import reporting_agent
from dotenv import load_dotenv

def main():
    goal = """ 帮我做一个国庆节去北京旅游的7天计划 """
    swarm = Swarm((plan_agent, reasoning_loop_agent),
                  #(plan_agent, reporting_agent),
                  sequence=False, event_driven=False)
    task = Task(
        swarm=swarm,
        input=goal,
        endless_threshold=5
    )
    Runners.sync_run_task(task)

if __name__ == '__main__':
    load_dotenv()
    main()
