import os
import uuid
from aworld.config.conf import AgentConfig
from aworld.core.agent.llm_agent import Agent
from aworld.core.task import Task
from aworld.config import ModelConfig, TaskConfig
from aworldspace.prompt.deepresearch_prompt import *
from aworld.runner import Runners
from dotenv import load_dotenv

model_config = ModelConfig(
    llm_provider="openai",
    llm_model_name=os.getenv("LLM_MODEL_NAME"),
    llm_base_url=os.getenv("LLM_BASE_URL"),
    llm_api_key=os.getenv("LLM_API_KEY")
)

agent_config = AgentConfig(
    llm_config=model_config,
    use_vision=False
)

# plan_agent 没有工具
plan_agent = Agent(
    name="plan_agent",
    desc="plan",
    conf=agent_config,
    system_prompt=plan_sys_prompt,
    step_reset=False,
    event_driven=False
)

def main():
    user_input = """
        I need a 7-day Japan itinerary from April 2 to April 8 2025, departing from Hangzhou, We want to see beautiful cherry blossoms and experience traditional Japanese culture (kendo, tea ceremonies, Zen meditation). We would like to taste matcha in Uji and enjoy the hot springs in Kobe. I am planning to propose during this trip, so I need a special location recommendation. Please provide a detailed itinerary and create a simple HTML travel handbook that includes a 7-day Japan itinerary, an updated cherry blossom table, attraction descriptions, essential Japanese phrases, and travel tips for us to reference throughout our journey.
        you need search and extract different info 1 times, and then write, at last use browser agent goto the html url and then, complete the task.
        """
    task_id = str(uuid.uuid4())
    task = Task(
        id=task_id,
        name=task_id,
        input=user_input,
        agent=plan_agent,
        event_driven=False,
        conf=TaskConfig(
            task_id=task_id,
            stream=False,
            ext={
                "origin_message": user_input
            }
        )
    )
    result = Runners.sync_run_task(task)
    print("result:", result)

if __name__ == '__main__':
    load_dotenv()
    main()
