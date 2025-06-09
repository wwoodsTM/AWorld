import os
import uuid
from aworld.config.conf import AgentConfig, ToolConfig
from aworld.core.agent.llm_agent import Agent
from aworld.core.task import Task
from aworld.config import ModelConfig, TaskConfig
from aworldspace.prompt.deepresearch_prompt import *
from aworld.runner import Runners

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

# reasoning_loop_agent
reasoning_loop_agent = Agent(
    name="reasoning_loop_agent",
    desc="reasoning_loop_agent",
    conf=agent_config,
    system_prompt=reasoning_loop_sys_prompt,
    mcp_servers=[
        #"ms-playwright", "google-search",
        "tavily"
    ],
    mcp_config={
        "mcpServers": {
            # "ms-playwright": {
            #     "command": "npx",
            #     "args": [
            #         "@playwright/mcp@0.0.27",
            #         "--no-sandbox",
            #         "--headless"
            #     ],
            #     "env": {
            #         "PLAYWRIGHT_TIMEOUT": "120000",
            #         "SESSION_REQUEST_CONNECT_TIMEOUT": "120"
            #     }
            # },
            # "google-search": {
            #     "command": "npx",
            #     "args": [
            #         "-y",
            #         "@adenot/mcp-google-search"
            #     ],
            #     "env": {
            #         "GOOGLE_API_KEY": os.environ["GOOGLE_API_KEY"],
            #         "GOOGLE_SEARCH_ENGINE_ID": os.environ["GOOGLE_CSE_ID"],
            #         "SESSION_REQUEST_CONNECT_TIMEOUT": "60"
            #     }
            # },
            "tavily": {
                "command": "npx",
                "args": ["-y", "tavily-mcp@0.2.2"],
                "env": {
                    "TAVILY_API_KEY": os.environ["TAVILY_API_KEY"],
                    "SESSION_REQUEST_CONNECT_TIMEOUT": "60"
                }
            }
        }
    }
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
        agent=reasoning_loop_agent,
        event_driven=False,
        conf=TaskConfig(
            task_id=task_id,
            stream=False,
            ext={
                "origin_message": user_input
            }
        )
    )
    Runners.sync_run_task(task)

if __name__ == '__main__':
    main()