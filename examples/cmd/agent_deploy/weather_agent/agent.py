import logging
import os
import json

from aworld.config.conf import AgentConfig, TaskConfig
from aworld.agents.llm_agent import Agent
from aworld.core.task import Task
from aworld.output.ui.base import AworldUI
from aworld.output.ui.markdown_aworld_ui import MarkdownAworldUI
from aworld.runner import Runners

logger = logging.getLogger(__name__)


class AWorldAgent:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_agent_info(self):
        return {"name": "Weather Agent", "description": "Query Real-time Weather"}

    async def run(self, prompt: str):
        llm_provider = os.getenv("LLM_PROVIDER_WEATHER", "openai")
        llm_model_name = os.getenv("LLM_MODEL_NAME_WEATHER")
        llm_api_key = os.getenv("LLM_API_KEY_WEATHER")
        llm_base_url = os.getenv("LLM_BASE_URL_WEATHER")
        llm_temperature = os.getenv("LLM_TEMPERATURE_WEATHER", 0.0)

        if not llm_model_name or not llm_api_key or not llm_base_url:
            raise ValueError(
                "LLM_MODEL_NAME, LLM_API_KEY, LLM_BASE_URL must be set in your envrionment variables"
            )

        agent_config = AgentConfig(
            llm_provider=llm_provider,
            llm_model_name=llm_model_name,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            llm_temperature=llm_temperature,
        )

        path_cwd = os.path.dirname(os.path.abspath(__file__))
        mcp_path = os.path.join(path_cwd, "mcp.json")
        with open(mcp_path, "r") as f:
            mcp_config = json.load(f)

        super_agent = Agent(
            conf=agent_config,
            name="weather_agent",
            system_prompt="You are a weather agent, you can query real-time weather information",
            mcp_config=mcp_config,
            mcp_servers=mcp_config.get("mcpServers", {}).keys(),
        )

        task = Task(input=prompt, agent=super_agent, event_driven=False, conf=TaskConfig(max_steps=20))

        rich_ui = MarkdownAworldUI()
        async for output in Runners.streamed_run_task(task).stream_events():
            logger.info(f"Agent Ouput: {output}")
            res = await AworldUI.parse_output(output, rich_ui)
            for item in res if isinstance(res, list) else [res]:
                yield item
