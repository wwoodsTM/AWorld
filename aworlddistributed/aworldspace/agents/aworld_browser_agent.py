import logging
import os

from aworld.config import ModelConfig
from aworld.config.conf import AgentConfig
from pydantic import BaseModel

from aworldspace.base_agent import AworldBaseAgent

SYSTEM_PROMPT = f"""You are an helpful AI assistant, aimed at solving any task presented by the user. 1"""


class Pipeline(AworldBaseAgent):
    class Valves(BaseModel):
        pass

    def __init__(self):
        self.valves = self.Valves()
        logging.info("default init success")

    async def get_agent_config(self, body):
        default_llm_provider = os.environ.get("LLM_PROVIDER")
        llm_model_name = os.environ.get("LLM_MODEL_NAME")
        llm_api_key = os.environ.get("LLM_API_KEY")
        llm_base_url = os.environ.get("LLM_BASE_URL")

        task = await self.get_task_from_body(body)
        if task:
            logging.info(f"task llm config is: {task.llm_provider}, {task.llm_model_name},{task.llm_base_url}")

        llm_config = ModelConfig(
            llm_provider=task.llm_provider if task and task.llm_provider else default_llm_provider,
            llm_model_name=task.llm_model_name if task and task.llm_model_name else llm_model_name,
            llm_api_key=task.llm_api_key if task and task.llm_api_key else llm_api_key,
            llm_base_url=task.llm_base_url if task and task.llm_base_url else llm_base_url,
            max_retries=task.max_retries if task and task.max_retries else 3
        )

        return AgentConfig(
            name=self.agent_name(),
            llm_config=llm_config,
            system_prompt=task.task_system_prompt if task and task.task_system_prompt else SYSTEM_PROMPT
        )

    def agent_name(self) -> str:
        return "DefaultBrowserAgent"

    async def get_mcp_servers(self, body) -> list[str]:
        task = await self.get_task_from_body(body)
        if task and task.mcp_servers:
            logging.info(f"mcp_servers from task: {task.mcp_servers}")
            return task.mcp_servers

        return [
            "aworld-playwright"
        ]

    async def load_mcp_config(self) -> dict:
        return {
            "mcpServers": {
                "aworld-playwright": {
                    "command": "npx",
                    "args": [
                        "playwright-mcp-aworld",
                        "--isolated"
                    ],
                    "env": {
                        "OSS_ENDPOINT": "${OSS_ENDPOINT}",
                        "OSS_ACCESS_KEY_ID": "${OSS_ACCESS_KEY_ID}",
                        "OSS_ACCESS_KEY_SECRET": "${OSS_ACCESS_KEY_SECRET}",
                        "OSS_BUCKET": "${OSS_BUCKET_NAME}",
                        "PLAYWRIGHT_TIMEOUT": "120000",
                        "SESSION_REQUEST_CONNECT_TIMEOUT": "120"
                    }
                }
            }
        }
