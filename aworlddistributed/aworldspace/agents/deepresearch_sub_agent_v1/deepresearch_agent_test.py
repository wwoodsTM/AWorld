import uuid
import os
from aworld.config import TaskConfig, AgentConfig, ModelConfig
from aworld.core.agent.swarm import Swarm
from aworld.core.task import Task
from aworld.runner import Runners
from plan_agent import PlanAgent
from web_search_agent import WebSearchAgent
from reasoning_loop_agent import ReasoningLoopAgent
from reporting_agent import ReportingAgent


def main():
    user_input = "7天北京旅游计划"

    agent_config = AgentConfig(
        llm_config=ModelConfig(
            llm_provider="openai",
            llm_model_name=os.getenv("LLM_MODEL_NAME"),
            llm_base_url=os.getenv("LLM_BASE_URL"),
            llm_api_key=os.getenv("LLM_API_KEY")
        ),
        use_vision=False
    )

    plan_agent = PlanAgent(
        name="plan_agent",
        desc="plan_agent",
        conf=agent_config,
    )

    web_search_agent = WebSearchAgent(
        name="web_search_agent",
        desc="web_search_agent",
        conf=agent_config,
        mcp_servers=["aworldsearch_server"],
        mcp_config={
            "mcpServers": {
                "aworldsearch_server": {
                    "command": "python",
                    "args": [
                        "-m",
                        "mcp_servers.aworldsearch_server"
                    ],
                    "env": {
                        "AWORLD_SEARCH_URL": "https://antragflowInside.alipay.com/v1/rpc/ragLlmSearch",
                        "AWORLD_SEARCH_TOTAL_NUM": "10",
                        "AWORLD_SEARCH_SLICE_NUM": "3",
                        "AWORLD_SEARCH_DOMAIN": "google",
                        "AWORLD_SEARCH_SEARCHMODE": "RAG_LLM",
                        "AWORLD_SEARCH_SOURCE": "lingxi_agent",
                        "AWORLD_SEARCH_UID": "2088802724428205"
                    }
                }
            }
        }
    )

    reasoning_loop_agent = ReasoningLoopAgent(
        name="reasoning_loop_agent",
        desc="reasoning_loop_agent",
        conf=agent_config
    )

    reporting_agent = ReportingAgent(
        name="reporting_agent",
        desc="reporting_agent",
        conf=agent_config
    )

    swarm = Swarm(plan_agent, web_search_agent, reasoning_loop_agent, reporting_agent,
                  sequence=True, event_driven=True)

    task = Task(
        id=str(uuid.uuid4()),
        swarm=swarm,
        input=user_input,
        endless_threshold=5,
        conf=TaskConfig(exit_on_failure=True)
    )

    result = Runners.sync_run_task(task)
    print("finalResult:", result)


if __name__ == "__main__":
    main()
