# coding: utf-8
# Copyright (c) 2025 inclusionAI.

import json
import os

from aworld.agents.gaia.agent import ExecuteAgent, PlanAgent
from aworld.agents.gaia.xy_prompts import *
from aworld.apps.gaia.utils import _check_task_completed, question_scorer
from aworld.config.common import Agents
from aworld.config.conf import AgentConfig, TaskConfig
from aworld.core.agent.swarm import Swarm
from aworld.core.client import Client
from aworld.core.task import Task
from aworld.dataset.gaia.benchmark import GAIABenchmark
from aworld.logs.util import logger

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Launch MCP servers with random port allocation"
    )
    parser.add_argument(
        "--port",
        type=int,
        help=f"Listening to port. Must be specified.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=0,
    )
    args = parser.parse_args()
    port = args.port
    start_idx = args.n
    end_idx = min(args.n + 40, 165)

    # Initialize client
    client = Client()

    # One sample for example
    gaia_dir = os.path.expanduser("~/Desktop/gaia-benchmark/GAIA")
    dataset = GAIABenchmark(gaia_dir).load()["valid"]

    # Create agents
    llm_api_key = os.getenv("LLM_API_KEY", "")
    llm_base_url = os.getenv("LLM_BASE_URL", "")

    planner = PlanAgent(
        conf=AgentConfig(
            name=Agents.PLAN.value,
            llm_provider="openai",
            llm_model_name="gpt-4o",
            llm_base_url=f"http://localhost:{port}",
            llm_api_key="dummy-key",
            llm_temperature=0.3,
        ),
        step_reset=False,
    )
    executor = ExecuteAgent(
        conf=AgentConfig(
            name=Agents.EXECUTE.value,
            llm_provider="openai",
            llm_model_name="gpt-4o",
            llm_base_url=f"http://localhost:{port}",
            llm_api_key="dummy-key",
            llm_temperature=0.3,
            system_prompt=execute_system_prompt,
        ),
        step_reset=False,
        tool_names=[],
        mcp_servers=["aworld", "google-search", f"playwright_{end_idx//40}"],
    )
    browser = ExecuteAgent(
        conf=AgentConfig(
            name=Agents.BROWSER.value,
            llm_provider="openai",
            llm_model_name="gpt-4o",
            llm_base_url=f"http://localhost:{port}",
            llm_api_key="dummy-key",
            system_prompt=browser_system_prompt,
        ),
        tool_names=[],
        mcp_servers=[f"playwright_{end_idx//40}"],
        step_reset=False,
    )
    swarm = Swarm((planner, executor), sequence=False)

    # Define a task
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(
        current_dir, "results", f"result_#{end_idx//40}_{start_idx}_{end_idx}.json"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            _results = json.load(f)
    else:
        _results = []

    try:
        for idx, sample in enumerate(dataset[start_idx:end_idx]):
            logger.info(
                f">>> ✈️ Progress bar: {str(idx+start_idx)}/{len(dataset)}."
                f"Current task {sample['task_id']}. "
            )

            if _check_task_completed(sample["task_id"], _results):
                logger.info(
                    f"The following task is already completed: \nTaskId: {sample['task_id']}\nQuestion: {sample['Question']}"
                )
                continue

            question = sample["Question"]
            logger.info(f">>> 🤔 Question: {question}")

            task = Task(
                input=question,
                swarm=swarm,
                conf=TaskConfig(task_id=sample["task_id"]),
                endless_threshold=15,
            )
            result = client.submit(task=[task])

            answer = result["task_0"]["answer"]
            logger.info(f"Task completed: {result['success']}")
            logger.info(f"Time cost: {result['time_cost']}")
            logger.info(f"Task Answer: {answer}")
            logger.info(f"Gold Answer: {sample['Final answer']}")
            logger.info(f"Level: {sample['Level']}")

            _result_info = {
                "index": idx + start_idx,
                "task_id": sample["task_id"],
                "question": sample["Question"],
                "level": sample["Level"],
                "model_answer": answer,
                "ground_truth": sample["Final answer"],
                "score": question_scorer(answer, sample["Final answer"]),
            }
            _results.append(_result_info)
            with open(save_path, "w") as f:
                sorted(_results, key=lambda x: x["index"])
                json.dump(_results, f, indent=4, ensure_ascii=False)
    except KeyboardInterrupt:
        logger.critical("KeyboardInterrupt")
    finally:
        logger.success(f"Results saved to {save_path} with #{len(_results)} recods!")
