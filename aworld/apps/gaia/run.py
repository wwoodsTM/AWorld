# coding: utf-8
# Copyright (c) 2025 inclusionAI.

import json
import os
import signal
import time
from contextlib import contextmanager

from sympy.parsing.sympy_parser import null

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


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


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
    parser.add_argument(
        "--end",
        type=int,
    )
    args = parser.parse_args()
    port = args.port
    start_idx = args.n
    end_idx = min(args.n + 40, 165) if args.end is None else args.end

    # Initialize client
    client = Client()

    # One sample for example
    gaia_dir = os.path.expanduser("~/Desktop/gaia-benchmark/GAIA")
    dataset = GAIABenchmark(gaia_dir).load()["valid"]

    # Create agents
    llm_api_key = os.getenv("LLM_API_KEY", "")
    llm_base_url = os.getenv("LLM_BASE_URL", "")

    # Define a task
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(
        current_dir, "results", f"result_#{end_idx//40}_{start_idx}_{end_idx}.json"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        try:
            with open(save_path, "r") as f:
                _results = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON format in {save_path}. Creating a new file.")
            _results = []
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

            # Set a time limit for processing each sample (e.g., 10 minutes = 600 seconds)
            try:
                with time_limit(15 * 60):  # Adjust the timeout value as needed
                    planner = PlanAgent(
                        conf=AgentConfig(
                            name=Agents.PLAN.value,
                            llm_provider="openai",
                            llm_model_name="gpt-4o",
                            llm_base_url=f"http://localhost:{port}",
                            llm_api_key="dummy-key",
                            llm_temperature=0.15,
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
                            llm_temperature=0.15,
                            system_prompt=execute_system_prompt,
                        ),
                        step_reset=False,
                        tool_names=[],
                        mcp_servers=["aworld_0"],
                    )

                    swarm = Swarm((planner, executor), sequence=False)
                    task = Task(
                        input=question,
                        swarm=swarm,
                        conf=TaskConfig(task_id=sample["task_id"]),
                        endless_threshold=6,
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
                        # Ensure all entries have the 'index' key before sorting
                        if all("index" in result for result in _results):
                            _results = sorted(_results, key=lambda x: x["index"])
                        json.dump(_results, f, indent=4, ensure_ascii=False)
            except TimeoutException:
                logger.warning(
                    f"Processing sample {sample['task_id']} timed out. Moving to next sample."
                )
                _result_info = {
                    "index": idx + start_idx,
                    "task_id": sample["task_id"],
                    "question": sample["Question"],
                    "level": sample["Level"],
                    "model_answer": "TIMEOUT",
                    "ground_truth": sample["Final answer"],
                    "score": False,
                }
                _results.append(_result_info)
                with open(save_path, "w") as f:
                    if all("index" in result for result in _results):
                        _results = sorted(_results, key=lambda x: x["index"])
                    json.dump(_results, f, indent=4, ensure_ascii=False)
                continue
    except KeyboardInterrupt:
        logger.critical("KeyboardInterrupt")
    finally:
        _result_info = {
            "index": idx + start_idx,
            "task_id": sample["task_id"],
            "question": sample["Question"],
            "level": sample["Level"],
            "model_answer": None,
            "ground_truth": sample["Final answer"],
            "score": False,
        }
        _results.append(_result_info)
        with open(save_path, "w") as f:
            json.dump(_results, f, indent=4, ensure_ascii=False)
        logger.success(f"Results saved to {save_path} with #{len(_results)} recods!")
