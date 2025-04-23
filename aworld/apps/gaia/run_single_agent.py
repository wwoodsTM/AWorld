# coding: utf-8
# Copyright (c) 2025 inclusionAI.
import argparse
import asyncio
import json
import os
import re
import signal
from contextlib import contextmanager

from aworld.agents.gaia.xy_prompts import single_execute_system_output
from aworld.apps.gaia.utils import _check_task_completed, question_scorer
from aworld.config.common import Agents
from aworld.config.conf import TaskConfig
from aworld.core.agent.base import Agent
from aworld.core.agent.swarm import Swarm
from aworld.core.client import Client
from aworld.core.task import Task
from aworld.dataset.gaia.benchmark import GAIABenchmark
from aworld.logs.util import logger
from aworld.mcp_servers.utils import OpenRouterModel, get_llm_config_from_os_environ


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
    parser = argparse.ArgumentParser(
        description="Launch MCP servers with random port allocation"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--end",
        type=int,
    )
    args = parser.parse_args()
    start_idx = args.start
    end_idx = min(args.start + 40, 165) if args.end is None else args.end

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
        current_dir,
        "results",
        f"claude_result_#{end_idx//40}_{start_idx}_{end_idx}.json",
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
            logger = logger.bind(agent="GAIA", task_id=sample["task_id"])
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
            # planner = PlanAgent(
            #     conf=get_llm_config_from_os_environ(
            #         model_name=OpenRouterModel.CLAUDE_37_SONNET,
            #         name=Agents.PLAN.value,
            #     ),
            #     step_reset=False,
            # )
            executor = Agent(
                conf=get_llm_config_from_os_environ(
                    model_name=OpenRouterModel.CLAUDE_37_SONNET,
                    name=Agents.EXECUTE.value,
                    system_prompt=single_execute_system_output,
                ),
                step_reset=False,
                tool_names=[],
                mcp_servers=["aworld"],
            )

            task = Task(
                input=question,
                agent=executor,
                conf=TaskConfig(task_id=sample["task_id"]),
            )
            # result = asyncio.run(task.run())
            result = client.submit(task=[task])

            answer = result["task_0"]["answer"]
            match = re.search(r"<answer>(.*?)</answer>", answer)
            if match:
                answer = match.group(1)
                logger.info(f"Agent answer: {answer}")
                logger.info(f"Correct answer: {sample['Final answer']}")

                if question_scorer(answer, sample["Final answer"]):
                    logger.success(f"Question {idx} Correct!")
                else:
                    logger.warning(f"Question {idx} Incorrect!")

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
    except KeyboardInterrupt:
        logger.critical("KeyboardInterrupt")
    finally:
        logger.success(f"Results saved to {save_path} with #{len(_results)} recods!")
