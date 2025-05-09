import argparse
import json
import logging
import os
import re
import traceback
from typing import Any, Dict, List

from dotenv import load_dotenv

from aworld.config.conf import AgentConfig, TaskConfig
from aworld.core.agent.base import Agent
from aworld.core.task import Task
from aworld.runner import Runners
from examples.gaia.prompt import system_prompt
from examples.gaia.utils import add_file_path, load_dataset_meta, report_results

parser = argparse.ArgumentParser()
parser.add_argument(
    "--start",
    type=int,
    default=0,
    help="Start index of the dataset",
)
parser.add_argument(
    "--end",
    type=int,
    default=20,
    help="End index of the dataset",
)
parser.add_argument(
    "--q",
    type=str,
    help="Question Index, e.g., 0-0-0-0-0. Highest priority: override other arguments if provided.",
)
args = parser.parse_args()


def setup_logging():
    logging_logger = logging.getLogger()
    logging_logger.setLevel(logging.INFO)

    log_file_name = (
        f"/super_agent_{args.q}.log"
        if args.q
        else f"/super_agent_{args.start}_{args.end}.log"
    )
    file_handler = logging.FileHandler(
        os.getenv(
            "LOG_FILE_PATH",
            "run_super_agent.log",
        )
        + log_file_name,
        mode="a",
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    logging_logger.addHandler(file_handler)


if __name__ == "__main__":
    load_dotenv()
    setup_logging()

    gaia_dataset_path = os.getenv("GAIA_DATASET_PATH", "./gaia_dataset")
    full_dataset = load_dataset_meta(gaia_dataset_path)

    agent_config = AgentConfig(
        llm_provider="openai",
        llm_model_name=os.getenv("LLM_MODEL_NAME", "gpt-4o"),
        llm_api_key=os.getenv("LLM_API_KEY", "your_openai_api_key"),
        llm_base_url=os.getenv("LLM_BASE_URL", "your_openai_base_url"),
    )
    super_agent = Agent(
        conf=agent_config,
        name="gaia_super_agent",
        system_prompt=system_prompt,
        mcp_servers=[
            "e2b-server",
            "filesystem",
            "terminal-controller",
            "excel",
            "calculator",
            # "playwright",
            "audio_server",
            "image_server",
            "video_server",
            "search_server",
            "download_server",
            "document_server",
            "browser_server",
            "youtube_server",
            "reasoning_server",
        ],
    )

    if os.path.exists(os.getenv("LOG_FILE_PATH") + "/results.json"):
        with open(
            os.getenv("LOG_FILE_PATH") + "/results.json", "r", encoding="utf-8"
        ) as results_f:
            results: List[Dict[str, Any]] = json.load(results_f)
    else:
        results: List[Dict[str, Any]] = []
    try:
        for i, dataset_i in enumerate(full_dataset[args.start : args.end]):
            # specify `task_id`
            if args.q and args.q != dataset_i["task_id"]:
                continue
            # only valid for args.q==None
            if not args.q:
                # blacklist
                if dataset_i["task_id"] in (
                    "676e5e31-a554-4acc-9286-b60d90a92d26",
                    "46719c30-f4c3-4cad-be07-d5cb21eee6bb",
                ):
                    continue

                # pass
                if any(
                    # Question Done and Correct
                    (result["task_id"] == dataset_i["task_id"] and result["is_correct"])
                    for result in results
                ) or any(
                    # Question Done and Incorrect, but Level is 3
                    (
                        result["task_id"] == dataset_i["task_id"]
                        and not result["is_correct"]
                        and dataset_i["Level"] == 3
                    )
                    for result in results
                ):
                    continue

            # run
            try:
                logging.info(f"Start to process: {dataset_i['task_id']}")
                logging.info(f"Detail: {dataset_i}")
                logging.info(f"Question: {dataset_i['Question']}")
                logging.info(f"Level: {dataset_i['Level']}")
                logging.info(f"Tools: {dataset_i['Annotator Metadata']['Tools']}")

                question = add_file_path(dataset_i, gaia_dataset_path)["Question"]

                result = Runners.sync_run_task(
                    task=Task(input=question, agent=super_agent, conf=TaskConfig())
                )

                match = re.search(r"<answer>(.*?)</answer>", result["task_0"]["answer"])
                if match:
                    answer = match.group(1)
                    logging.info(f"Agent answer: {answer}")
                    logging.info(f"Correct answer: {dataset_i['Final answer']}")

                    if answer == dataset_i["Final answer"]:
                        logging.info(f"Question {i} Correct!")
                    else:
                        logging.info("Incorrect!")

                # Create the new result record
                new_result = {
                    "task_id": dataset_i["task_id"],
                    "level": dataset_i["Level"],
                    "question": question,
                    "answer": answer,
                    "response": dataset_i["Final answer"],
                    "is_correct": answer == dataset_i["Final answer"],
                }

                # Check if this task_id already exists in results
                existing_index = next(
                    (
                        i
                        for i, result in enumerate(results)
                        if result["task_id"] == dataset_i["task_id"]
                    ),
                    None,
                )

                if existing_index is not None:
                    # Update existing record
                    results[existing_index] = new_result
                    logging.info(
                        f"Updated existing record for task_id: {dataset_i['task_id']}"
                    )
                else:
                    # Append new record
                    results.append(new_result)
                    logging.info(
                        f"Added new record for task_id: {dataset_i['task_id']}"
                    )

            except Exception as e:
                logging.error(f"Error processing {i}: {traceback.format_exc()}")
                continue
    except KeyboardInterrupt:
        pass
    finally:
        # report
        report_results(results)
        with open(
            os.getenv("LOG_FILE_PATH") + "/results.json", "w", encoding="utf-8"
        ) as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
