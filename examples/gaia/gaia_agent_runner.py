import argparse
import json
import logging
import os
import re
import sys
import traceback
from typing import Any, Dict, List

from dotenv import load_dotenv

from aworld.config.conf import AgentConfig, TaskConfig
from aworld.core.agent.base import Agent
from aworld.core.task import Task
from aworld.runner import Runners
from examples.gaia.prompt import system_prompt
from examples.gaia.utils import (
    add_file_path,
    load_dataset_meta_dict,
    question_scorer,
)

aworld_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(aworld_path)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


# Create log directory if it doesn't exist
if not os.path.exists(os.getenv("LOG_FILE_PATH")):
    os.makedirs(os.getenv("LOG_FILE_PATH"))

with open("gaia_agent_runner.log", "w") as f:
    f.write(f"Start gaia agent runner!\n")

if __name__ == "__main__":
    print(f"Start gaia agent runner!")
    
    with open("gaia_agent_runner.log", "w") as f:
        f.write(f"Start gaia agent runner!\n")
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--prompt", type=str, default="")
    args = parser.parse_args()

    logger.info(f"Gaia agent runner parameter: provider={args.provider}, model={args.model}, prompt={args.prompt}")

    load_dotenv()

    gaia_dataset_path = os.getenv("GAIA_DATASET_PATH", "./GAIA/2023/test")
    full_dataset = load_dataset_meta_dict(gaia_dataset_path)
    logger.info(f"Total questions: {len(full_dataset)}")

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
            # "filesystem",
            "terminal-controller",
            "excel",
            "calculator",
            "ms-playwright",
            "audio_server",
            "image_server",
            "video_server",
            "search_server",
            "download_server",
            "document_server",
            # "browser_server",
            "youtube_server",
            "reasoning_server",
        ],
    )

    try:
        prompt = args.prompt

        logger.info(f"Start to process: {prompt}")

        json_data = json.loads(prompt)
        task_id = json_data["task_id"]

        data_item = full_dataset[task_id]

        question = add_file_path(
            data_item, file_path=gaia_dataset_path
        )["Question"]

        result = Runners.sync_run_task(
            task=Task(input=question, agent=super_agent, conf=TaskConfig())
        )

        match = re.search(r"<answer>(.*?)</answer>", result["task_0"]["answer"])
        if match:
            answer = match.group(1)
            logger.info(f"Agent answer: {answer}")
            logger.info(f"Correct answer: {data_item['Final answer']}")

            if question_scorer(answer, data_item["Final answer"]):
                logger.info(f"Question {task_id} Correct!")
            else:
                logger.info(f"Question {task_id} Incorrect!")

        # Create the new result record
        new_result = {
            "task_id": data_item["task_id"],
            "level": data_item["Level"],
            "question": question,
            "answer": data_item["Final answer"],
            "response": answer,
            "is_correct": question_scorer(answer, data_item["Final answer"]),
        }

        logger.info(f"Result: {new_result}")
    except Exception as e:
        logger.error(f"Error processing {args.prompt}: {traceback.format_exc()}")
