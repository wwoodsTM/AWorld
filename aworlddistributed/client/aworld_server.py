import asyncio
import logging
import os
import uuid
import random
import socket

from pathlib import Path
from typing import Any, Dict
from datetime import datetime

from aworlddistributed.client.aworld_client import AworldTask, AworldTaskClient


AWORLD_TASK_CLIENT = AworldTaskClient(
    know_hosts = ["localhost:9999"]
)

def get_local_ip():
    try:
        # build UDP socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # connect to an external address (no need to connect)
        s.connect(("8.8.8.8", 80))
        # get local IP
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"

async def download_with_timerange(know_hosts: list[str], start_time, end_time, save_path):
    # create client
    client = AworldTaskClient(know_hosts = know_hosts)

    # 1. download task results to file
    file_path = await client.download_task_results(
        start_time=start_time,
        end_time=end_time,
        save_path=save_path
    )

    # 2. parse local jsonl file
    local_results = client.parse_task_results_file(save_path)

    # 3. analyze results data
    for result in local_results:
        print(f"Submit User ID: {result['user_id']}, Task ID: {result['task_id']},Status: {result['status']}, Replays: {result['result_data']['replays_file'] if result['result_data'] else ''}")

async def _run_gaia_task(gaia_task: AworldTask, delay: int, background: bool = False) -> None:
    """Run a single Gaia task with the given question ID.

    Args:
        gaia_task_id: The ID of the question to process
    """
    global AWORLD_TASK_CLIENT
    await asyncio.sleep(delay)

    # Submit task to Aworld server
    await AWORLD_TASK_CLIENT.submit_task(gaia_task, background=True)

    # Get and print task result
    task_result = await AWORLD_TASK_CLIENT.get_task_state(task_id=gaia_task.task_id)
    if not background:
        logging.info(f"execute task_result#{gaia_task.task_id} is {task_result.data if task_result else None}")
    else:
        logging.info(f"submit task_result#{gaia_task.task_id} background success, please use task_id get task_result await a moment")


async def _batch_run_gaia_task(gaia_tasks: list[AworldTask]) -> None:
    """Run multiple Gaia tasks in parallel.

    """
    tasks = [
        _run_gaia_task(gaia_task, index * 3, background=True)
        for index, gaia_task in enumerate(gaia_tasks)
    ]
    await asyncio.gather(*tasks)


def add_file_path(task: Dict[str, Any]):
    split = "validation" if task["Annotator Metadata"]["Steps"] != "" else "test"
    if task["file_name"]:
        file_path = Path(f"/app/aworldspace/datasets/gaia_dataset/2023/{split}/" + task["file_name"])
        if file_path.suffix in [".pdf", ".docx", ".doc", ".txt"]:
            task["Question"] += f" Here are the necessary document files: {file_path}"

        elif file_path.suffix in [".jpg", ".jpeg", ".png"]:
            task["Question"] += f" Here are the necessary image files: {file_path}"

        elif file_path.suffix in [".xlsx", "xls", ".csv"]:
            task[
                "Question"
            ] += f" Here are the necessary table files: {file_path}, for processing excel file, you can use the excel tool or write python code to process the file step-by-step and get the information."

        elif file_path.suffix in [".py"]:
            task["Question"] += f" Here are the necessary python files: {file_path}"

        else:
            task["Question"] += f" Here are the necessary files: {file_path}"

    return task


CUSTOM_SYSTEM_PROMPT = f""" **PLEASE CUSTOM IT **"""

async def worker(queue):
    while True:
        gaia_task = await queue.get()
        try:
            await _run_gaia_task(gaia_task, random.randint(0, 60))
        except Exception as e:
            logging.error(f"Task {gaia_task.task_id} failed: {e}")
        finally:
            queue.task_done()

async def aworld_rollout_manager(gaia_task_ids):
    print(len(gaia_task_ids))

    custom_mcp_servers = [
        "e2b-code-server",
        "terminal-controller",
        "excel",
        "calculator",
        "ms-playwright",
        "audio_server",
        "image_server",
        "google-search",
        # "video_server",
    ]
    queue = asyncio.Queue(maxsize=5)
    workers = [asyncio.create_task(worker(queue)) for _ in range(5)]

    for i in range(len(gaia_task_ids)):
        gaia_task_id = gaia_task_ids[i]
        task_id = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + gaia_task_id + "_" + str(uuid.uuid4())
        gaia_task = AworldTask(
            task_id=task_id,
            agent_id="gaia_agent",
            agent_input=gaia_task_id,
            session_id="session_id",
            user_id="SYSTEM",
            client_id=get_local_ip(),
            mcp_servers=custom_mcp_servers,
            max_retries=1,
            ext_info={
                "llm_timeout": 900
            },
        )
        await queue.put(gaia_task)

    await queue.join()

    for w in workers:
        w.cancel()