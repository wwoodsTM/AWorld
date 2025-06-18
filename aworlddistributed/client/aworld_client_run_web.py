# Initialize AworldTaskClient with server endpoints
import asyncio
import random
import uuid

from client.aworld_client import AworldTask, AworldTaskClient

AWORLD_TASK_CLIENT = AworldTaskClient(
    know_hosts=["localhost:5555"],
)


async def _run_web_task(web_question_id: str) -> None:
    """Run a single Web task with the given question ID.

    Args:
        web_question_id: The ID of the question to process
    """
    global AWORLD_TASK_CLIENT
    task_id = str(uuid.uuid4())

    # Submit task to Aworld server
    await AWORLD_TASK_CLIENT.submit_task(
        AworldTask(
            task_id=task_id,
            agent_id="playwright_agent",
            agent_input=web_question_id,
            session_id="session_id",
            user_id="SYSTEM",
            max_steps=50,
        )
    )


    # Get and print task result
    task_result = await AWORLD_TASK_CLIENT.get_task_state(task_id=task_id)
    print(task_result)


async def _batch_run_web_task(start_i: int, end_i: int, dataset: str) -> None:
    """Run multiple Web tasks in parallel.

    Args:
        start_i: Starting question ID
        end_i: Ending question ID
    """
    tasks = [
        _run_web_task(dataset + ':' + str(i))
        for i in range(start_i, end_i + 1)
    ]
    await asyncio.gather(*tasks)


async def _batch_run_web_task_with_data(data_list) -> None:
    """Run multiple Web tasks in parallel.

    Args:
        [{task: "", web: ""}] ...
    """
    import json
    tasks = [
        _run_web_task('Task:' + json.dumps(data))
        for data in data_list
    ]
    await asyncio.gather(*tasks)


if __name__ == '__main__':

    # input example
    data_list = [
        {
            "task": "Find the latest news about Netflix stock",
            "web": "google"
        }
    ]
    for i in range(1):
        asyncio.run(_batch_run_web_task_with_data(data_list))