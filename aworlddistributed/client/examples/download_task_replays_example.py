import aiohttp
import asyncio
from tqdm import tqdm

HOST = "http://localhost:9999/"
async def download_task_replays(task_id_list: list[str], save_path: str = "task_replays.zip"):
    """
    Download task replays asynchronously with progress bar

    Args:
        task_id_list (list[str]): List of task IDs
        save_path (str): Path to save the zip file
    """
    url = f"{HOST}api/v1/tasks/task_replays"
    headers = {"Content-Type": "application/json"}
    data = {"task_id_list": task_id_list}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data, headers=headers) as response:
            if response.status == 200:
                total_size = int(response.headers.get('content-length', 0))

                with open(save_path, "wb") as f, tqdm(
                        desc="📥 Processing",
                        total=total_size,
                        unit='iB',
                        unit_scale=True,
                        unit_divisor=1024,
                ) as pbar:
                    while True:
                        chunk = await response.content.read(1024)
                        if not chunk:
                            break
                        size = f.write(chunk)
                        pbar.update(size)
                print(f"✅ File saved to: {save_path}")
            else:
                error_data = await response.json()
                print(f"❌ Failed to download file: {error_data}")


task_ids = ["task_id1","task_id2"]
asyncio.run(download_task_replays(task_ids))