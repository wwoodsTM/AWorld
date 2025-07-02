import asyncio
from datetime import datetime
import os
import oss2
from tqdm import tqdm

from client.aworld_client import AworldTaskClient
from aworld.logs.util import logger

OSS_PREFIX = 'aworld'
async def download_replay_file(replay_file_url: str, pbar: tqdm, semaphore: asyncio.Semaphore, local_dir: str):
    async with semaphore:
        oss_url = os.path.join(OSS_PREFIX, replay_file_url)

        try:
            auth = oss2.Auth(os.environ["OSS_ACCESS_KEY_ID"], os.environ["OSS_ACCESS_KEY_SECRET"])
            bucket_name = os.environ["OSS_BUCKET"]
            endpoint = os.environ["OSS_ENDPOINT"]
            bucket = oss2.Bucket(auth, endpoint, bucket_name)

            os.makedirs(local_dir, exist_ok=True)
            filename = os.path.basename(replay_file_url)
            local_path = os.path.join(local_dir, filename)

            await asyncio.to_thread(bucket.get_object_to_file, oss_url, local_path)
            logger.info(f"Successfully downloaded {oss_url} to {local_path}")
        except Exception as e:
            logger.error(f"Failed to download {oss_url}: {e}")
        finally:
            pbar.update(1)

async def download_with_timerange(know_hosts: list[str], start_time, end_time, user_id=None, agent_id=None, status=None, download_replay=True):
    # create client
    client = AworldTaskClient(know_hosts = know_hosts)
    replay_dir = "results/" + str(datetime.now().strftime("%Y%m%d_%H%M%S")) + "/replays"
    save_path = "results/" + str(datetime.now().strftime("%Y%m%d_%H%M%S")) + "/january_tasks.jsonl"

    # 1. download task results to file
    file_path = await client.download_task_results(
        start_time=start_time,
        end_time=end_time,
        user_id=user_id,
        agent_id=agent_id,
        status=status,
        save_path=save_path
    )

    # 2. parse local jsonl file
    local_results = client.parse_task_results_file(save_path)

    # 3. group by user_id and status and print stats
    from collections import defaultdict
    import json

    download_replay_urls = []
    task_stats = defaultdict(lambda: defaultdict(int))
    for result in local_results:
        user_id = result.get('user_id', 'unknown_user')
        status = result.get('status', 'unknown_status')
        task_stats[user_id][status] += 1
        if result['status'] == 'SUCCESS':
            if replay_file := result.get("result_data", {}).get("replays_file"):
                download_replay_urls.append(replay_file)

    print("\n--- Task Statistics ---")
    print(json.dumps(task_stats, indent=2))
    print("-----------------------\n")

    if not download_replay:
        logger.info(f"🚀 [TASK_DOWNLOAD] download_replay is False, skip download replay")
        return
    
    if download_replay_urls:
        semaphore = asyncio.Semaphore(4)
        logger.info(f"Downloading {len(download_replay_urls)} replay files with 4 concurrent workers.")
        with tqdm(total=len(download_replay_urls), desc="Downloading Replays") as pbar:
            download_tasks = [download_replay_file(url, pbar, semaphore, replay_dir) for url in download_replay_urls]
            await asyncio.gather(*download_tasks)


if __name__ == '__main__':
    asyncio.run(download_with_timerange(know_hosts= ["http://localhost:9999"],
                    start_time="2025-05-02 04:00:00",
                    end_time="2025-07-02 18:00:00",
                    # agent_id='gaia_agent',
                    # status='SUCCESS',
                    download_replay=True))
