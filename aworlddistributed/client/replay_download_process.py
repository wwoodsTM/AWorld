import os
import time
import json
import oss2
import re
import aiohttp
import asyncio
import tqdm
import requests
import zipfile
import random

from dotenv import load_dotenv
from oss2.credentials import EnvironmentVariableCredentialsProvider
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

load_dotenv()
REPLAY_DOWNLOAD_URL = os.getenv("REPLAY_DOWNLOAD_URL", "http://example.com/api/v1/tasks/task_replays")

def replay_status_check(log_file_path, gaia_agent, user_id):
    tasks = []
    with open(log_file_path, 'r') as file:
        result = file.readlines()
        for line in result:
            try:
                data = json.loads(line.strip())
                if data["agent_id"] == agent_id and data["user_id"]==user_id:
                    tasks.append({"task_id":data["task_id"], "status":data["status"]})
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue
    return tasks

def replay_download_requests(task_id_list, save_path, time_stamp):
    os.makedirs(save_path, exist_ok=True)
    
    url = REPLAY_DOWNLOAD_URL
    headers = {
        'Content-Type': 'application/json'
    }
    data = {
        "task_id_list": task_id_list
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
    except requests.RequestException as e:
        return (False, f"❌ Network error: {str(e)}")

    if response.status_code == 200:
        zip_file_path = f"{save_path}/{time_stamp}.zip"
        
        with open(zip_file_path, "wb") as f:
            f.write(response.content)
        print(f"ZIP file has been saved as {zip_file_path}")

        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                print("Structure of the ZIP file:")
                zip_ref.printdir()
                zip_ref.extractall(save_path)
                return (True, f"The files have been successfully extracted to the directory: {save_path}")
        except zipfile.BadZipFile:
            return (False, "❌ Error: The downloaded file is not a valid ZIP file.")
    else:
        return (False, f"❌ Error: Request failed, status code: {response.status_code}, Response content: {response.text}")


def replay_process(file_path, num_generations):
    replays_dir = Path(file_path)

    valid_count = 0
    file_count = 0
    message_final_merge = []
    for replay_file in replays_dir.rglob("*.json"):
        file_count += 1
        try:
            with open(replay_file, 'r') as f:
                replay_data = json.load(f)
            last_exp_data = replay_data[-1]['exp_data']
            if last_exp_data:
                actions = last_exp_data.get('actions', [])
                if actions == []:
                    answer_content = "uncompleted"
                else:
                    actions_str = json.dumps(actions)
                    if '<answer>' in actions_str and '</answer>' in actions_str:
                        valid_count += 1
                        print(f"Found answer tag in {replay_file}, count: {valid_count}")
                        match = re.search(r'<answer>(.*?)</answer>', actions_str, re.DOTALL)
                        if match:
                            answer_content = match.group(1)
                            print(f"Answer content: {answer_content}")
                        else:
                            answer_content = ""
                            print("No answer content found.")

                message_final = []
                message = last_exp_data["messages"]
                for i in range(len(message)):
                    if message[i]["role"] in ["system", "user"]:
                        if message[i]["role"] == "user":
                            split_index = message[i]["content"].find('\nHere are the step-by-step hints provided for you')
                            if split_index != -1:
                                message[i]["content"] = message[i]["content"][:split_index]
                        message_final.append(
                            {
                                "role": message[i]["role"],
                                "content": message[i]["content"],
                            }
                        )
                    elif message[i]["role"] == "assistant" and "tool_calls" in message[i].keys():
                        if message[i]["tool_calls"][0]["function"]["arguments"]:
                            arguments = json.loads(message[i]["tool_calls"][0]["function"]["arguments"])
                        else:
                            arguments = ""
                        function_call = {
                            "name": message[i]["tool_calls"][0]["function"]["name"],
                            "arguments": arguments
                        }
                        if message[i]["content"] != "" and message[i]["content"] is not None:
                            message_final.append(
                                {
                                    "role": "assistant",
                                    "content": message[i]["content"],
                                }
                            )
                        message_final.append(
                            {
                                "role": "tool_call",
                                "content": json.dumps(function_call, ensure_ascii=False),
                            }
                        )
                    elif message[i]["role"] == "tool":
                        last_content = message[i-1]["content"] 
                        if last_content is None:
                            last_content = ""
                        message_final.append(
                            {
                                "role": "tool",
                                "content": message[i]["content"].replace(last_content,""),
                            }
                        )
                try:
                    response = last_exp_data["actions"][0]["policy_info"]
                    message_final.append(
                        {
                            "role": "assistant", 
                            "content": response
                        }
                    )
                    message_final_merge.append((message_final, "success"))
                except:
                    message_final.append({
                        "role": "assistant",
                        "content": "No response was received. Please try again later."
                    })
                    message_final_merge.append((message_final, "length"))
        except Exception as e:
            print(f"Error: {e}")
            continue

    def pad_list_to_length(my_list, num):
        if len(my_list) >= num:
            return my_list[:num]
        else:
            bc_messages = random.choices(my_list, k=num - len(my_list))
            return my_list + bc_messages

    message_final_merge = pad_list_to_length(message_final_merge, num_generations)

    return message_final_merge