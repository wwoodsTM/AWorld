import sys
import os
from pathlib import Path
import logging

from aworld.agents import AndroidAgent
from aworld.env_secrets import secrets

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 获取项目根目录路径 (operator 目录)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)  # 确保我们的路径优先

from langchain_openai import ChatOpenAI
import asyncio
from dotenv import load_dotenv
load_dotenv()

async def main():
    # 创建历史保存目录
    history_dir = Path(root_dir) / "agents" / "android_agents" / "history"
    history_dir.mkdir(exist_ok=True)
    history_path = str(history_dir / "agent_history.json")

    logger = logging.getLogger(__name__)
    logger.info("正在初始化Android Agent...")

    agent = AndroidAgent(
        task="Open Chrome app and search hangzhou weather",
        avd_name="Medium_Phone_API_35",
        llm=ChatOpenAI(
            model=secrets.android_model_id,
            openai_api_key=secrets.android_openai_api_key,
            base_url=secrets.android_base_url
        ),
        # 新增的设置
        max_failures=3,
        retry_delay=5,
        save_history=True,
        history_path=history_path,
        max_actions_per_step=10,
        validate_output=True,
        message_context="Android automation task for map navigation"
    )
    
    try:
        logger.info("开始执行任务...")
        await agent.run(max_steps=20)  # 限制最大步数为20
    except Exception as e:
        logger.error(f"任务执行出错: {str(e)}")
    finally:
        logger.info("任务结束，正在停止Agent...")
        agent.stop()

if __name__ == "__main__":
    asyncio.run(main()) 