# coding: utf-8
# Copyright (c) 2025 inclusionAI.

import time

from aworld.core.common import Tools

from aworld.core.client import Client
from aworld.core.agent.base import BaseAgent
from aworld.agents.gym.agent import GymDemoAgent as GymAgent

from aworld.config.conf import AgentConfig
from aworld.logs.util import logger
from aworld.core.envs.tool import AsyncTool, ToolFactory
from aworld.core.task import Task
from aworld.virtual_environments import OpenAIGym


async def async_run_gym_game(agent: BaseAgent, tool: AsyncTool):
    gym_tool = tool
    logger.info('observation space: {}'.format(gym_tool.env.observation_space))
    logger.info('action space: {}'.format(gym_tool.env.action_space))
    logger.info('rende mode: {}'.format(gym_tool.env.render_mode))

    try:
        # init env tool state
        state, info = await gym_tool.reset()
        while True:
            # render
            await gym_tool.render()
            # agent policy action, also can use llm, only an example
            action = await agent.async_policy(state, info)
            logger.info(f"action: {action}")
            # env tool state and reward info based on action
            state, reward, done, truncated, info = await gym_tool.step(action=action)
            logger.info(f'state: {state}; reward: {reward}')

            if done:
                logger.info("game done!")
                break
            time.sleep(1)
    finally:
        await gym_tool.close()


def main():
    # use default config
    gym_tool = ToolFactory(Tools.GYM.value)
    agent = GymAgent(AgentConfig())

    # can run tasks like this:
    client = Client()
    task = Task(agent=agent, tools=[gym_tool])
    res = client.submit([task], parallel=False)
    return res


if __name__ == "__main__":
    main()

    # We can run the task use utility method, as follows:
    # async run gym
    # agym_tool = AOpenAIGym({'env_id': 'CartPole-v1'})
    # asyncio.run(async_run_gym_game(agent=agent, tool=agym_tool))
