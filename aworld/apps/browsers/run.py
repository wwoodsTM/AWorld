# coding: utf-8
# Copyright (c) 2025 inclusionAI.
from aworld.core.envs.tool import ToolFactory
from aworld.core.agent.swarm import Swarm

from aworld.core.client import Client
from aworld.core.common import Agents, Tools
from aworld.core.task import Task
from aworld.agents.browser.agent import BrowserAgent
from aworld.config.conf import AgentConfig
from aworld.agents.browser.config import BrowserAgentConfig
from aworld.virtual_environments import BrowserTool
from aworld.virtual_environments.conf import BrowserToolConfig

if __name__ == '__main__':
    client = Client()

    inner_llm_model_config=ModelConfig(
        llm_provider="openai",
        llm_model_name="gpt-4o-mini",
        llm_temperature=0.3,
        # llm_api_key="sk-duKYX5S7l5jtHPpC5277469201Af4596B244F20b3eFfA7F4",
        # llm_base_url="https://aihubmix.com/v1",
        # llm_api_key="sk-zk269394c1f9a516d4a292c560e3b75b03267e26d31c1055",
        # llm_base_url="https://api.zhizengzeng.com/v1",
        llm_api_key="dummy-key",
        llm_base_url="http://30.230.177.89:5000",
        max_input_tokens = 128000
    )

    browser_tool_config = BrowserToolConfig(window_w=1280, window_h=720, keep_browser_open=True,inner_llm_model_config=inner_llm_model_config)

    agent_config = BrowserAgentConfig(
        agent_name=Agents.BROWSER.value,
        llm_provider="openai",
        llm_model_name="gpt-4o",
        llm_num_ctx=32000,
        llm_temperature=1,
        # llm_api_key="sk-duKYX5S7l5jtHPpC5277469201Af4596B244F20b3eFfA7F4",
        # llm_base_url="https://aihubmix.com/v1",
        # llm_api_key="sk-zk269394c1f9a516d4a292c560e3b75b03267e26d31c1055",
        # llm_base_url="https://api.zhizengzeng.com/v1",
        llm_api_key="dummy-key",
        llm_base_url="http://30.230.177.89:5000",
        max_actions_per_step=10
    )

    # agent_config = AgentConfig(
    #     agent_name=Agents.BROWSER.value,
    #     llm_provider="chatopenai",
    #     llm_model_name="gpt-4o-mini",
    #     llm_num_ctx=32000,
    #     llm_temperature=1,
    #     max_actions_per_step=10,
    #     max_steps=100,
    # )
    

    
    

    task_config = {
        'max_steps': 100,
        'max_actions_per_step': 100
    }

#     client.submit(
#         GeneralTask(input="""step1: first go to https://www.dangdang.com/ and search for 'the little prince' and rank by sales from high to low, get the first 5 results and put the products info in memory.
# step 2: write each product's title, price, discount, and publisher information to a fully structured HTML document with write_to_file, ensuring that the data is presented in a table with visible grid lines.
# step3: open the html file in browser by go_to_url""",
#                     swarm=Swarm(BrowserAgent(conf=agent_config)),
#                     tools=[BrowserTool(conf=browser_tool_config)],
#                     task_config=task_config))

    # client.submit(
    #     GeneralTask(input="""访问www.baidu.com，搜索姚明的信息，找到他的百度百科介绍页，打开并将页面html存到本地""",
    #                 swarm=Swarm(BrowserAgent(conf=agent_config)),
    #                 tools=[BrowserTool(conf=browser_tool_config)],
    #                 task_config=task_config))

    client.submit(
        Task(input="""step1: first go to https://www.dangdang.com/ and search for 'the little prince' and rank by sales from high to low, get the first 5 results and put the products info in memory.
    step 2: write each product's title, price, discount, and publisher information to a fully structured HTML document with write_to_file, ensuring that the data is presented in a table with visible grid lines.
    step3: open the html file in browser by go_to_url""",
                    swarm=Swarm(BrowserAgent(conf=agent_config)),
                    tools=[ToolFactory(Tools.BROWSER.value, conf=browser_tool_config)],
                    task_config=task_config))