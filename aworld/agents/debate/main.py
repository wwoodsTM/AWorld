import os
from typing import Dict, Any, Union, List, Optional

from dotenv import load_dotenv
from pydantic import Field

from aworld.agents.debate.planAgent import user_assignment_prompt, user_assignment_system_prompt, \
    user_debate_system_prompt, user_debate_prompt, DebatePlanAgent
from aworld.agents.debate.search_agent import SearchAgent
from aworld.config import load_config, AgentConfig, TaskConfig
from aworld.core.agent.base import BaseAgent
from aworld.core.agent.swarm import Swarm
from aworld.core.client import Client
from aworld.core.common import Observation, ActionModel, ActionResult
from aworld.core.envs.tool import ToolFactory
from aworld.core.task import Task
from aworld.logs.util import logger


#
# class DebateArena(BaseModel):
#     propositions: list[BaseAgent]
#     opposition: list[BaseAgent]
#     moderator: Optional[BaseAgent]
#     judges: Optional[BaseAgent]
#     display_panel: str
#
#


class DebateAgent(BaseAgent):
    topic: str = Field(default=None, description="The topic of the debate")

    opinion: str = Field(default=None, description="The opinion of the agent")

    planner_agent: BaseAgent = Field(default=None, description="The planner agent")

    search_agent: BaseAgent = Field(default=None, description="The search agent")

    def __init__(self, name: str, topic: str, opinion: str, planner_agent: BaseAgent, search_agent: BaseAgent,
                 conf: AgentConfig):
        conf.name = name
        super().__init__(conf)
        self.topic = topic
        self.opinion = opinion
        self.planner_agent = planner_agent
        self.search_agent = search_agent
        self.steps = 0

    def policy(self, observation: Observation, info: Dict[str, Any] = {}, **kwargs) -> Union[
        List[ActionModel], None]:
        ## step 1: 对方观点
        opponent_claim = observation.content

        ## step 2: 呼叫己方，布置搜索任务，并赋值到observation里面
        messages = [{'role': 'system', 'content': user_assignment_system_prompt},
                    {'role': 'user',
                     'content': user_assignment_prompt.format(topic=self.topic, claim=opponent_claim,
                                                              opinion=self.opinion)}]

        llm_result = self.llm.invoke(
            input=messages,
        )

        search_goal = llm_result.content
        print("search goal:", search_goal)
        observation = Observation(content=search_goal,
                                  action_result=[ActionResult(content='start', keep=True)])  # 拼接observation，作为plan task

        step = 0
        extract_memorys = []
        search_materials = []

        ## step 3: 依据布置的任务（observation），呼叫planAgent：由其决定调用哪个干活agent（search_agent），再由具体干活的agent调用更具体的tool，并不断collect搜索的资料
        # while True:
        for i in range(3):
            plan_policy = self.planner_agent.policy(observation=observation)
            print("plan_policy:", i, plan_policy)

            if plan_policy[0].agent_name == "search_agent":
                goal = plan_policy[0].params['task']
                observation = Observation(content=goal)
                while True:
                    policy = self.search_agent.policy(observation=observation)
                    print("search_agent_policy:", policy)

                    if policy[0].tool_name == '[done]':
                        # print(policy[0].policy_info)
                        # observation = Observation(action_result=[ActionResult(content=policy[0].policy_info, keep=True)])
                        observation = Observation(content=str(policy[0].policy_info),
                                                  action_result=[
                                                      ActionResult(content=str(policy[0].policy_info), keep=True)])
                        break

                    tool = ToolFactory(policy[0].tool_name, conf=load_config(f"{policy[0].tool_name}.yaml"))

                    observation, reward, terminated, _, info = tool.step(policy)

                    print("search_observation:", i, observation)
                    print("search_observation_details:", i, observation.content)
                    search_materials.append(observation.content)


            elif "done" in plan_policy[0].agent_name:
                print("now is done.")
                logger.info(plan_policy[0].params['text'])
                logger.info("task is done.")
                break

            else:
                print("invalid agent name.")
                observation = Observation(content="invalid agent name, please try again", action_result=[
                    ActionResult(content="invalid agent name, please try again.", keep=True)])
                continue

        ## 获得本次搜索返回的内容
        print("search_materials:", search_materials)

        # Save search_materials to a text file
        output_file_path = "search_materials.txt"
        with open(output_file_path, "w", encoding="utf-8") as f:
            for i, material in enumerate(search_materials):
                f.write(f"Search Result #{i + 1}:\n")
                f.write(str(material))
                f.write("\n\n" + "=" * 50 + "\n\n")

        ## step 4: 呼叫己方，布置搜索任务，并赋值到observation里面
        messages = [{'role': 'system', 'content': user_debate_system_prompt},
                    {'role': 'user',
                     'content': user_debate_prompt.format(claim=opponent_claim, player='Michael Jordan',
                                                          search_materials=search_materials)}]

        llm_result = self.llm.invoke(
            input=messages,
        )

        user_response = llm_result.content

        print("user_response:", user_response)


if __name__ == '__main__':
    load_dotenv()

    agentConfig = AgentConfig(
        llm_provider="chatopenai",
        llm_model_name="bailing_moe_plus_function_call",
        llm_base_url=os.environ['LLM_BASE_URL'],
        llm_api_key=os.environ['LLM_API_KEY'],
        max_steps=100,
    )

    planner_agent = DebatePlanAgent(conf=agentConfig)
    search_agent = SearchAgent(conf=agentConfig)

    # Initialize client
    client = Client()

    agent1 = DebateAgent(name="agent1", topic="Who's GOAT? Jordan or Lebron", opinion="Jordan",
                         planner_agent=planner_agent, search_agent=search_agent, conf=agentConfig)
    agent2 = DebateAgent(name="agent2", topic="Who's GOAT? Jordan or Lebron", opinion="Lebron",
                         planner_agent=planner_agent, search_agent=search_agent, conf=agentConfig)

    # Create swarm for multi-agents
    # define (head_node, tail_node) edge in the topology graph
    # NOTE: the correct order is necessary
    swarm = Swarm(agent1, (agent1, agent2), (agent2, agent1))

    # Define a task
    task = Task(input="杭州适合年轻人生活吗?", swarm=swarm, conf=TaskConfig())

    # Run task
    result = client.submit(task=[task])

    print(f"Task completed: {result['success']}")
    print(f"Time cost: {result['time_cost']}")
    print(f"Task Answer: {result['task_0']['answer']}")
