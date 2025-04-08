# coding: utf-8
import os
import json
from typing import Dict, Any, List, Union

from aworld.config.conf import AgentConfig
from aworld.core.agent.base import AgentFactory, BaseAgent
from aworld.core.common import Observation, ActionModel
from aworld.core.envs.tool_desc import get_tool_desc_by_name
from aworld.models.utils import tool_desc_transform
from aworld.core.envs.tool import ToolFactory
from aworld.config.conf import load_config

sys_prompt = "You are a helpful write agent."

prompt = """
Please act as a write agent, constructing appropriate keywords and search terms, using search toolkit to collect relevant information, including urls, webpage snapshots, etc.

Here are the write task: {task}

Here is the reference information: {reference}

Based on the above write task, please help me summarize the reference information in an organized way, without losing key information.
"""
#
# response_format = """1. RESPONSE FORMAT: You must ALWAYS respond with valid JSON in this exact format:
# {"action":[{{"one_action_name": {{// action-specific parameter}}}}, // ... more actions in sequence]}
# """
response_format = """ """

@AgentFactory.register(name='write_agent', desc="write agent")
class WriteAgent(BaseAgent):

    def __init__(self, conf: AgentConfig, **kwargs):
        super(WriteAgent, self).__init__(conf, **kwargs)
        self.tool_desc = tool_desc_transform({'write_tool': get_tool_desc_by_name('write_tool')})

    # Step3
    def name(self) -> str:
        return 'write_agent'

    def policy(self, observation: Observation, info: Dict[str, Any] = {}, **kwargs) -> Union[
        List[ActionModel], None]:

        if observation.action_result is not None and len(observation.action_result) != 0 and observation.action_result[0].is_done:
            self._finished = True

            return [ActionModel(tool_name="[done]", policy_info=observation.content)]

        messages = [{'role': 'system', 'content': sys_prompt},
                    {'role': 'user', 'content': prompt.format(task=observation.content['task'], reference=observation.content['refer'], tool_desc=self.tool_desc) + response_format}]
        print("writeAgent_messages: ", messages)

        llm_result = self.llm.invoke(
            input=messages,
        )

        tool_calls = llm_result.content
        # print("tool_calls: ", tool_calls)
        return tool_calls
        # tool_calls = llm_result.choices[0].message.tool_calls
        # if tool_calls:
        # return self._result(tool_calls)
        # else:
        #     # use variable `content` to do something if there is no need to call the tools
        #     # Unable to use tools, replan by plan agent
        #     return [ActionModel(agent_name=Agents.PLAN.value, policy_info=content)]

    def _result(self, data):

        data = json.loads(data.replace("```json","").replace("```",""))
        actions = data.get("action", [])
        parsed_results = []

        for action in actions:
            # 遍历 action 中的键值对
            for key, value in action.items():
                # 分割 action_name 和 tool_name
                if "__" in key:
                    tool_name, action_name = key.split("__", 1)

                # 提取 params
                params = value

                # 将解析结果存入列表
                parsed_results.append(ActionModel(tool_name=tool_name,
                                       action_name=action_name,
                                       params=params))
        return parsed_results


if __name__ == '__main__':
    agentConfig = AgentConfig(
        llm_provider="chatopenai",
        llm_model_name="gpt-4o",
        llm_base_url="http://localhost:5000",
        llm_api_key="dummy-key",
    )

    writeAgent = WriteAgent(agentConfig)

    goal = "Write a detailed argument supporting Michael Jordan as the best basketball player ever, beyond the factors of speed, power, IQ, and commercial value, and opposing the claim that Lebron James is the best based on those factors."
#     refer = """
# {
#   "cherry_blossom_forecast_2025": {
#     "last_update": "March 21, 2025",
#     "overview": "The cherry blossom season, known as sakura season, is a magical time in Japan that marks the arrival of spring. It is celebrated by enjoying hanami, the tradition of viewing and enjoying the beauty of the cherry bloom.",
#     "forecast_details": {
#       "general_timing": "The cherry blossom season in Japan takes place in spring, usually between March and April.",
#       "bloom_progression": "The cherry blossom usually starts from the south, at the end of March, in Kyushu, and blooms northward until early May in Hokkaido.",
#       "cities_forecast": [
#         {
#           "city": "Kagoshima",
#           "estimated_opening": "March 25",
#           "estimated_full_bloom": "April 5"
#         },
#         {
#           "city": "Fukuoka",
#           "estimated_opening": "March 26",
#           "estimated_full_bloom": "April 4"
#         },
#         {
#           "city": "Kochi",
#           "estimated_opening": "March 25",
#           "estimated_full_bloom": "April 1"
#         },
#         {
#           "city": "Hiroshima",
#           "estimated_opening": "March 27",
#           "estimated_full_bloom": "April 6"
#         },
#         {
#           "city": "Tokyo",
#           "estimated_opening": "March 24",
#           "estimated_full_bloom": "March 30"
#         },
#         {
#           "city": "Nagoya",
#           "estimated_opening": "March 26",
#           "estimated_full_bloom": "April 4"
#         },
#         {
#           "city": "Wakayama",
#           "estimated_opening": "March 27",
#           "estimated_full_bloom": "April 4"
#         },
#         {
#           "city": "Kyoto",
#           "estimated_opening": "March 28",
#           "estimated_full_bloom": "April 6"
#         },
#         {
#           "city": "Osaka",
#           "estimated_opening": "March 29",
#           "estimated_full_bloom": "April 5"
#         },
#         {
#           "city": "Kanazawa",
#           "estimated_opening": "April 3",
#           "estimated_full_bloom": "April 9"
#         },
#         {
#           "city": "Sendai",
#           "estimated_opening": "April 5",
#           "estimated_full_bloom": "April 10"
#         },
#         {
#           "city": "Nagano",
#           "estimated_opening": "April 10",
#           "estimated_full_bloom": "April 15"
#         },
#         {
#           "city": "Aomori",
#           "estimated_opening": "April 18",
#           "estimated_full_bloom": "April 22"
#         },
#         {
#           "city": "Sapporo",
#           "estimated_opening": "April 26",
#           "estimated_full_bloom": "April 30"
#         }
#       ],
#       "source": "Japan Meteorological Corporation"
#     },
#     "travel_tips": "If you’re visiting Japan during the 2025 Easter holiday, consider visiting the Tohoku region to catch the cherry blossoms in their late-season bloom, in late April.",
#     "hanami_traditions": "Hanami often extends into the evening, with yozakura (night cherry blossoms) illuminated by lanterns and lights. Sakura festivals take place across the country, featuring traditional performances and food stalls offering sakura-themed delicacies."
#   }
# }
# ```
#
# ```json
# {
#   "tea_ceremony": [
#     {
#       "title": "Kyoto: Tea Tasting in a Traditional Teahouse in Kiyomizu",
#       "duration": "45 minutes",
#       "rating": 4.9,
#       "price": "40 USD per person",
#       "link": "jing-du-l96826/jing-du-zai-qing-shui-de-chuan-tong-cha-guan-pin-cha-t551173/?ranking_uuid=2b467017-4eae-4445-980c-f4acd3bf71ec"
#     },
#     {
#       "title": "Kyoto: Nishiki Market Tea Ceremony and Koto Performance",
#       "duration": "1 hour",
#       "rating": 4.7,
#       "price": "51 USD per person",
#       "link": "jing-du-l96826/jing-du-jin-shi-chang-cha-dao-yu-gu-zheng-biao-yan-t788883/?ranking_uuid=7d2cdca7-487a-4f82-b70b-42a6e4c74d09"
#     },
#     {
#       "title": "Kyoto: 45-Minute Tea Ceremony Experience",
#       "duration": "45 minutes",
#       "rating": 4.3,
#       "price": "22 USD per person",
#       "link": "jing-du-l96826/jing-du-cha-dao-ti-yan-t60894/?ranking_uuid=7d2cdca7-487a-4f82-b70b-42a6e4c74d09"
#     }
#   ],
#   "kendo": [],
#   "zen_meditation": []
# }
#
# : ```json
# {
#   "activities": [
#     {
#       "title": "京都：隐秘寺庙中的禅宗体验",
#       "duration": "2 小时",
#       "type": "小团",
#       "rating": 4.9,
#       "reviews": 269,
#       "price": "120USD",
#       "location": "京都",
#       "link": "jing-du-l96826/si-ren-yin-mi-si-miao-zhong-de-gao-ji-shan-zong-ti-yan-t521265/?ranking_uuid=0f6f3654-426f-4844-85e1-c4972393e286"
#     },
#     {
#       "title": "京都：与僧侣在私人寺庙进行禅修",
#       "duration": "70 分钟",
#       "type": "小团",
#       "rating": 5,
#
# 2025-03-26 22:52:17,290 - common - INFO - [agent] Content (continued):      "reviews": 102,
#       "price": "67USD",
#       "location": "京都",
#       "link": "jing-du-l96826/jing-du-yin-mi-si-miao-de-shan-zong-ming-xiang-t622825/?ranking_uuid=0f6f3654-426f-4844-85e1-c4972393e286"
#     },
#     {
#       "title": "京都寺庙禅修和庭院游览，附带午餐",
#       "duration": "4.5 小时",
#       "type": "小团",
#       "rating": 4.9,
#       "reviews": 24,
#       "price": "95USD",
#       "location": "京都",
#       "link": "jing-du-fu-l240/jing-du-si-miao-shan-xiu-he-ting-yuan-you-lan-fu-dai-wu-can-t385301/?ranking_
# 2025-03-26 22:52:17,290 - common - INFO - [agent] Content (continued): uuid=0f6f3654-426f-4844-85e1-c4972393e286"
#     },
#     {
#       "title": "京都：剑道和武士制服和装备体验",
#       "duration": "2 小时",
#       "rating": 4.9,
#       "reviews": 272,
#       "price": "107USD",
#       "location": "京都",
#       "link": "jing-du-l96826/jing-du-jian-dao-he-wu-shi-ti-yan-t698276/?ranking_uuid=0f6f3654-426f-4844-85e1-c4972393e286"
#     },
#     {
#       "title": "京都：剑道武士体验之旅",
#       "duration": "2.5 小时",
#       "rating": 5,
#       "reviews": 12,
#       "price": "120USD",
#       "location": "京都",
#       "l
# 2025-03-26 22:52:17,290 - common - INFO - [agent] Content (continued): ink": "jing-du-l96826/jing-du-jian-dao-wu-shi-ti-yan-zhi-lu-t540433/?ranking_uuid=0f6f3654-426f-4844-85e1-c4972393e286"
#     },
#     {
#       "title": "京都：通过茶道和坐禅进行禅修",
#       "duration": "2 小时",
#       "type": "小团",
#       "rating": 5,
#       "reviews": 35,
#       "price": "87USD",
#       "location": "京都",
#       "link": "jing-du-l96826/jing-du-tong-guo-cha-dao-he-zuo-shan-jin-xing-shan-xiu-t729159/?ranking_uuid=0f6f3654-426f-4844-85e1-c4972393e286"
#     }
#   ]
# }
#
# : ```json
# {
#   "UjiMatchaTastingExperienceLocations": [
#     {
#       "name": "Chazuna",
#       "description": "Chazuna offers a matcha-making workshop, tea-leaf-grinding, and various crafts, including decorating a tea canister with high-quality Japanese washi paper.",
#       "address": "203-1 Todo Maruyama, Uji City, Kyoto, 611-0013",
#       "access": "4 minute walk from Keihan Uji Station, 12 minute walk from JR Uji Station",
#       "website": "https://chazunayoyaku.rsvsys.jp/events/list",
#       "opening_hours": "Daily, 9 a.m.–5 p.m. (Museum final admission at 4:30 p.m.)"
#     },
#     {
#       "name": "Taiho-an",
#       "description": "Taiho-an is a tea house that offers a relaxed environment to experience a traditional Japanese tea ceremony.",
#       "image": "https://www.kyototourism.org/wp/wp-content/uploads/2023/12/DSCF8737-1024x683.jpg"
#     },
#     {
#       "name": "Incense Kitchen",
#       "description": "Incense Kitchen provides an opportunity to make matcha incense.",
#       "image": "https://www.kyototourism.org/wp/wp-content/uploads/2023/12/DSCF8737-1024x683.jpg"
#     },
#     {
#       "name": "Uji Teahouse, Takumi no Yakata",
#       "description": "Uji Teahouse, Takumi no Yakata offers a tasting experience of the range of Uji tea.",
#       "image": "https://www.kyototourism.org/wp/wp-content/uploads/2023/12/DSCF8737-1024x683.jpg"
#     },
#     {
#       "name": "Fukujuen, Uji Tea Workshop",
#       "description": "Fukujuen, Uji Tea Workshop allows visitors to roast their own hoji-cha tea.",
#       "image": "https://www.kyototourism.org/wp/wp-content/uploads/2023/12/DSCF8737-1024x683.jpg"
#     }
#   ]
# }
#
# ```
#
# : ```json
# {
#   "romantic_proposal_spots": [
#     {
#       "location": "Tokyo",
#       "description": "Tokyo offers several beautiful spots for a romantic proposal during cherry blossom season. Popular locations include Ueno Park, Shinjuku Gyoen, and Chidorigafuchi, where cherry blossoms create a stunning backdrop."
#     },
#     {
#       "location": "Kyoto",
#       "description": "Kyoto is renowned for its historical charm and cherry blossoms. Maruyama Park and the Philosopher's Path are ideal spots for a romantic proposal amidst the blooming sakura."
#     },
#     {
#       "location": "Osaka",
#       "description": "Osaka Castle Park is a picturesque location for a proposal, with cherry blossoms surrounding the historic castle, creating a romantic atmosphere."
#     },
#     {
#       "location": "Hokkaido",
#       "description": "Hokkaido's Matsumae Park is famous for its cherry blossoms and offers a serene setting for a romantic proposal."
#     },
#     {
#       "location": "Nara",
#       "description": "Nara Park, known for its free-roaming deer and cherry blossoms, provides a unique and enchanting setting for a marriage proposal."
#     }
#   ]
# }"""
    refer = "Jordan scores 31.1, while Lebron scores 27 on average. Moreover, Jordan never asks for his superstar friends join his team."
    task = {
        "task": goal,
        "refer": refer,
    }

    observation = Observation(content=task)
    policy = writeAgent.policy(observation=observation)
    print("policy:", policy)
    # while True:
    #     policy = writeAgent.policy(observation=observation)
    #
    #     if policy[0].tool_name == '[done]':
    #         break
    #
    #     tool = ToolFactory(policy[0].tool_name, conf=load_config)
    #
    #     observation, reward, terminated, _, info = tool.step(policy)
    #     print(observation)

    # print("policy_info: ", policy[0].policy_info)