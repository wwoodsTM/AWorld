# coding: utf-8
# Copyright (c) 2025 inclusionAI.

from aworld.config.conf import AgentConfig, TaskConfig
from aworld.core.agent.base import Agent
from aworld.core.task import Task

if __name__ == '__main__':
    agent_config = AgentConfig(
        llm_provider="openai",
        llm_model_name="gpt-4o",
        # llm_api_key="YOUR_API_KEY",
        # llm_base_url="http://localhost:5080"
    )

    sys_prompt_zh = """
            你是一个智能助手，用户会向你提问。你现在可以调用以下工具来帮助用户完成任务。每个工具有不同的功能和参数，工具的定义如下：

    【工具列表】
    {tool_list}
    ...（此处可扩展更多工具，格式统一）

    【规则说明】
    - 如果用户的问题可以通过调用上述工具解决，请判断应该使用哪些工具，并从用户输入中抽取出各个参数的内容。
    - 你的回答**必须严格按照JSON结构输出**，不允许有任何自然语言描述，不要使用任何Markdown格式，不要添加```json或```标记或有任何的换行字符如("\n"、"\r"等)，直接输出JSON字符串：
       {{ 
        "use_tool_list":[{{
          "tool":"工具名",
          "arguments": {{
            "参数1名": "参数1值",
            "参数2名": "参数2值"
          }}
        }}]
        }}
    - 如果用户的问题无法用上述工具解决，不允许返回空的工具列表，仅输出最终答复即可
    - 重要：只返回一个没有换行的纯JSON字符串，不需要任何额外格式化或标记

    你有工具可以调用。每次只能选择一个工具/或直接输出最终结果，不得一直递归/死循环调用工具。如果已连续多次调用工具后依然无法满足用户需求，必须用你现有得到的所有工具结果生成最终答复。
            """

    sys_prompt = """
                You are an intelligent assistant. Users will ask you questions.
                
                
                Now you can call the following tools to help users complete tasks.Each tool has different functions and parameters, defined as follows:
                <Tool_List>
                    {tool_list}
                    ...(More tools can be added here, uniform format)
                </Tool_List>
                
                <Rule_Explanation>
                    - If the user's question can be solved by calling the above tools, determine which tools to use and extract the parameter contents from the user's input.
                    - Your response **must strictly output in JSON structure**, no natural language description allowed, no Markdown formatting, no ```json or ``` tags, and no newline characters like ("\n", "\r", etc.), output the JSON string directly:
                       {{
                        "use_tool_list":[{{
                          "tool":"tool_name",
                          "arguments": {{
                            "param1_name": "param1_value",
                            "param2_name": "param2_value"
                          }}
                        }}]
                        }}
                    - If the user's question cannot be solved by the above tools, do not return an empty tool list; output only the final response.
                    - Important: Only return a pure JSON string without any extra formatting or markers.
                </Rule_Explanation>
                
                You have tools to call. Choose one tool at a time / or directly output the final result, no recursive/dead loop calls to tools. If multiple consecutive calls to tools still fail to meet user needs, you must generate a final response using all existing tool results obtained.
                """


    tool_prompt = """
                    The tools was called:
                    <action_list>
                        {action_list} 
                    </action_list>
                    
                     the tool returned the result:
                     <tool_result>
                        {result} 
                    </tool_result>
                     
                     Please summarize it in natural language facing the user based on the original question.
                    """

    search_sys_prompt = "You are a helpful agent."
    search = Agent(
        conf=agent_config,
        name="search_agent",
        # todo:tool_promot
        system_prompt=sys_prompt,
        tool_prompt= tool_prompt,
        mcp_servers=["amap-amap-sse"],  # MCP server name for agent to use
        use_call_tool=False
        #mcp_servers = ["simple-calculator"]  # MCP server name for agent to use
    )

    # Define a task
    Task(
        #input="Hotels within 1 kilometer of West Lake in Hangzhou", agent=search, conf=TaskConfig()
        input="杭州西湖一公里以内的3星级酒店,列出10家即可，用英文回复", agent=search, conf=TaskConfig()
    ).run()
