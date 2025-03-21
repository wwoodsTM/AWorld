# Apps

Use cases based on AWorld's framework implementation.

How to use the framework to develop a simple applications:
```python
from aworld.core.client import Client
from aworld.core.task import GeneralTask

# create client instance
client = Client()
# create agent based on `aworld.core.agent.base.BaseAgent` by yourself
agent = YourAgent()
# create tools with its actions follow existing tools in the virtual_environments package in the framework, 
# or use tools already in framework
tool = YourTool()
# create a task, which with an agent or swarm and the tools needed. 
# If params `tools` value is None, means all available tools can be used by default
task = GeneralTask(agent=agent, tools=[tool])

# use client
res = client.submit(task, parallel=False)
```