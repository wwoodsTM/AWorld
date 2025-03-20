# App tasks

- Simple task

```python
from aworld.core.task import Task
from aworld.apps.gym.run import run_gym_game


class GymTask(Task):
    def __init__(self, conf):
        super(GymTask, self).__init__(conf)

    def run(self):
        run_gym_game(self.conf.get("env_tool_id"),
                     render_mode=self.conf.get("render_mode", "human"))


GymTask({"env_tool_id": 'CartPole-v0', "render_mode": 'human'}).run()

```

- Demo task, pseudo code for specifying agents and tools in the environment.

```python
from aworld.core.env_tool import ToolFactory
from aworld.agents.base import AgentFactory
from aworld.config.conf import AgentConfig, ToolConfig
from aworld.core.task import Task


class DemoTask(Task):
    def __init__(self, conf):
        super(DemoTask, self).__init__(conf)

    def run(self):
        _tool_name, _agent_name = self._env_tool_with_agent()
        agent = AgentFactory(_agent_name, AgentConfig())
        env = ToolFactory(_tool_name, ToolConfig())
        observation, info = env.reset()
        while True:
            action = agent.policy(observation)
            observation, _, _, _, info = env.step(action)
            if info.finished:
                break

    def _env_tool_with_agent(self):
        _tool = self.conf.get("tool_name")
        _agent = self.conf.get("agent_name")
        return _tool, _agent
```
