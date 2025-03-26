# Core Components

Common functionality and system components.

- `agent/`: Base agent for sub agents and description of already registered agents.
- `envs/`: The environment and its tools, as well as the related actions of the tools. It is a three-level and one to
  many structure.
- `swarm`: Interactive collaboration between multiple agents that interact with the environment tools. It can be
  considered as a multi-agent workflow.
- `task`: Complete runnable specific work that includes data, agents, and environment.
- `client`: Submit various tasks to execute and obtain results.

![Architecture](../../readme_assets/framework_arch.png)