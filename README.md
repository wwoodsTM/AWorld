<p align="center">
  <img src="readme_assets/aworld_logo.png" alt="AWorld Logo" width="100"/>
  <br>
  <span align="center" style="font-size: 24px;">
    <b><span style="color: #1677FF;">A</span><span style="color: var(--color-fg-default, #333333);">World</span></b>
  </span>
</p>

> **Build, evaluate and run General Multi-Agent Assistance with ease**

Through AWorld (short for Agent World), you can quickly build real-world scenarios or task automation into agentic prototypes, then extend them into a generic agent or a team of agents to assist your real needs, like Manus.

Hope AWorld would bridge the gap between theoretical MAS (Multi-Agent System) capabilities and practical implementation in real-world applications and guide you into the AGI World. *GLHF!*

![AWorld Framework](readme_assets/framework_arch.png)

## Installation
With pip (Python>=3.11):
```bash
python setup.py install
```

## Environment Configuration
Configure the following environment variables according to your selected AI models:
```plaintext
# AI Model API Keys
CLAUDE_API_KEY=sk-xxxx        # Anthropic Claude
DEEP_SEEK_API_KEY=sk-xxxx     # DeepSeek AI
MISTRAL_API_KEY=sk-xxxx       # Mistral AI
OPENAI_API_KEY=sk-xxxx        # OpenAI
GOOGLE_API_KEY=sk-xxxx        # Google AI
AZURE_OPENAI_API_KEY=sk-xxxx  # Azure OpenAI
QWEN_API_KEY=sk-xxxx          # Alibaba Qwen
MOONSHOT_API_KEY=sk-xxxx      # Moonshot AI
```

## Usage
### Running Pre-defined Agents ([demo code]())
Below are two demos showcasing how a single agent can use both a browser and a phone, respectively.

<table>
  <tr>
    <td width="50%">
      <a href="https://www.youtube.com/watch?v=R7keOLrRDoM" target="_blank">
        <img src="https://img.youtube.com/vi/R7keOLrRDoM/0.jpg" alt="AWorld Browser Demo" width="95%">
      </a>
    </td>
    <td width="50%">
      <a href="https://www.youtube.com/watch?v=TYh3iqDeIoQ" target="_blank">
        <img src="https://img.youtube.com/vi/TYh3iqDeIoQ/0.jpg" alt="AWorld Mobile Demo" width="95%">
      </a>
    </td>
  </tr>
</table>

### or Creating Your Own Agents ([Quick Start Tutorial](./aworld/agents/README.md))
Here is a multi-agent example of running a level2 task from the [GAIA](https://huggingface.co/gaia-benchmark) benchmark:

```python
from aworld.agents.gaia.agent import PlanAgent, ExecuteAgent
from aworld.core.client import Client
from aworld.core.swarm import Swarm
from aworld.core.task import GeneralTask
from aworld.config.conf import AgentConfig, TaskConfig
from aworld.dataset.mock import mock_dataset

# Initialize client
client = Client()

# One sample for example
test_sample = mock_dataset("gaia")

# Create agents
agent_config = AgentConfig(
    llm_provider="openai",
    llm_model_name="gpt-4o",
)
agent1 = PlanAgent(conf=agent_config)
agent2 = ExecuteAgent(conf=agent_config)

# Create swarm for multi-agents
# define (head_node, tail_node) edge in the topology graph
# NOTE: the correct order is necessary
swarm = Swarm((agent1, agent2))

# Define a task
task = GeneralTask(input=test_sample, swarm=swarm, conf=TaskConfig())

# Run task
result = client.submit(task=[task])

print(f"Task completed: {result['success']}")
print(f"Time cost: {result['time_cost']}")
print(f"Task Answer: {result['task_0']['answer']}")
```
```
Task completed: True
Time cost: 26.431413888931274
Task Answer: Time-Parking 2: Parallel Universe
```
<p align="left">
  <video src="https://github.com/user-attachments/assets/84ccf675-08df-47c1-bd0e-416480ad7cff" controls="controls" muted="muted" style="width: 45%;"></video>
</p>

## Framework Architecture

AWorld uses a client-server architecture with three main components:

1. **Client-Server Architecture**: Similar to [ray](https://github.com/ray-project/ray), this architecture:
    - Decouples agents and environments for better scalability and flexibility
    - Provides a unified interaction protocol for all agent-environment interactions

2. **Agent/Actor**: 
   - Encapsulates system prompts, tools, and models with the capability to hand off execution to other agents
   - Agent fields and properties:

    | Field        | Type      | Description                                                           |
    |--------------|-----------|-----------------------------------------------------------------------|
    | `id`         | string    | Unique identifier for the agent                                       |
    | `name`       | string    | Name of the agent                                                     |
    | `model_name` | string    | LLM model name of the agent                                           |
    | `_llm`       | object    | LLM model instance based on model_name (e.g., "gpt-4", "claude-3")    |
    | `conf`       | BaseModel | Configuration inheriting from pydantic BaseModel                      |
    | `dict_conf`  | dict      | Dictionary-structured configuration for safe key access               |
    | `memory`     | object    | Storage for maintaining context across interactions                   |
    | `tool_names` | list      | List of tools the agent can use                                       |
    | `handoffs`   | list      | List of other agents this agent can delegate tasks to                 |
    | `finished`   | bool      | Flag indicating whether the agent has completed its task              |

3. **Environment/World Model**: Various tools and models in the environment
   - Computer interfaces (browser, shell, functions, etc.)
   - World Model

   | Tools | Description |
   |-------|-------------|
   | `browser` | Controls web browsers for navigation, form filling, and interaction with web pages |
   | `android` | Manages Android device simulation for mobile app testing and automation |
   | `shell` | Executes shell commands for file operations and system interactions |
   | `code` | Runs code snippets in various languages for data processing and automation |
   | `search` | Performs web searches and returns structured results for information gathering and summary |
   | `document` | Handles file operations including reading, writing, and managing directories |


## Dual Purpose Framework

AWorld serves two complementary purposes:

### Agent Evaluation
Standardized benchmarking of agent capabilities under a unified protocol:
- Unified task definitions to run both customized and public benchmarks
- Efficient and stable execution environment
- Detailed test reports measuring efficiency (steps to completion), completion rates, and token costs

### Model Training
Continuous improvement through a collaborative competition cycle:
- Agent models improve to overcome challenges 
- World models (environments) evolve to present new, more complex scenarios

## Key Features

- 🌐 **Environment Multi-Tool Support**: 
  - [x] Browsers (Chrome, Firefox)
  - [x] Android device simulation
  - [x] Shell, code (Python), and apis (e.g., google_search)
  - [x] File system (writing, managing on going)
  - [ ] Cloud sandbox for quick and stable deployment

- 🤖 **AI-Powered Agents**:
  - [x] Agent initialization
  - [x] Delegation between multiple agents
  - [ ] Asynchronous delegation
  - [ ] Human delegation (e.g., for password entry)
  - [ ] Pre-deployed open source LLMs powered by state-of-the-art [inference frameworks](https://github.com/alipay/PainlessInferenceAcceleration)


- 🔄 **Standardized Protocol**:
  - [ ] Client-server protocol compatible with Model Contest Protocol (MCP)
  - [x] Environment interfaces following [gymnasium](https://gymnasium.farama.org/api/env/#gymnasium.Env.step) standards
  - [x] Custom agent-environment protocol

- 🎛️ **Web Interface**:
  - [x] Chat interface for diverse user queries
  - [x] Server configuration dashboard
  - [ ] Real-time monitoring tools
  - [ ] Performance reporting

- 🧠 **Benchmarks and Samples**:
  - [ ] Support standardized benchmarks by default, e.g., GAIA, WebArena
  - [ ] Support customized benchmarks
  - [ ] Support generating training samples

## Contributing

If you use AWorld in your research or wish to contact us, please use the following BibTeX entry:

```bibtex
@software{aworld2025,
  author = {Agent Team at Ant Group},
  title = {AWorld: A Unified Agent Playground for Computer and Phone Use Tasks},
  year = {2025},
  url = {https://github.com/inclusionAI/AWorld},
  version = {0.1.0},
  publisher = {GitHub},
  email = {chenyi.zcy at antgroup.com}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
