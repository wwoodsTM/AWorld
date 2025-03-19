<p align="center">
  <img src="readme_assets/aworld_logo.png" alt="AWorld Logo" width="100"/>
  <br>
  <span align="center" style="font-size: 24px;">
    <b><span style="color: #1677FF;">A</span><span style="color: var(--color-fg-default, #333333);">World</span></b>
  </span>
</p>

> **Build, evaluate and run General Multi-Agent Assistance with ease**

Through AWorld (short for Agent World), you can quickly build real-world scenarios or task automation into agentic prototypes, then extend them into a generic agent or a team of agents to assist your real needs, like Manus.

*Hope AWorld would guide you into the AGI World. GLHF!*


![AWorld Framework](readme_assets/framework_arch.png)

## Installation
With pip (Python>=3.11):
```bash
pip install aworld
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
Easily configure and run a pre-defined agent through our web interface:
```bash
aworld-web start
```
Next, input a user query for a quick test. Below are two demos showcasing how a single agent can use both a browser and a phone, respectively.

<p align="left">
  <video src="https://github.com/user-attachments/assets/01ea37e8-6544-4632-b2c3-29a7e356dba8" controls="controls" muted="muted" style="width: 45%;"></video>
</p>

### or Creating Your Own Agents
Here is an example of running a level2 task from the [GAIA](https://huggingface.co/gaia-benchmark) benchmark:

```python
from core.client import Client
from task.gaia.agent import PlanAgent, ExcuteAgent
from task.gaia.gaia_task import GeneralTask
from task.gaia.swarm import Swarm
from config.conf import AgentConfig, TaskConfig
from task.gaia.tools import mock_dataset

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
agent2 = ExcuteAgent(conf=agent_config)

# Create swarm for multi-agents
# define (head_node, tail_node) edge in the topology graph
swarm = Swarm((agent1, agent2))

# Define a task
task = GeneralTask(input=test_sample, swarm=swarm, conf=TaskConfig())

# Run task
result = client.submit(task=[task])

print(f"Task completed: {result['success']}")
print(f"Time cost: {result['time_cost']}")
print(f"Task Answer: {result['task_0']['answer']}")
```

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
