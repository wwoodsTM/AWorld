import os

from openai import OpenAI

from aworld.config.conf import AgentConfig
from aworld.logs.util import logger
from aworld.models.llm import call_llm_model, get_llm_model


def openai_test(base_url, api_key):
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    completion = client.chat.completions.create(
        model="anthropic/claude-3.7-sonnet",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                        },
                    },
                ],
            }
        ],
    )
    logger.success(f"openai driver: {completion.choices[0].message.content}")


def aworld_test(base_url, api_key):
    config = AgentConfig(
        llm_provider="openai",
        llm_model_name="anthropic/claude-3.7-sonnet",
        llm_base_url=base_url,
        llm_api_key=api_key,
        llm_temperature=0.15,
    )

    model = get_llm_model(config)
    logger.info(model)

    completion = call_llm_model(
        llm_model=model, messages=[{"role": "user", "content": "hello"}]
    )
    logger.success(f"aworld driver: {completion.content}")


if __name__ == "__main__":
    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY")
    logger.info(f"\n>>> base_url: {base_url}" f"\n>>> api_key: {api_key}")

    openai_test(base_url, api_key)
    aworld_test(base_url, api_key)
