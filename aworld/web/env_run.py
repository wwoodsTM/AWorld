# coding: utf-8
# Copyright (c) 2025 inclusionAI.

import argparse
import os
import gradio as gr

from gradio.themes import Default, Base

from aworld.agents import BrowserAgent, AndroidAgent
from aworld.config import AgentConfig
from aworld.models.llm import model_names
from aworld.task.base import GeneralTask
from aworld.virtual_environments import BrowserTool, AndroidTool
from aworld.virtual_environments.conf import BrowserToolConfig, AndroidToolConfig

# define CSS
custom_css = """
    .gradio-container { 
        max-width: 1000px; 
        margin: auto; 
        background-color: #f9f9f9; 
        padding: 28px; 
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .gradio-markdown h1 {
        font-size: 28px;
        color: #333;
        text-align: center;
        margin-bottom: 20px;
    }
    .title-markdown {
        color: gray;
        font-weight: normal;
    }
    .gradio-textbox {
        width: 1500px !important; 
    }
    #chat-textbox {
        width: 600px !important;
    }
    .gradio-textbox textarea {
        background-color: #fff;
        border: 1px solid #ddd;
        border-radius: 5px;
        font-size: 16px;
        padding: 10px;
    }
    .gradio-chatbot {
        background-color: #fff; 
        border: 1px solid #ddd; 
        border-radius: 10px;
        padding: 15px; 
        max-height: 400px; 
        overflow-y: auto;
    }
    .gradio-chatbot .user {
        background-color: #007bff; 
        color: white;
        border-radius: 10px 10px 0 10px; 
        padding: 10px; 
        margin: 5px 0; 
        max-width: 70%;
        align-self: flex-end;
    }
    .gradio-chatbot .bot {
        background-color: #f1f1f1;
        color: #333; 
        border-radius: 10px 10px 10px 0;
        padding: 10px; 
        margin: 5px 0; 
        max-width: 70%; 
        align-self: flex-start;
    }
    .gradio-examples {
        background-color: #f1f1f1; 
        border: 1px solid #ddd; 
        border-radius: 5px;
        padding: 10px; 
        margin-top: 10px;
    }
    .gradio-examples .example {
        background-color: #fff; 
        border: 1px solid #ddd; 
        border-radius: 5px; 
        padding: 5px 10px;
        margin: 5px 0; 
        cursor: pointer;
    }
    .gradio-examples .example:hover {
        background-color: #f9f9f9;
    }
    footer {
        display: none !important;
    }
"""

theme_map = {
    "Default": Default(),
    "Base": Base()
}


def default_config():
    """Prepare the default configuration"""
    return {
        "tool": "browserTool",
        "max_steps": 100,
        "max_actions_per_step": 10,
        "use_vision": True,
        "tool_calling_method": "auto",
        "llm_provider": "**",
        "llm_model_name": "**",
        "llm_num_ctx": 32000,
        "llm_temperature": 1.0,
        "llm_base_url": "",
        "llm_api_key": "",
        "use_own_browser": os.getenv("CHROME_PERSISTENT_SESSION", "false").lower() == "true",
        "keep_browser_open": False,
        "headless": False,
        "disable_security": True,
        "enable_recording": True,
        "window_w": 1280,
        "window_h": 1100,
        "save_recording_path": "./tmp/record_videos",
        "save_trace_path": "./tmp/traces",
        "save_agent_history_path": "./tmp/agent_history",
        "query": "go to google.com and type 'OpenAI' click search and give me the first url",
        "config_data": "",
        "avd_name": "",
        "adb_path": "",
        "emulator_path": "",
    }


def update_model_dropdown(llm_provider, api_key=None, base_url=None):
    """
    Update the model name dropdown with predefined models for the selected provider.
    """
    # Use API keys from .env if not provided
    if not api_key:
        api_key = os.getenv(f"{llm_provider.upper()}_API_KEY", "")
    if not base_url:
        base_url = os.getenv(f"{llm_provider.upper()}_BASE_URL", "")

    # Use predefined models for the selected provider
    if llm_provider in model_names:
        return gr.Dropdown(choices=model_names[llm_provider], value=model_names[llm_provider][0], interactive=True)
    else:
        return gr.Dropdown(choices=[], value="", interactive=True, allow_custom_value=True)


# Chat


def chat_with_bot(query, **kwargs):
    agent_config = AgentConfig(**kwargs)
    if(kwargs.get("tool") == "browserTool"):
        agent = BrowserAgent(agent_config)
        tool_config = BrowserToolConfig(**kwargs)
        tool = BrowserTool(tool_config)
    else:
        agent = AndroidAgent(agent_config)
        tool_config = AndroidToolConfig(**kwargs)
        tool = AndroidTool(tool_config)

    response = GeneralTask(input=query, agent=agent,tools=[tool]).run()

    return response


def create_ui(config, theme_name="Base"):
    with gr.Blocks(
            title="GUI Navigation", theme=theme_map[theme_name], css=custom_css
    ) as ui:
        with gr.Row():
            gr.Markdown(
                """
                # 🌐 GUI Navigation
                ### Control your web/app with AI assistance
                """,
                elem_classes=["header-text"],
            )

        with gr.Tabs() as tabs:
            with gr.TabItem("Env Settings", id=1):
                with gr.Group():
                    gr.Markdown("Tools Setting")
                    tool = gr.Radio(
                        ["browserTool", "appTool"],
                        label="Tools",
                        value=config['tool'],
                        info="Choose the type of tool you want to utilize for the task",
                    )
                    with gr.Column():
                        max_steps = gr.Slider(
                            minimum=1,
                            maximum=200,
                            value=config['max_steps'],
                            step=1,
                            label="Max Run Steps",
                            info="The maximum number of steps that the task is allowed to execute",
                        )
                        max_actions_per_step = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=config['max_actions_per_step'],
                            step=1,
                            label="Max Actions per Step",
                            info="The maximum number of actions that the task can perform in each step",
                        )
                with gr.Group():
                    gr.Markdown("Browser Setting")
                    with gr.Row():
                        use_own_browser = gr.Checkbox(
                            label="Use Own Browser",
                            value=config['use_own_browser'],
                            info="use local browser",
                        )
                        keep_browser_open = gr.Checkbox(
                            label="Keep Browser Open",
                            value=config['keep_browser_open'],
                            info="Keep Browser remain Open",
                        )
                        headless = gr.Checkbox(
                            label="Headless Mode",
                            value=config['headless'],
                            info="run in headless mode",
                        )
                        disable_security = gr.Checkbox(
                            label="Disable Security",
                            value=config['disable_security'],
                            info="Disable browser security",
                        )
                        enable_recording = gr.Checkbox(
                            label="Enable Recording",
                            value=config['enable_recording'],
                            info="Enable saving recordings",
                        )

                    with gr.Row():
                        window_w = gr.Number(
                            label="Window Width",
                            value=config['window_w'],
                            info="Browser window width",
                        )
                        window_h = gr.Number(
                            label="Window Height",
                            value=config['window_h'],
                            info="Browser window height",
                        )

                    save_recording_path = gr.Textbox(
                        label="Recording Path",
                        placeholder="e.g. ./tmp/record_videos",
                        value=config['save_recording_path'],
                        info="Path to save browser recordings",
                        interactive=True,  # Allow editing only if recording is enabled
                    )

                    save_trace_path = gr.Textbox(
                        label="Trace Path",
                        placeholder="e.g. ./tmp/traces",
                        value=config['save_trace_path'],
                        info="Path to save Agent traces",
                        interactive=True,
                    )

                    save_agent_history_path = gr.Textbox(
                        label="Agent History Save Path",
                        placeholder="e.g., ./tmp/agent_history",
                        value=config['save_agent_history_path'],
                        info="Specify the directory where agent history should be saved.",
                        interactive=True,
                    )
                with gr.Group():
                    gr.Markdown("Android Setting")
                    avd_name = gr.Textbox(
                        label="avd name",
                        value=config['avd_name'],
                        info="Android virtual device name",
                        interactive=True,  # Allow editing only if recording is enabled
                    )
                    adb_path = gr.Textbox(
                        label="adb path",
                        value=config['adb_path'],
                        info="Path of the Android Debug Bridge (ADB)",
                        interactive=True,  # Allow editing only if recording is enabled
                    )
                    emulator_path = gr.Textbox(
                        label="emulator path ",
                        value=config['emulator_path'],
                        info="Path of the Android Emulator",
                        interactive=True,  # Allow editing only if recording is enabled
                    )

            with gr.TabItem("Agent Configuration", id=2):
                with gr.Group():
                    llm_provider = gr.Dropdown(
                        choices=[provider for provider, model in model_names.items()],
                        label="LLM Provider",
                        value=config['llm_provider'],
                        info="Pick the provider of the large language model you'll be using"
                    )
                    llm_model_name = gr.Dropdown(
                        label="Model Name",
                        choices=model_names['openai'],
                        value=config['llm_model_name'],
                        interactive=True,
                        allow_custom_value=True,  # Allow users to input custom model names
                        info="Select the specific name of the large language model"
                    )
                    llm_num_ctx = gr.Slider(
                        minimum=2 ** 8,
                        maximum=2 ** 16,
                        value=config['llm_num_ctx'],
                        step=1,
                        label="Max Context Length",
                        info="Determines the maximum amount of context the model can handle",
                        visible=config['llm_provider'] == "ollama"
                    )
                    llm_temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=config['llm_temperature'],
                        step=0.1,
                        label="Temperature",
                        info="Adjusts the randomness of the model's output"
                    )
                    with gr.Row():
                        llm_base_url = gr.Textbox(
                            label="Base URL",
                            value=config['llm_base_url'],
                            info="The URL of the API endpoint that the large language model uses for communication"
                        )
                        llm_api_key = gr.Textbox(
                            label="API Key",
                            type="password",
                            value=config['llm_api_key'],
                            info="Your unique API key required for accessing the large language model's services"
                        )
                    with gr.Row():
                        config_data = gr.TextArea(
                            label="Other Config by JSON",
                            placeholder="",
                            value=config['config_data'],
                            info="Enter JSON formatted text",
                            interactive=True,  # Allow editing only if recording is enabled
                        )

            def update_llm_num_ctx_visibility(llm_provider):
                return gr.update(visible=llm_provider == "ollama")

            # Bind the change event of llm_provider to update the visibility of context length slider
            llm_provider.change(
                fn=update_llm_num_ctx_visibility,
                inputs=llm_provider,
                outputs=llm_num_ctx
            )
            with gr.TabItem("Chat", id=3):
                # chatbot = gr.Chatbot(label="Chat history", bubble_full_width=False)
                state = gr.State()
                with gr.Row(elem_classes="input-row"):
                    with gr.Column(scale=9, min_width=0, elem_classes="input-col"):
                        query = gr.Textbox(
                            label="Input task",
                            placeholder="Please enter your task and start.",
                            lines=4,
                            max_lines=5,
                            elem_id="chat-textbox",
                            value="go to google.com and type '支付宝' click search and click the first search result"
                        )
                    with gr.Column(scale=1, min_width=0, elem_classes="button-col"):
                        submit_button = gr.Button("start", variant="primary", elem_classes="full-height-button")

                submit_button.click(
                    chat_with_bot,
                    # inputs=[query, state],
                    inputs=[query, state, tool, llm_provider, llm_model_name, llm_num_ctx, llm_temperature,
                            llm_base_url,
                            llm_api_key,
                            use_own_browser, keep_browser_open, headless, disable_security, window_w, window_h,
                            save_recording_path, save_agent_history_path, save_trace_path,  # Include the new path
                            enable_recording, max_steps, max_actions_per_step, avd_name, adb_path, emulator_path],
                    # outputs=[chatbot, state]
                    outputs=[state]
                )

        # Attach the callback to the LLM provider dropdown
        llm_provider.change(
            lambda provider, api_key, base_url: update_model_dropdown(provider, api_key, base_url),
            inputs=[llm_provider, llm_api_key, llm_base_url],
            outputs=llm_model_name
        )

        # Add this after defining the components
        enable_recording.change(
            lambda enabled: gr.update(interactive=enabled),
            inputs=enable_recording,
            outputs=save_recording_path
        )

    return ui


def init():
    parser = argparse.ArgumentParser(description="Gradio UI for Browser Agent")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind to")
    parser.add_argument("--port", type=int, default=6688, help="Port to listen on")
    parser.add_argument("--theme", type=str, default="Base", choices=theme_map.keys(), help="Theme to use for the UI")
    parser.add_argument("--dark-mode", action="store_true", help="Enable dark mode")
    args = parser.parse_args()

    config_dict = default_config()

    ui = create_ui(config_dict, theme_name=args.theme)
    ui.launch(server_name=args.ip, server_port=args.port)


if __name__ == '__main__':
    init()
