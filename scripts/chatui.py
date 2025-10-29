import gradio as gr
from juristiq.ui.chat import ChatClient
from juristiq.inference.models import ModelName
from juristiq.inference.prompts import load_template
from juristiq.config.inference import JudgeInferenceParams
from juristiq.config.templates import AMAZON_NOVA_EVALUATION_SYSTEM_ROLE_PROMPT


_UNINITIALIZED_TEXT = "🔴 Not initialized"
_INITIALIZED_TEXT = "🟢 Initialized"
_juristiq_client = ChatClient(inference_params=JudgeInferenceParams())


def initialize_client(model_id: str, system_prompt: str):

    _juristiq_client.initialize(model_id=model_id, 
                                system_prompt=system_prompt)

    return _INITIALIZED_TEXT


def change_system_prompt(prompt: str):

    _juristiq_client.uninitialize()

    return prompt, _UNINITIALIZED_TEXT


def handle_query(query, chat_history):

    if not _juristiq_client.is_initialized():
        chat_history.append((query, "Initialize the Juristiq agent first."))
        yield chat_history
        return

    chat_history.append(("User", query))
    chat_history.append(("Legal Assistant", ""))

    for text in _juristiq_client.send_query(query):
        chat_history[-1] = ("Legal Assistant", chat_history[-1][1] + text)
        yield chat_history


def create_config_panel():

    gr.Markdown("### ⚙️ Configuration")
    model_id = gr.Dropdown(label="Bedrock model ID", 
                           choices=[ModelName.NOVA_LITE.value, ModelName.NOVA_PRO.value, ModelName.GPT_OSS_20B.value], 
                           value=ModelName.NOVA_LITE.value)
    system_prompt = gr.Textbox(label="System prompt",
                               lines=10,
                               max_lines=20,
                               value=load_template(AMAZON_NOVA_EVALUATION_SYSTEM_ROLE_PROMPT))
    init_btn = gr.Button("Initialize client", variant="primary")

    return model_id, system_prompt, init_btn
        

def create_chat_panel():

    gr.Markdown("### 💬 Chat with Legal Assistant")
    chatbot = gr.Chatbot(label="Chat History", height=450)
    query_input = gr.Textbox(placeholder="Ask about legal document...", label="Query Input")
    send_btn = gr.Button("🚀 Send Query", variant="primary")
    send_btn.click(
        handle_query,
        inputs=[query_input, chatbot],
        outputs=[chatbot],
    )

    return chatbot, query_input, send_btn


def create_dashboard_panel(system_prompt):

    gr.Markdown("### 📊 Dashboard")
    status_display = gr.Textbox(value=_UNINITIALIZED_TEXT, 
                                label="Client initialization status", 
                                interactive=False)
    total_queries = gr.Number(value=0, label="Total Queries", interactive=False)

    gr.Markdown("### Predefined system prompts")
    predefined_system_prompts = [
        ("Compliance review", load_template(AMAZON_NOVA_EVALUATION_SYSTEM_ROLE_PROMPT)),
        ("General questions", "You are a highly skilled Legal AI Assistant specializing in contract compliance review.")
    ]
    for prompt in predefined_system_prompts:
        btn = gr.Button(prompt[0])
        btn.click(fn=lambda x=prompt[1]: change_system_prompt(x), 
                    outputs=[system_prompt, status_display])
        
    init_btn.click(initialize_client,
                inputs=[model_id, system_prompt],
                outputs=[status_display])
    send_btn.click(None, None, total_queries, js="()=>{elem=document.querySelector('input[aria-label=\"Total Queries\"]');elem.value=parseInt(elem.value||0)+1}")


with gr.Blocks(theme=gr.themes.Base()) as ui:
  gr.Markdown("## ☁️ <span style='color:#f2a900;'>Legal Assistant</span>")
  with gr.Row():
    with gr.Column(scale=1, min_width=200):
        model_id, system_prompt, init_btn = create_config_panel()
        
    with gr.Column(scale=2, min_width=500):
        chatbot, query_input, send_btn = create_chat_panel()

    with gr.Column(scale=1, min_width=200):
        create_dashboard_panel(system_prompt)

   
gr.Markdown("---\n<p style='text-align:center;color:gray'>Legal Assistant © 2025</p>")


if __name__ == "__main__":
    ui.launch()
