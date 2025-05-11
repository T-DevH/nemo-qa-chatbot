"""Chat component for NeMo QA Chatbot."""

import gradio as gr
from typing import List, Dict, Any, Callable

def create_chat_component(chat_func: Callable):
    """Create chat component.
    
    Args:
        chat_func: Chat function
    
    Returns:
        Chat component
    """
    with gr.Column() as chat_component:
        chatbot = gr.Chatbot(height=600)
        with gr.Row():
            question = gr.Textbox(
                label="Question",
                placeholder="Type your question here...",
                show_label=False
            )
            submit_btn = gr.Button("Send")
    
    return chat_component, chatbot, question, submit_btn

# This file is intended for chat components.
# Add your chat code here.
