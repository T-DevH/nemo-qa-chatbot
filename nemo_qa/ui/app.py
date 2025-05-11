"""Gradio application for NeMo QA Chatbot."""

import logging
import requests
import gradio as gr
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def create_ui(api_url: str):
    """Create Gradio UI.
    
    Args:
        api_url: URL of the API server
    
    Returns:
        Gradio interface
    """
    # Create Gradio blocks
    with gr.Blocks(title="NeMo QA Chatbot") as demo:
        gr.Markdown("# NeMo QA Chatbot")
        gr.Markdown("Ask questions and get answers from the LLAMA3 8B model fine-tuned with LoRA.")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=600)
                with gr.Row():
                    question = gr.Textbox(
                        label="Question",
                        placeholder="Type your question here...",
                        show_label=False
                    )
                    submit_btn = gr.Button("Send")
            
            with gr.Column(scale=2):
                with gr.Tab("Explainability"):
                    with gr.Row():
                        confidence = gr.Label(label="Confidence")
                    
                    with gr.Row():
                        attention_heatmap = gr.Heatmap(
                            label="Attention Visualization",
                            height=400
                        )
                
                with gr.Tab("Context"):
                    context = gr.Textbox(
                        label="Context",
                        placeholder="Add optional context here...",
                        lines=10
                    )
                    
                    clear_context_btn = gr.Button("Clear Context")
        
        gr.Markdown("## Examples")
        examples = gr.Examples(
            examples=[
                ["What is LoRA fine-tuning?"],
                ["Explain the advantages of LLAMA3 over LLAMA2."],
                ["How does NeMo Curator improve data quality?"]
            ],
            inputs=[question]
        )
        
        # Define chat function
        def chat_func(message, history, context_text):
            history_list = [[h[0], h[1]] for h in history]
            
            # Call API
            response = requests.post(
                f"{api_url}/api/chat",
                json={
                    "question": message,
                    "context": context_text,
                    "history": history_list
                }
            ).json()
            
            # Extract explainability data
            explainability = response.get("explainability", {})
            attention_scores = explainability.get("attention_scores", [])