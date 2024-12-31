import gradio as gr
from huggingface_hub import hf_hub_download
import pickle
import gradio as gr
import numpy as np
import subprocess
import shutil
# Define the function to process the input file and model selection
def process_file(file, model_name):
    with open(file.name, 'r') as f:
        content = f.read()
    saved_test_dataset = "test.txt"
    saved_test_label = "saved_test_label.txt"
    
    # Save the uploaded file content to a specified location
    shutil.copyfile(file.name, saved_test_dataset)
    # For demonstration purposes, we'll just return the content with the selected model name
    subprocess.run(["python", "src/test_saved_model.py"])
    return f"Model: {model_name}\nContent:\n{content}"

# List of models for the dropdown menu
models = ["Model A", "Model B", "Model C"]

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# File Processor with Model Selection")
    gr.Markdown("Upload a .txt file and select a model from the dropdown menu.")
    
    with gr.Row():
        file_input = gr.File(label="Upload a .txt file", file_types=['.txt'])
        model_dropdown = gr.Dropdown(choices=models, label="Select a model")
    
    output_text = gr.Textbox(label="Output")

    btn = gr.Button("Submit")
    btn.click(fn=process_file, inputs=[file_input, model_dropdown], outputs=output_text)

# Launch the app
demo.launch()
