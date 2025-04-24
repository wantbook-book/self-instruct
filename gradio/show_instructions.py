import gradio as gr
import json
from functools import partial
instructions = None
idx = 0
def load_instructions(file_path: str):
    global instructions, idx
    idx = 0
    instructions = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            instructions.append(item['instruction'])
    
    return 0, instructions[0]
    

def update_instruction(inc: int):
    global idx, instructions
    idx += inc
    if idx < 0:
        idx = 0
    if idx >= len(instructions):
        idx = len(instructions) - 1
    return idx, instructions[idx]


with gr.Blocks(title="Show Instructions") as demo:
    with gr.Row():
        file_path = gr.Textbox(label="File Path", placeholder="Enter file path here")
        load_btn = gr.Button("Load Instructions")
    idx = gr.Textbox(label="Index", )
    instruction = gr.Textbox(label="Instruction", placeholder="Instruction will be shown here")
    load_btn.click(load_instructions, inputs=[file_path], outputs=[idx, instruction])

    with gr.Row():
        next_btn = gr.Button("Next")
        prev_btn = gr.Button("Previous")

    next_btn.click(partial(update_instruction, 1), inputs=[], outputs=[idx, instruction])
    prev_btn.click(partial(update_instruction, -1), inputs=[], outputs=[idx, instruction])
demo.launch()
