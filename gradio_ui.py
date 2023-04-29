import torch
import numpy as np
import gradio as gr
from gpt import GPT
from transformers import GPT2Tokenizer

class GPT2Generator:
    def __init__(self, model, tokenizer, device = "cuda"):
        self.device = device
        self.model = model
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = tokenizer
        
    @torch.no_grad()
    def generate_text(self, text, max_length=50, temperature = 1.0): # probabilistic prediction
        tokenized = self.tokenizer('<bos>' + text, return_tensors="pt")
        inputs = tokenized["input_ids"].to(self.device)
        
        for _ in range(max_length):
            inputs = torch.asarray(inputs).reshape(1, -1)
            outputs, _ = self.model(inputs.to(self.device))

            outputs = outputs[:, -1] / temperature
            outputs = torch.softmax(outputs, dim=-1)

            probs = outputs.squeeze().detach().cpu().numpy()

            next_token = np.random.choice(len(probs), p=probs)

            inputs = inputs.tolist()[0]
            inputs.append(next_token)

            if next_token == self.tokenizer.encode(self.tokenizer.eos_token)[0] or len(inputs) >= max_length:
                break
        
        generated_text = self.tokenizer.decode(inputs, skip_special_tokens=True)
        print(f"Generated text: {generated_text}")
        return generated_text



tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
SPECIAL_TOKENS = {'bos_token':'<bos>','eos_token' :'<eos>', 'pad_token':'<pad>', 'sep_token': '<sep>'}
tokenizer.add_special_tokens(SPECIAL_TOKENS)

model = GPT(
    vocab_size=len(tokenizer), #tokenizer.vocab_size,
    n_layers = 8,
    n_heads = 8,
    d_model= 512,
    resid_dropout=0.1,
    attn_dropout=0.1,
    d_ff = 2048,
    ff_dropout=0.1,
    embed_dropout=0.1,
    max_len=64,
    pad_idx = tokenizer.encode(tokenizer.pad_token)[0]
)

try:
    model.load_state_dict(torch.load("saved_models/gpt_model.pt"))
except:
    print("Can't load pre-trained model. Exiting...")
    exit()



    
generator = GPT2Generator(model, tokenizer)
def generate_text(text: str, model_output_len: int, temperature: float = 1.0):
    return generator.generate_text(text, model_output_len, temperature)
inputs = [
    gr.Textbox(label="Input"),
    gr.Slider(minimum=10, maximum=64, value=32, step=1, label="Output length"),
    gr.Slider(minimum=0.0, maximum=1.0, value=1.0, label="Temperature"),
]
outputs = gr.Textbox(label="Generated text:")
title = "GPT Text Generation"
description = "Generate text using GPT model"
gr.Interface(
    generate_text,
    inputs,
    outputs,
    title=title,
    description=description
).launch(share=True)