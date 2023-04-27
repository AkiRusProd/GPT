import os
# I like to move downloaded datasets to another location
os.environ['HF_DATASETS_CACHE'] = 'D:\\Code\\Huggingface_cache\\'
os.environ['HUGGINGFACE_HUB_CACHE'] = 'D:\\Code\\Huggingface_cache\\'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import RAdam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from transformers import GPT2Tokenizer, GPT2TokenizerFast
from gpt import GPT
from torch.utils.data import DataLoader

from transformers import DataCollatorWithPadding

# torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
SPECIAL_TOKENS = {'bos_token':'<bos>','eos_token' :'<eos>', 'pad_token':'<pad>', 'sep_token': '<sep>'}
tokenizer.add_special_tokens(SPECIAL_TOKENS)


from data_loader import MyDataset
dataset = torch.load("dataset.pt")



data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=data_collator,
)


class GPTInterface:
    def __init__(self, model, tokenizer, optimizer = None, scheduler = None, loss_fn = None, dataloader = None):
        self.model = model
        self.model.to("cuda")
        
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.dataloader = dataloader

    def fit(self, epochs = 30, save_path = None):
        self.model.train()
        
        losses = []
        for epoch in range(epochs):
            tqdm_range = tqdm(enumerate(self.dataloader), total=len(self.dataloader))

            for i, batch in tqdm_range:
                self.optimizer.zero_grad()

                inputs = batch["input_ids"].to("cuda")
               
                targets = inputs[:, 1:].contiguous()
                   
                logits, _ = self.model(inputs)
                logits = logits[:, :-1].contiguous()
                     
                loss = self.loss_fn(logits.view(-1, len(tokenizer)), targets.view(-1))

                losses.append(loss.item())
                loss.backward()
                self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()

                tqdm_range.set_description(
                                f"training | loss: {loss.item():.7f} | epoch {epoch + 1}/{epochs}"
                        )

        if save_path is not None:
            torch.save(self.model.state_dict(), save_path)

    # def predict(self, text, max_length = 50): # deterministic prediction
    #     self.model.eval()

    #     tokenized = self.tokenizer('<bos>' + text, return_tensors="pt")
    #     inputs = tokenized["input_ids"].to("cuda")

    #     for i in range(max_length):
    #         inputs = torch.asarray(inputs).reshape(1, -1)
            
    #         outputs, _ = self.model(inputs.to("cuda"))
            
    #         idx = torch.argmax(outputs, dim=-1)[:, -1].item()
    #         # idx = torch.argmax(nn.Softmax(dim = -1)(outputs), dim=-1)[:, -1].item()

    #         inputs = inputs.tolist()[0]
    #         inputs.append(idx)

    #         if idx == tokenizer.encode(tokenizer.eos_token)[0] or len(inputs) >= max_length:
    #             break

    #     return self.tokenizer.decode(inputs, skip_special_tokens=True)

    def predict(self, text, max_length=50, temperature=0.3): # probabilistic prediction
        self.model.eval()

        tokenized = self.tokenizer('<bos>' + text, return_tensors="pt")
        inputs = tokenized["input_ids"].to("cuda")

        for i in range(max_length):
            inputs = torch.asarray(inputs).reshape(1, -1)
            outputs, _ = self.model(inputs.to("cuda"))

            outputs = outputs[:, -1] / temperature
            outputs = torch.softmax(outputs, dim=-1)

            probs = outputs.squeeze().detach().cpu().numpy()

            next_token = np.random.choice(len(probs), p=probs)

            inputs = inputs.tolist()[0]
            inputs.append(next_token)

            if next_token == self.tokenizer.encode(self.tokenizer.eos_token)[0] or len(inputs) >= max_length:
                break
        return self.tokenizer.decode(inputs, skip_special_tokens=True)
    

# model = GPT(
#     vocab_size=len(tokenizer), #tokenizer.vocab_size,
#     n_layers = 16,
#     n_heads = 12,
#     d_model= 768,
#     resid_dropout=0.1,
#     attn_dropout=0.1,
#     d_ff = 3072,
#     ff_dropout=0.1,
#     embed_dropout=0.1,
#     max_len=128,
#     pad_idx = tokenizer.encode(tokenizer.pad_token)[0]
# )

model = GPT(
    vocab_size=len(tokenizer), #tokenizer.vocab_size,
    n_layers = 6,
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
    print("Can't load pre-trained model. Start training from scratch")

optimizer = AdamW(model.parameters(), lr=1.5e-4, betas=(0.9, 0.98), eps=1e-9) #lr = 0.0001; 5e-4; 3e-5
# scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
loss_fn = nn.CrossEntropyLoss().to("cuda") #ignore_index = tokenizer.encode(tokenizer.pad_token)[0]




gpt_interface = GPTInterface(
        model = model, 
        tokenizer = tokenizer, 
        optimizer = optimizer, 
        scheduler = None,
        loss_fn = loss_fn, 
        dataloader = dataloader)


# gpt_interface.fit(epochs = 30, save_path = "saved_models/gpt_model.pt")

print(gpt_interface.predict("I"))


# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# print(count_parameters(model))












