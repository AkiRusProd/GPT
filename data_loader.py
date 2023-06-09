import torch
import os

# I like to move downloaded datasets to another location
os.environ['HF_DATASETS_CACHE'] = 'D:\\Code\\Huggingface_cache\\'


from transformers import DataCollatorWithPadding
from transformers import GPT2Tokenizer
from datasets import load_dataset


class MyDataset(torch.utils.data.Dataset):
    def __init__(self,  data, tokenizer, max_length=64): #128
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []

        for text in data['text']:
           
            form = '<bos>' + text + '<eos>'

            encodings_dict = tokenizer(form, 
                                        truncation=True, 
                                        max_length=max_length, 
                                        padding="max_length")

            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attn_masks': self.attn_masks[idx]
        }
    


if __name__ == '__main__':
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    SPECIAL_TOKENS = {'bos_token':'<bos>','eos_token' :'<eos>', 'pad_token':'<pad>', 'sep_token': '<sep>'}
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    
    data = load_dataset("tweet_eval", "emoji", split="train")
    print(data)

    dataset = MyDataset(data, tokenizer)
    torch.save(dataset, 'dataset.pt')
