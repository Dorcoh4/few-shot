import transformers
import datasets
from evaluate import load
from torch.utils.data import DataLoader
from accelerate import Accelerator

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b", torch_dtype=torch.float16).cuda()

# the fast tokenizer currently does not work correctly
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", use_fast=False)
# tokenizer.padding_side = "left"
dataset = datasets.load_dataset("bookcorpus")['train']
batch_size = 1


# def tokenize_function(examples):
#     input_ids = []
#     attention_mask = []
#     for example in examples["text"]:
#         # print(example)
#         encoded_batch = tokenizer.encode_plus(example, return_tensors="pt")
#         input_ids.append(encoded_batch['input_ids'].squeeze())
#         attention_mask.append(encoded_batch['attention_mask'].squeeze())
#     return {'input_ids': input_ids, 'attention_mask': attention_mask}


tokenized_data = dataset.select(range(100)) #.map(tokenize_function, batched=True)
# tokenized_data = tokenized_data.remove_columns("text")
dataloader = DataLoader(tokenized_data, shuffle=True, batch_size=batch_size)
# perplexity = load("perplexity", module_type="metric")
for batch in dataloader:
    batch_encoding = tokenizer(batch['text'], return_tensors="pt")
    results = model(input_ids=batch_encoding['input_ids'].cuda())

    x = 3














