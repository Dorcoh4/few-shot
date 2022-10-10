import sys

import main

import transformers
import datasets
from evaluate import load
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = main.model_name


def check_perplex(model, dataloader, accelerator, high_bound, low_bound):
    dataloader = accelerator.prepare(dataloader)
    model = model.to(accelerator.device)
    all_lows = torch.load("all_low.pt")
    all_highs = torch.load("all_high.pt")
    accelerator.print(all_lows)
    accelerator.print(all_highs)

def main1():
    with torch.no_grad():
        accelerator = Accelerator()
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        model.eval()
        dataset = datasets.load_dataset("bookcorpus")['train'].shuffle(seed=1034).select(range(50000))
        tokenized_data = main.tokenize_data(dataset, tokenizer)
        dataloader = DataLoader(tokenized_data, shuffle=False, batch_size=1)
        check_perplex(model, dataloader, accelerator, 100, 0)



if __name__ == '__main__':
    sys.exit(main1())