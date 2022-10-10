import sys

import main

import datasets
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = main.model_name


def save_examples(model, dataloader, accelerator, tokenizer, high_bound, low_bound):
    dataloader= accelerator.prepare(dataloader)
    model = model.to(accelerator.device)
    highs = []
    lows = []
    progress_bar = tqdm(range(len(dataloader)))
    for batch in dataloader:
        outputs = model(input_ids=batch['input_ids'], labels=batch['input_ids'])

        if outputs.loss.item() > high_bound:
            highs.append(tokenizer.batch_decode(batch['input_ids'])[0].lstrip("</s>"))
            # accelerator.print("high", tokenizer.batch_decode(batch['input_ids']))
        elif outputs.loss.item() < low_bound:
            # accelerator.print("low", tokenizer.batch_decode(batch['input_ids']))
            lows.append(tokenizer.batch_decode(batch['input_ids'])[0].lstrip("</s>"))
        # print(len(all_loss))
        # for loss in all_loss:
        #     losses.append(loss)
        progress_bar.update(1)
    all_lows = accelerator.gather_for_metrics((lows,))
    all_highs = accelerator.gather_for_metrics((highs,))
    if accelerator.is_main_process:
        torch.save(all_lows, "all_low.pt")
        torch.save(all_highs, "all_high.pt")

def main1():
    with torch.no_grad():
        accelerator = Accelerator()
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model.eval()
        dataset = datasets.load_dataset("bookcorpus")['train'].shuffle(seed=1034).select(range(50000))
        tokenized_data = main.tokenize_data(dataset, tokenizer)
        dataloader = DataLoader(tokenized_data, shuffle=False, batch_size=1)
        save_examples(model, dataloader, accelerator, tokenizer, 7.75, 3.99)


if __name__ == '__main__':
    sys.exit(main1())