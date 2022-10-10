import sys

import main
import uuid
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
        input_ids = batch['input_ids']
        outputs = model(input_ids=input_ids, labels=input_ids)
        # curr_low = None
        # curr_high = None
        if outputs.loss.item() > high_bound:
        #     curr_high = input_ids
            # all_highs = accelerator.gather_for_metrics((input_ids,))
            # accelerator.print("high", tokenizer.batch_decode(batch['input_ids']))
            # for high in all_highs:
            highs.append(tokenizer.batch_decode(input_ids)[0].lstrip("</s>"))
        elif outputs.loss.item() < low_bound:
            lows.append(tokenizer.batch_decode(input_ids)[0].lstrip("</s>"))
        #     curr_low = input_ids
            # accelerator.print("low", tokenizer.batch_decode(batch['input_ids']))
            # all_lows = accelerator.gather_for_metrics((input_ids,))
            # for low in all_lows:
            #     lows.append(tokenizer.batch_decode(low)[0].lstrip("</s>"))
        # print(curr_low)
        # print(curr_high)
        # print(f"input ids {input_ids}")
        # # all_together = accelerator.gather_for_metrics((input_ids, outputs.loss))
        # accelerator.print(f"all_together {all_together}")
        # # print(len(all_loss))
        # all_lows = all_together
        # all_highs = all_together
        # for low in all_lows:
        #     if low is not None:
        #         lows.append(low)
        # # all_highs = accelerator.gather_for_metrics((curr_high,))
        # # print(len(all_loss))
        # for high in all_highs:
        #     if high is not None:
        #         highs.append(high)
        # # print(len(all_loss))
        # # for loss in all_loss:
        # #     losses.append(loss)
        progress_bar.update(1)

    # all_lows = accelerator.gather_for_metrics((lows,))
    # all_highs = accelerator.gather_for_metrics((highs,))
    # print(lows)
    # print(highs)
    accelerator.print(highs)
    accelerator.print(lows)
    # if accelerator.is_main_process:
    torch.save(highs, f"all_low_{str(uuid.uuid4())}.pt")
    torch.save(lows, f"all_high_{str(uuid.uuid4())}.pt")

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