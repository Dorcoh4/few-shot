import sys

import main
import uuid
import datasets
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import torch

model_name = main.model_name


def save_examples(model, dataloader, accelerator, tokenizer, high_bound, low_bound):
    # dataloader= accelerator.prepare(dataloader)
    # model = model.to(accelerator.device)
    model.parallelize()
    highs = []
    lows = []
    progress_bar = tqdm(range(len(dataloader)))
    for batch in dataloader:
        input_ids = batch['input_ids'].cuda()
        outputs = model(input_ids=input_ids, labels=input_ids)
        # curr_low = None
        # curr_high = None
        if outputs.loss.item() > high_bound:
        #     curr_high = input_ids
            # all_highs = accelerator.gather_for_metrics((input_ids,))
            # accelerator.print("high", tokenizer.batch_decode(batch['input_ids']))
            # for high in all_highs:
            highs.append(tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0])
        elif outputs.loss.item() < low_bound:
            lows.append(tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0])
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
    print(highs)
    print(lows)
    # if accelerator.is_main_process:
    torch.save(highs, f"{main.output_dir}/all_low_{str(uuid.uuid4())}.pt")
    torch.save(lows, f"{main.output_dir}/all_high_{str(uuid.uuid4())}.pt")

def main1():
    args = main.get_args()
    with torch.no_grad():
        accelerator = None
        model = AutoModelForSeq2SeqLM.from_pretrained(main.model_name)
        tokenizer = AutoTokenizer.from_pretrained(main.model_name)
        model.eval()
        dataset = main.get_data()
        tokenized_data = main.tokenize_data(dataset, tokenizer)
        dataloader = DataLoader(tokenized_data, shuffle=False, batch_size=1)
        q5 = torch.load(f"{main.output_dir}/quantile-0.05.pt")
        q95 = torch.load(f"{main.output_dir}/quantile-0.95.pt")
        save_examples(model, dataloader, accelerator, tokenizer, q95, q5)


if __name__ == '__main__':
    sys.exit(main1())