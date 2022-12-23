import sys
import os
import main
from main import MyDataset
import uuid
import datasets
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from experiment_module import ExperimentModule

model_name = main.model_name


def save_examples(e_model, dataloader, high_bound, low_bound):
    # dataloader= accelerator.prepare(dataloader)
    # model = model.to(accelerator.device)
    highs = []
    lows = []
    progress_bar = tqdm(range(len(dataloader)))
    for batch in dataloader:
        input_ids = batch['input_ids'].cuda()
        loss = e_model.get_loss(input_ids)
        # curr_low = None
        # curr_high = None
        if loss.item() > high_bound:
        #     curr_high = input_ids
            # all_highs = accelerator.gather_for_metrics((input_ids,))
            # accelerator.print("high", tokenizer.batch_decode(batch['input_ids']))
            # for high in all_highs:
            highs.append(batch['text'][0])
        elif loss.item() < low_bound:
            lows.append(batch['text'][0])
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
    print(len(highs))
    # print(highs)
    print(len(lows))
    # if accelerator.is_main_process:
    torch.save(lows, f"{main.output_dir}/all_low_{str(uuid.uuid4())}.pt")
    torch.save(highs, f"{main.output_dir}/all_high_{str(uuid.uuid4())}.pt")

def main1():
    args = main.get_args()
    with torch.no_grad():
        all_data = main.get_single_data(args)

        e_model = ExperimentModule(args.model_name, args.method)
        e_model.parallelize()
        e_model.model.eval()
        tokenized_examples = e_model.tokenizer(all_data)
        tokenized_examples['text'] = all_data
        # print("FORDOR")
        print(len(all_data))
        # print(len(tokenized_examples))
        tokenized_data = MyDataset(tokenized_examples)
        dataloader = DataLoader(tokenized_data, shuffle=False, batch_size=1)
        q5 = torch.load(f"{main.output_dir}/quantile-{args.qlow}.pt")
        q95 = torch.load(f"{main.output_dir}/quantile-{args.qhigh}.pt")
        print("quantiles:")
        print(q5)
        print(q95)
        save_examples(e_model, dataloader, q95, q5)


if __name__ == '__main__':
    sys.exit(main1())