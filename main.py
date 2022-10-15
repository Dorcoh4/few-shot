import sys

import transformers
import datasets
from evaluate import load
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
import argparse
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import torch

print(f"device count: {torch.cuda.device_count()}")

model_name = "bigscience/T0_3B"
output_dir = "."

def tokenize_data(dataset, tokenizer):

    # the fast tokenizer currently does not work correctly
    # tokenizer.padding_side = "left"

    # context_length = 512

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_data = dataset.map(tokenize_function, batched=True)
    tokenized_data = tokenized_data.remove_columns("text")
    tokenized_data.set_format("torch")
    return tokenized_data



def print_quantiles(model, dataloader, accelerator):
    losses = []
    # dataloader = accelerator.prepare(dataloader)
    # model = model.to(accelerator.device)
    # model = model.cuda()
    progress_bar = tqdm(range(len(dataloader)))
    print("Going over data.........")
    for batch in dataloader:
        input_ids = batch['input_ids'].cuda()
        outputs = model(input_ids=input_ids, labels=input_ids)

        # all_loss = accelerator.gather_for_metrics((outputs.loss.unsqueeze(0),))
        # print(len(all_loss))
        # for loss in all_loss:
        losses.append(outputs.loss.unsqueeze(0))
        progress_bar.update(1)
    res = torch.cat(losses)
    res = res.type(torch.FloatTensor)
    # try:
    #     perplexity = torch.exp(loss)
    # except OverflowError:
    #     perplexity = float("inf")
    q_list = [0.01, 0.99] + [float(x) / 100 for x in range(0,101,5)]
    global output_dir
    for q in q_list:
        torch.save(torch.quantile(res, q), f"{output_dir}/quantile-{q}.pt")
    # accelerator.print(f"quantile 0.01  {torch.quantile(res,0.01)}")
    # accelerator.print(f"quantile 0.05  {torch.quantile(res, 0.05)}")
    # accelerator.print(f"quantile 0.1  {torch.quantile(res, 0.1)}")
    # accelerator.print(f"quantile 0.15  {torch.quantile(res, 0.15)}")
    # accelerator.print(f"quantile 0.2  {torch.quantile(res, 0.2)}")
    # accelerator.print(f"quantile 0.25  {torch.quantile(res, 0.25)}")
    # accelerator.print(f"quantile 0.3  {torch.quantile(res, 0.3)}")
    # accelerator.print(f"quantile 0.4  {torch.quantile(res, 0.4)}")
    # accelerator.print(f"quantile 0.5  {torch.quantile(res, 0.5)}")
    # accelerator.print(f"quantile 0.6  {torch.quantile(res, 0.6)}")
    # accelerator.print(f"quantile 0.7  {torch.quantile(res, 0.7)}")
    # accelerator.print(f"quantile 0.75  {torch.quantile(res, 0.75)}")
    # accelerator.print(f"quantile 0.8  {torch.quantile(res, 0.8)}")
    # accelerator.print(f"quantile 0.85  {torch.quantile(res, 0.85)}")
    # accelerator.print(f"quantile 0.9  {torch.quantile(res, 0.9)}")
    # accelerator.print(f"quantile 0.95  {torch.quantile(res, 0.95)}")
    # accelerator.print(f"quantile 0.99  {torch.quantile(res, 0.99)}")
    # accelerator.print(f"length {res.size()}")

def get_data():
    dataset = datasets.load_dataset("bookcorpus", split="train[:10%]").shuffle(seed=1034).select(range(100))
    return dataset

def get_args():
    parser = argparse.ArgumentParser(description=""" FORDOR

            """, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--output_dir', dest='output_dir', required=True,
                        help='Where shall we write intermediate models + final data to?')
    parser.add_argument('--model_name', dest='model_name', required=True,
                        help='FORDOR')
    args = parser.parse_args()
    global model_name
    global output_dir
    model_name = args.model_name
    output_dir = args.output_dir

    return args

def main1():
    print("starting")
    global model_name
    args = get_args()
    with torch.no_grad():
        accelerator = None
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.parallelize()
        model.eval()
        dataset = get_data()
        tokenized_data = tokenize_data(dataset, tokenizer)
        dataloader = DataLoader(tokenized_data, shuffle=False, batch_size=1)
        print_quantiles(model, dataloader, accelerator)


if __name__ == '__main__':
    sys.exit(main1())








