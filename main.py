import sys

import transformers
import datasets
from evaluate import load
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print(f"device count: {torch.cuda.device_count()}")

model_name = "facebook/opt-1.3b"

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
    dataloader = accelerator.prepare(dataloader)
    model = model.to(accelerator.device)
    progress_bar = tqdm(range(len(dataloader)))
    accelerator.print("Going over data.........")
    for batch in dataloader:
        input_ids = batch['input_ids']
        outputs = model(input_ids=input_ids, labels=input_ids)

        all_loss = accelerator.gather_for_metrics((outputs.loss.unsqueeze(0),))
        # print(len(all_loss))
        for loss in all_loss:
            losses.append(loss)
        progress_bar.update(1)
    res = torch.cat(losses)
    res = res.type(torch.FloatTensor)
    # try:
    #     perplexity = torch.exp(loss)
    # except OverflowError:
    #     perplexity = float("inf")

    accelerator.print(f"quantile 0.01  {torch.quantile(res,0.01)}")
    accelerator.print(f"quantile 0.05  {torch.quantile(res, 0.05)}")
    accelerator.print(f"quantile 0.1  {torch.quantile(res, 0.1)}")
    accelerator.print(f"quantile 0.15  {torch.quantile(res, 0.15)}")
    accelerator.print(f"quantile 0.05  {torch.quantile(res, 0.05)}")
    accelerator.print(f"quantile 0.2  {torch.quantile(res, 0.2)}")
    accelerator.print(f"quantile 0.25  {torch.quantile(res, 0.25)}")
    accelerator.print(f"quantile 0.3  {torch.quantile(res, 0.3)}")
    accelerator.print(f"quantile 0.4  {torch.quantile(res, 0.4)}")
    accelerator.print(f"quantile 0.5  {torch.quantile(res, 0.5)}")
    accelerator.print(f"quantile 0.6  {torch.quantile(res, 0.6)}")
    accelerator.print(f"quantile 0.7  {torch.quantile(res, 0.7)}")
    accelerator.print(f"quantile 0.75  {torch.quantile(res, 0.75)}")
    accelerator.print(f"quantile 0.8  {torch.quantile(res, 0.8)}")
    accelerator.print(f"quantile 0.85  {torch.quantile(res, 0.85)}")
    accelerator.print(f"quantile 0.9  {torch.quantile(res, 0.9)}")
    accelerator.print(f"quantile 0.95  {torch.quantile(res, 0.95)}")
    accelerator.print(f"quantile 0.99  {torch.quantile(res, 0.99)}")
    accelerator.print(f"length {res.size()}")
def main1():
    print("starting")
    with torch.no_grad():
        accelerator = Accelerator()
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        model.eval()
        dataset = datasets.load_dataset("bookcorpus")['train'].shuffle(seed=1034).select(range(50000))
        tokenized_data = tokenize_data(dataset, tokenizer)
        dataloader = DataLoader(tokenized_data, shuffle=False, batch_size=1)
        print_quantiles(model, dataloader, accelerator)


if __name__ == '__main__':
    print("startinger")
    sys.exit(main1())








