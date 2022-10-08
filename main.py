import transformers
import datasets
from evaluate import load
from torch.utils.data import DataLoader
from accelerate import Accelerator

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

with torch.no_grad():
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-2.7b", torch_dtype=torch.float16)
    accelerator = Accelerator()
    model.eval()
    # the fast tokenizer currently does not work correctly
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b", use_fast=False)
    # tokenizer.padding_side = "left"
    dataset = datasets.load_dataset("bookcorpus")['train']
    batch_size = 1
    context_length = 512

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_data = dataset.map(tokenize_function, batched=True)
    tokenized_data = tokenized_data.remove_columns("text")
    tokenized_data.set_format("torch")
    dataloader = DataLoader(tokenized_data, shuffle=False, batch_size=batch_size)
    # perplexity = load("perplexity", module_type="metric")
    losses = []

    dataloader = accelerator.prepare(dataloader)
    model = model.to(accelerator.device)
    for batch in dataloader:
        outputs = model(input_ids=batch['input_ids'], labels=batch['input_ids'])
        all_loss = accelerator.gather_for_metrics((outputs.loss.unsqueeze(0),))
        for loss in all_loss:
            losses.append(loss)
    res = torch.cat(losses)
    res = res.type(torch.FloatTensor)
    # try:
    #     perplexity = torch.exp(loss)
    # except OverflowError:
    #     perplexity = float("inf")

    print(f"quantile 0.01  {torch.quantile(res,0.01)}")
    print(f"quantile 0.05  {torch.quantile(res, 0.05)}")
    print(f"quantile 0.1  {torch.quantile(res, 0.1)}")
    print(f"quantile 0.15  {torch.quantile(res, 0.15)}")
    print(f"quantile 0.2  {torch.quantile(res, 0.2)}")
    print(f"quantile 0.25  {torch.quantile(res, 0.25)}")
    print(f"quantile 0.3  {torch.quantile(res, 0.3)}")
    print(f"quantile 0.5  {torch.quantile(res, 0.4)}")
    print(f"quantile 0.6  {torch.quantile(res, 0.6)}")
    print(f"quantile 0.7  {torch.quantile(res, 0.7)}")
    print(f"quantile 0.75  {torch.quantile(res, 0.75)}")
    print(f"quantile 0.8  {torch.quantile(res, 0.8)}")
    print(f"quantile 0.85  {torch.quantile(res, 0.85)}")
    print(f"quantile 0.9  {torch.quantile(res, 0.9)}")
    print(f"quantile 0.95  {torch.quantile(res, 0.95)}")
    print(f"quantile 0.99  {torch.quantile(res, 0.99)}")










