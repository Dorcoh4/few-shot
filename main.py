import transformers
import datasets
from evaluate import load
from torch.utils.data import DataLoader
from accelerate import Accelerator

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

with torch.no_grad():
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b", torch_dtype=torch.float16)
    accelerator = Accelerator()
    model.eval()
    # the fast tokenizer currently does not work correctly
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", use_fast=False)
    # tokenizer.padding_side = "left"
    dataset = datasets.load_dataset("bookcorpus")['train']
    batch_size = 1
    context_length = 512

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_data = dataset.map(tokenize_function, batched=True)
    tokenized_data = tokenized_data.remove_columns("text")
    tokenized_data.set_format("torch")
    dataloader = DataLoader(tokenized_data, shuffle=True, batch_size=batch_size)
    # perplexity = load("perplexity", module_type="metric")
    losses = []

    dataloader, model = accelerator.prepare(dataloader, model)
    for batch in dataloader:
        # batch_encoding = tokenizer(batch['text'], return_tensors="pt")
        outputs = model(input_ids=batch['input_ids'], labels=batch['input_ids'])
        all_loss = accelerator.gather_for_metrics((outputs.loss,))
        for loss in all_loss:
            losses.append(loss)
        x = 3
    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")

    print(perplexity.item())











