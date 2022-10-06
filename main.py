import transformers
import datasets

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("facebook/opt-6.7b", torch_dtype=torch.float16).cuda()

# the fast tokenizer currently does not work correctly
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b", use_fast=False)
dataset = datasets.load_dataset("bookcorpus")
x = 3














