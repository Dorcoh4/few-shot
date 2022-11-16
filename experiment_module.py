from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration
import torch
from parallelformers import parallelize

def get_module_and_tokenizer_by_name(name):
    if name.startswith("bigscience/T0"):
        return AutoModelForSeq2SeqLM.from_pretrained(name), AutoTokenizer.from_pretrained(name)
    elif name.startswith("facebook/opt"):
        return AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16), AutoTokenizer.from_pretrained(name, use_fast=False)
    elif name.startswith("t5-"):
        return T5ForConditionalGeneration.from_pretrained(name), T5Tokenizer.from_pretrained(name)
    # else:
    #     # print("FORDOR1")
    #     model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
    #     tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    #     return model, tokenizer


class ExperimentModule:
    def __init__(self, name):
        self.name = name
        model, tokenizer = get_module_and_tokenizer_by_name(name)
        self.model = model
        self.tokenizer = tokenizer

    def get_loss(self, input_ids):
        if self.name.startswith("bigscience/T0"):
            empty_ids = torch.tensor([[self.tokenizer.eos_token_id]]).cuda()
            outputs = self.model(input_ids=empty_ids, labels=input_ids)
            return outputs.loss
        elif self.name.startswith("facebook/opt"):
            return self.model(input_ids=input_ids, labels=input_ids).loss
        elif self.name.startswith("t5-"):
            empty_ids = torch.tensor([[self.tokenizer.eos_token_id]]).cuda()
            outputs = self.model(input_ids=empty_ids, labels=input_ids)
            return outputs.loss
        else:
            empty_ids = torch.tensor([[self.tokenizer.eos_token_id]]).cuda()
            outputs = self.model(input_ids=empty_ids, labels=input_ids)
            return outputs.loss


    def parallelize(self):
        if self.name.startswith("facebook/opt"): #or  "gpt-neo" in self.name:
            # print("FORDOR2")
            # parallelize(self.model, num_gpus=5, fp16=True, verbose='detail')
            self.model = self.model.cuda()
        else:
            self.model.parallelize()