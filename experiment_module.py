from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration
import torch
import random
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
    def __init__(self, name, method):
        self.name = name
        model, tokenizer = get_module_and_tokenizer_by_name(name)
        self.model = model
        self.tokenizer = tokenizer
        self.method = method

    def get_loss(self, input_ids):
        if self.name.startswith("bigscience/T0"):
            empty_ids = torch.tensor([[self.tokenizer.eos_token_id]]).cuda()
            outputs = self.model(input_ids=empty_ids, labels=input_ids)
            return outputs.loss
        elif self.name.startswith("facebook/opt"):
            if self.method == "perplexity":
                return self.model(input_ids=input_ids, labels=input_ids).loss
            else:
                output = self.model(input_ids=input_ids, output_attentions=True)
                attentions = output.attentions
                if self.method == "attn1":
                    all_layers_all_heads = attentions[-1][0] #FORDOR is this the right one?
                elif self.method == "attn2":
                    all_layers_all_heads = sum(attentions)[0]
                elif self.method == "attn3":
                    all_layers_all_heads = attentions[0][0]
                sum_attentions = all_layers_all_heads[:, -1, :].sum(dim=0)
                first_half = sum_attentions[1 : 1 + (len(sum_attentions)-1)//2]
                second_half = sum_attentions[1 + (len(sum_attentions)-1)//2:]
            return first_half.sum()/len(first_half) - second_half.sum()/len(second_half)

        elif self.name.startswith("t5-"):
            empty_ids = torch.tensor([[self.tokenizer.eos_token_id]]).cuda()
            outputs = self.model(input_ids=empty_ids, labels=input_ids)
            return outputs.loss
        else:
            empty_ids = torch.tensor([[self.tokenizer.eos_token_id]]).cuda()
            outputs = self.model(input_ids=empty_ids, labels=input_ids)
            return outputs.loss

    def get_word_from_slice(self, input_ids):
        output = self.model(input_ids=input_ids, output_attentions=True)
        attentions = output.attentions
        if self.method == "attn1":
            all_layers_all_heads = attentions[-1][0]  # FORDOR is this the right one?
        elif self.method == "attn2":
            all_layers_all_heads = sum(attentions)[0]
        elif self.method == "attn3":
            all_layers_all_heads = attentions[0][0]
        elif self.method == "rollout":
            rollout = attentions[0][0]
            rollout = rollout.type(torch.float32)
            for l in range(1,len(attentions)):
                current = attentions[l][0]
                current = current.type(torch.float32)
                rollout = torch.matmul(current, rollout)
            all_layers_all_heads = rollout
        sum_attentions = all_layers_all_heads[:, -1, :].sum(dim=0)
        return torch.argmax(sum_attentions[4:input_ids[0].tolist().index(50118)])

    def get_word(self, input_ids):
        output = self.model(input_ids=input_ids, output_attentions=True)
        attentions = output.attentions
        if self.method == "attn1":
            all_layers_all_heads = attentions[-1][0]  # FORDOR is this the right one?
        elif self.method == "attn2":
            all_layers_all_heads = sum(attentions)[0]
        elif self.method == "attn3":
            all_layers_all_heads = attentions[0][0]
        elif self.method == "rollout":
            rollout = attentions[0][0]
            for l in range(1,len(attentions)):
                current = attentions[l][0]
                rollout = torch.matmul(current, rollout)
            all_layers_all_heads = rollout
        sum_attentions = all_layers_all_heads[:, -1, :].sum(dim=0)
        return torch.argmax(sum_attentions[1:])

    def parallelize(self):
        if self.name.startswith("facebook/opt"): #or  "gpt-neo" in self.name:
            # print("FORDOR2")
            # parallelize(self.model, num_gpus=5, fp16=True, verbose='detail')
            self.model = self.model.cuda()
        else:
            self.model.parallelize()

    def get_answer(self, generated_ids, prompt_tokens):
        if self.name.startswith("facebook/opt"): #or  "gpt-neo" in self.name:
            return self.tokenizer.batch_decode(generated_ids[:,prompt_tokens.size()[1]:], skip_special_tokens=True)[0]
        else:
            return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
