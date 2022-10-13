import os
import random
import sys

import main

import transformers
import datasets
from evaluate import load
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import torch

model_name = main.model_name


def check_perplex(model, dataloader, tokenizer, accelerator, high_bound, low_bound, all_lows, all_highs):
    # dataloader, tokenizer = accelerator.prepare(dataloader, tokenizer)
    # model = model.to(accelerator.device)
    model.parallelize()

    for file in os.listdir("."):
        if file.startswith("all_low_"):
            curr_lows = torch.load(file)
            for low in curr_lows:
                all_lows.append(low)
        elif file.startswith("all_high_"):
            curr_highs = torch.load(file)
            for high in curr_highs:
                all_highs.append(high)

    # all_lows = torch.load("all_low.pt")
    # all_highs = torch.load("all_high.pt")
    # accelerator.print(f"all_high {len(all_highs)} : {all_highs[0]}")
    # accelerator.print(f"all_high {len(all_lows)} : {all_lows[0]}")
    def craft_prompt(example):
        no_str = "no."
        yes_str = "yes."
        prompt_q = "is the following sentence short?\n"
        used_examples = [example]
        high_ex = example
        while high_ex in used_examples:
            high_ex = random.choice(all_highs)
        used_examples.append(high_ex)
        high_ex2 = high_ex
        while high_ex2 in used_examples:
            high_ex2 = random.choice(all_highs)
        used_examples.append(high_ex2)
        high_ex3 = high_ex
        while high_ex3 in used_examples:
            high_ex3 = random.choice(all_highs)
        used_examples.append(high_ex3)
        low_ex = example
        while low_ex in used_examples:
            low_ex = random.choice(all_lows)
        used_examples.append(low_ex)
        low_ex2 = example
        while low_ex2 in used_examples:
            low_ex2 = random.choice(all_lows)
        used_examples.append(low_ex2)
        low_ex3 = example
        while low_ex3 in used_examples:
            low_ex3 = random.choice(all_lows)
        used_examples.append(low_ex3)

        return f"{prompt_q}{high_ex}\n{no_str}\n{prompt_q}{low_ex}\n{yes_str}\n{prompt_q}{high_ex2}\n{no_str}\n" \
               f"{prompt_q}{low_ex2}\n{yes_str}\n{prompt_q}{high_ex3}\n{no_str}\n{prompt_q}{low_ex3}\n{yes_str}\n{prompt_q}{example}\n"
    win_cnt = 0
    tot_cnt = 0
    unk_cnt = 0
    batch_cnt = 0
    progress_bar = tqdm(range(len(dataloader)))
    for batch in dataloader:
        input_ids = batch['input_ids'].cuda()
        outputs = model(input_ids=input_ids, labels=input_ids)
        # curr_low = None
        # curr_high = None
        curr_example = None
        target = None
        if outputs.loss.item() > high_bound:
            curr_example = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
            target = "no"
        elif outputs.loss.item() < low_bound:
            curr_example = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
            target = "yes"
        if curr_example is not None:
            prompt = craft_prompt(curr_example)
            prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            # print("what is this ",prompt_tokens)
            generated_ids = model.generate(prompt_tokens, max_new_tokens=3)
            # generated_text = tokenizer.batch_decode(generated_ids[:,prompt_tokens.size()[1]:], skip_special_tokens=True)[0]
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            if target in generated_text.lower(): #FORDOR
                win_cnt += 1
            elif not ("no" in generated_text.lower() or "yes" in generated_text.lower()):
                unk_cnt += 1
                print(f"unknown : {tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]}")
            else:
                print(f"WRONG! expected {target} got {generated_text} score: {outputs.loss.item()} prompt: {prompt}")
            tot_cnt += 1

            # print(prompt_tokens.size())
            # print(f"FORDOR actual result::: {generated_text} -expected- ({target})")
        batch_cnt +=1
        progress_bar.update(1)
        if batch_cnt % 1000 == 0 and tot_cnt > 0:
            print(f"tot : {tot_cnt}, wins : {win_cnt}, unks: {unk_cnt}")
            print(f"win% = {float(win_cnt) / tot_cnt}, unk% = {float(unk_cnt) / tot_cnt}")
    print("FINALLY:")
    print(f"tot : {tot_cnt}, wins : {win_cnt}, unks: {unk_cnt}")
    print(f"win% = {float(win_cnt) / tot_cnt}, unk% = {float(unk_cnt) / tot_cnt}")


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, texts):
        self.texts = texts
        # self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.texts.items()}
        # item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.texts)

    def shuffle(self):
        # c = list(zip(self.texts, self.labels))

        self.texts = random.shuffle(self.texts)

        # a, b = zip(*c)

def main1():
    with torch.no_grad():
        accelerator = None
        args = main.get_args()
        model = AutoModelForSeq2SeqLM.from_pretrained(main.model_name)
        tokenizer = AutoTokenizer.from_pretrained(main.model_name)
        model.eval()
        all_lows = []
        all_highs = []
        for file in os.listdir("."):
            if file.startswith("all_low_"):
                curr_lows = torch.load(file)
                for low in curr_lows:
                    all_lows.append(low)
            elif file.startswith("all_high_"):
                curr_highs = torch.load(file)
                for high in curr_highs:
                    all_highs.append(high)
        all_examples = all_lows + all_highs

        #dataset.shuffle()
        # dataset = main.get_data()
        tokenized_examples = tokenizer(all_examples)
        tokenized_data = MyDataset(tokenized_examples)
        dataloader = DataLoader(tokenized_data, shuffle=True, batch_size=1)
        q5 = torch.load(f"{main.output_dir}/quantile-0.05.pt")
        q95 = torch.load(f"{main.output_dir}/quantile-0.95.pt")
        check_perplex(model, dataloader, tokenizer, accelerator, q95, q5, all_lows, all_highs)




if __name__ == '__main__':
    sys.exit(main1())
