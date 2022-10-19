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

high_pp_target = "no"
low_pp_target = "yes"
prompt_q = "Is this sentence common?"
prompt_after = ""
labels = []

def check_perplex(model, dataloader, tokenizer, accelerator, high_bound, low_bound, all_lows, all_highs, collect_labels):
    # dataloader, tokenizer = accelerator.prepare(dataloader, tokenizer)
    # model = model.to(accelerator.device)
    model.parallelize()

    for file in os.listdir(main.output_dir):
        if file.startswith("all_low_"):
            curr_lows = torch.load(f"{main.output_dir}/{file}")
            for low in curr_lows:
                all_lows.append(low)
        elif file.startswith("all_high_"):
            curr_highs = torch.load(f"{main.output_dir}/{file}")
            for high in curr_highs:
                all_highs.append(high)

    # all_lows = torch.load("all_low.pt")
    # all_highs = torch.load("all_high.pt")
    # accelerator.print(f"all_high {len(all_highs)} : {all_highs[0]}")
    # accelerator.print(f"all_high {len(all_lows)} : {all_lows[0]}")
    def craft_prompt(example):

        # no_str = "yes."
        # yes_str = "no."
        # prompt_q = "Does this sentence perplex you?"
        post_example = "" if (prompt_after == "" or prompt_after is None) else prompt_after + "\n"
        used_examples = [example]
        few_shot = ""
        for i in range(main.shot):
            if i%2 == 0:
                example_list = all_highs
                curr_target = high_pp_target
            else:
                example_list = all_lows
                curr_target = low_pp_target
            new_ex = example
            while new_ex in used_examples:
                new_ex = random.choice(example_list)
            used_examples.append(new_ex)

            few_shot += f"{prompt_q}\n\"{new_ex}\"\n{post_example}{curr_target}.\n###\n"
        # high_ex2 = high_ex
        # while high_ex2 in used_examples:
        #     high_ex2 = random.choice(all_highs)
        # used_examples.append(high_ex2)
        # high_ex3 = high_ex
        # while high_ex3 in used_examples:
        #     high_ex3 = random.choice(all_highs)
        # used_examples.append(high_ex3)
        # low_ex = example
        # while low_ex in used_examples:
        #     low_ex = random.choice(all_lows)
        # used_examples.append(low_ex)
        # low_ex2 = example
        # while low_ex2 in used_examples:
        #     low_ex2 = random.choice(all_lows)
        # used_examples.append(low_ex2)
        # low_ex3 = example
        # while low_ex3 in used_examples:
        #     low_ex3 = random.choice(all_lows)
        # used_examples.append(low_ex3)

        # return f"{prompt_q} {high_ex} {no_str} {prompt_q} {low_ex} {yes_str} {prompt_q} {high_ex2} {no_str} " \
        #        f"{prompt_q} {low_ex2} {yes_str} {prompt_q} {high_ex3} {no_str} {prompt_q} {low_ex3} {yes_str} {prompt_q} {example} "
        return f"{few_shot}\"{example}\"\n{prompt_q}\n{post_example}"
    win_cnt = [0 for i in range(1001)]
    tot_cnt = 0
    unk_cnt = [0 for i in range(1001)]
    batch_cnt = 0
    print ("FORDOR", len(dataloader))
    empty_ids = torch.tensor([[tokenizer.eos_token_id]]).cuda()
    progress_bar = tqdm(range(len(dataloader)))
    perms = [list(range(len(dataloader))) for i in range(1001)]
    for i in range(1000):
        random.shuffle(perms[i])
    for batch in dataloader:
        input_ids = batch['input_ids'].cuda()
        outputs = model(input_ids=empty_ids, labels=input_ids)
        # outputs2 = model(input_ids=input_ids, labels=input_ids)
        # if outputs1.loss.item() != outputs2.loss.item():
        #     print(f"oh its different!!! {outputs1.loss.item()} {outputs2.loss.item()}")
        # outputs = outputs2
        # curr_low = None
        # curr_high = None
        curr_example = None
        target = None
        curr_example = batch['text'][0]
        # if outputs.loss.item() > (high_bound + low_bound)/2:

        target = high_pp_target if outputs.loss.item() > (high_bound + low_bound)/2.0 else low_pp_target
        # elif outputs.loss.item() < low_bound:
        #     # curr_example = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
        if collect_labels:
            labels.append(target)
        elif curr_example is not None:
            prompt = craft_prompt(curr_example)
            prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            # print("what is this ",prompt_tokens)
            generated_ids = model.generate(prompt_tokens, max_new_tokens=3)
            # generated_text = tokenizer.batch_decode(generated_ids[:,prompt_tokens.size()[1]:], skip_special_tokens=True)[0]
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            lower_text = generated_text.lower()
            for j in range(1001):
                target = labels[perms[j][batch_cnt]]
                if lower_text == target or (lower_text.startswith(target) and not lower_text[len(target)].isalnum()): #FORDOR
                    win_cnt[j] += 1
                elif not (high_pp_target in lower_text or low_pp_target in lower_text):
                    unk_cnt[j] += 1
                    # print(f"unknown : {tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]}")
                    # print(f"WRONG! expected {target} got {generated_text} score: {outputs.loss.item()} prompt: {prompt}")
            tot_cnt += 1

            # print(prompt_tokens.size())
            # print(f"FORDOR actual result::: {generated_text} -expected- ({target})")
        batch_cnt +=1
        progress_bar.update(1)
        res = torch.tensor(win_cnt) / tot_cnt
        # res = [float(win_cnt[j]) / tot_cnt for j in range(1000)]


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
        # if batch_cnt % 100 == 0 and tot_cnt > 0:
            # print(f"tot : {tot_cnt}, wins : {win_cnt}, unks: {unk_cnt}")
            # print(f"win% = {float(win_cnt) / tot_cnt}, unk% = {float(unk_cnt) / tot_cnt}")
    # print("FINALLY:")
    # print(f"tot : {tot_cnt}, wins : {win_cnt}, unks: {unk_cnt}")
    # print(f"win% = {float(win_cnt) / tot_cnt}, unk% = {float(unk_cnt) / tot_cnt}")
    if not collect_labels:
        q_list = [float(x) / 1000 for x in range(0, 1001)]
        global output_dir
        for q in q_list:
            # torch.save(torch.quantile(res, q), f"{output_dir}/quantile-{q}.pt")
            print(f"quantile {q}  {torch.quantile(res, q)}")


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, texts):
        self.texts = texts
        # self.labels = labels

    def __getitem__(self, idx):
        item = {key: (torch.tensor(val[idx]) if key != 'text' else val[idx]) for key, val in self.texts.items()}
        # item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.texts['input_ids'])

    def shuffle(self):
        # c = list(zip(self.texts, self.labels))

        self.texts = random.shuffle(self.texts)

        # a, b = zip(*c)

def main1():
    with torch.no_grad():
        accelerator = None
        args = main.get_args()
        global prompt_q
        global prompt_after
        global high_pp_target
        global low_pp_target
        if args.prompt_q is not None:
            prompt_q = args.prompt_q
        if args.prompt_after is not None:
            prompt_after = args.prompt_after
        if args.high_pp_target is not None:
            high_pp_target = args.high_pp_target
        if args.low_pp_target is not None:
            low_pp_target = args.low_pp_target
        model = AutoModelForSeq2SeqLM.from_pretrained(main.model_name)
        tokenizer = AutoTokenizer.from_pretrained(main.model_name)
        model.eval()
        all_lows = []
        all_highs = []
        for file in os.listdir(main.output_dir):
            if file.startswith("all_low_"):
                curr_lows = torch.load(f"{main.output_dir}/{file}")
                for low in curr_lows:
                    all_lows.append(low)
            elif file.startswith("all_high_"):
                curr_highs = torch.load(f"{main.output_dir}/{file}")
                for high in curr_highs:
                    all_highs.append(high)
        all_examples = all_lows + all_highs

        #dataset.shuffle()
        # dataset = main.get_data()
        # tokenized_examples = [tokenizer(example) for example in all_examples]
        tokenized_examples = tokenizer(all_examples)
        tokenized_examples['text'] = all_examples
        # print("FORDOR")
        print(len(all_examples))
        # print(len(tokenized_examples))
        tokenized_data = MyDataset(tokenized_examples)
        dataloader = DataLoader(tokenized_data, shuffle=False, batch_size=1)
        q5 = torch.load(f"{main.output_dir}/quantile-0.05.pt")
        q95 = torch.load(f"{main.output_dir}/quantile-0.95.pt")
        print("quantiles:")
        print(q5)
        print(q95)
        check_perplex(model, dataloader, tokenizer, accelerator, q95, q5, all_lows, all_highs, True)
        check_perplex(model, dataloader, tokenizer, accelerator, q95, q5, all_lows, all_highs, False)


if __name__ == '__main__':
    sys.exit(main1())
