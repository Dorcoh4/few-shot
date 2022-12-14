import os
import random
import sys

import main
from main import MyDataset
import transformers
import datasets
from evaluate import load
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from experiment_module import ExperimentModule


model_name = main.model_name

high_pp_target = "no"
low_pp_target = "yes"
prompt_q = "Is this a probable sentence?"
prompt_after = ""

def check_perplex(e_model, dataloader, high_bound, low_bound, all_lows, all_highs):
    # dataloader, tokenizer = accelerator.prepare(dataloader, tokenizer)
    # model = model.to(accelerator.device)
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
        permutation = list(range(main.shot))
        random.shuffle(permutation)
        for i in range(main.shot):
            if permutation[i] % 2 == 0:
                example_list = all_highs
                curr_target = high_pp_target
            else:
                example_list = all_lows
                curr_target = low_pp_target
            new_ex = example
            while new_ex in used_examples:
                new_ex = random.choice(example_list)
            used_examples.append(new_ex)

            few_shot += f"Question: {prompt_q}\nSentence: {new_ex}\n{post_example}Answer: {curr_target}\n###\n"
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
        return f"{few_shot}Question: {prompt_q}\nSentence: {example}\n{post_example}Answer:"
    win_cnt = 0
    tot_cnt = 0
    unk_cnt = 0
    batch_cnt = 0
    progress_bar = tqdm(range(len(dataloader)))
    for batch in dataloader:
        input_ids = batch['input_ids'].cuda()
        loss = e_model.get_loss(input_ids)
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

        target = high_pp_target if loss.item() > (high_bound + low_bound)/2.0 else low_pp_target
        # elif outputs.loss.item() < low_bound:
        #     # curr_example = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]

        if curr_example is not None:
            prompt = craft_prompt(curr_example)
            prompt_tokens = e_model.tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            # print("what is this ",prompt_tokens)
            generated_ids = e_model.model.generate(prompt_tokens, max_new_tokens=3)
            generated_text = e_model.get_answer(generated_ids, prompt_tokens)
            
            # generated_text = tokenizer.batch_decode(generated_ids[:,prompt_tokens.size()[1]:], skip_special_tokens=True)[0]
            lower_text = generated_text.lower().lstrip()
            if lower_text == target or (lower_text.startswith(target) and not lower_text[len(target)].isalnum()): #FORDOR
                win_cnt += 1
            elif not (high_pp_target in lower_text or low_pp_target in lower_text):
                unk_cnt += 1
                print(f"unknown : {e_model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]}")
            else:
                print(f"WRONG! expected {target} got {generated_text} score: {loss.item()} prompt: {prompt}")
            tot_cnt += 1

            # print(prompt_tokens.size())
            # print(f"FORDOR actual result::: {generated_text} -expected- ({target})")
        batch_cnt +=1
        progress_bar.update(1)
        if batch_cnt % 100 == 0 and tot_cnt > 0:
            print(f"tot : {tot_cnt}, wins : {win_cnt}, unks: {unk_cnt}")
            print(f"win% = {float(win_cnt) / tot_cnt}, unk% = {float(unk_cnt) / tot_cnt}")
    print("FINALLY:")
    print(f"tot : {tot_cnt}, wins : {win_cnt}, unks: {unk_cnt}")
    print(f"win% = {float(win_cnt) / tot_cnt}, unk% = {float(unk_cnt) / tot_cnt}")




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

        all_lows, all_highs = main.get_double_data_w_names(main.output_dir, "all_low_", " all_high_")
        all_examples = all_lows + all_highs

        e_model = ExperimentModule(args.model_name, args.method)
        e_model.parallelize()
        e_model.model.eval()

        # dataset = main.get_data()
        # tokenized_examples = [tokenizer(example) for example in all_examples]
        tokenized_examples = e_model.tokenizer(all_examples)
        tokenized_examples['text'] = all_examples
        # print("FORDOR")
        print(len(all_examples))
        # print(len(tokenized_examples))
        tokenized_data = MyDataset(tokenized_examples)
        dataloader = DataLoader(tokenized_data, shuffle=True, batch_size=1)
        q5 = torch.load(f"{main.output_dir}/quantile-{args.qlow}.pt")
        q95 = torch.load(f"{main.output_dir}/quantile-{args.qhigh}.pt")
        print("quantiles:")
        print(q5)
        print(q95)
        check_perplex(e_model, dataloader, q95, q5, all_lows, all_highs)


if __name__ == '__main__':
    sys.exit(main1())
