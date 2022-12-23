import os
import random
import sys

import uuid
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

def qa_attn(e_model, dataloader):


    win_cnt = 0
    tot_cnt = 0
    unk_cnt = 0
    batch_cnt = 0
    to_save = []
    progress_bar = tqdm(range(len(dataloader)))
    for batch in dataloader:
        input_ids = batch['input_ids'].cuda()
        # loss = e_model.get_loss(input_ids)
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

        target = "positive" if batch['labels'] != 0 else "negative"
        # elif outputs.loss.item() < low_bound:
        #     # curr_example = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]

        if curr_example is not None:
            # prompt = craft_prompt(curr_example)
            # prompt_tokens = e_model.tokenizer(in, return_tensors="pt").input_ids.cuda()
            # print("what is this ",prompt_tokens)
            chosen_token = e_model.get_word_from_slice(input_ids)
            to_save.append((batch['text'][0], chosen_token.item()))
            # generated_ids = e_model.model.generate(input_ids, max_new_tokens=3)
            # generated_text = e_model.get_answer(generated_ids, input_ids)
            # generated_text = tokenizer.batch_decode(generated_ids[:,prompt_tokens.size()[1]:], skip_special_tokens=True)[0]
    #         lower_text = generated_text.lower().lstrip()
    #         if lower_text == target or (lower_text.startswith(target) and not lower_text[len(target)].isalnum()): #FORDOR
    #             win_cnt += 1
    #         elif not ("negative" in lower_text or "positive" in lower_text):
    #             unk_cnt += 1
    #             print(f"unknown : {e_model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]}")
    #         else:
    #             print(f"WRONG! expected {target} got {generated_text} score: X sentence: {batch['text']}")
    #         tot_cnt += 1
    #
    #
    #         # print(prompt_tokens.size())
    #         # print(f"FORDOR actual result::: {generated_text} -expected- ({target})")
    #     batch_cnt +=1
        progress_bar.update(1)
    #     if batch_cnt % 100 == 0 and tot_cnt > 0:
    #         print(f"tot : {tot_cnt}, wins : {win_cnt}, unks: {unk_cnt}")
    #         print(f"win% = {float(win_cnt) / tot_cnt}, unk% = {float(unk_cnt) / tot_cnt}, e_win% = {float(win_cnt) / (tot_cnt - unk_cnt)}")
    # print("FINALLY:")
    # print(f"tot : {tot_cnt}, wins : {win_cnt}, unks: {unk_cnt}")
    # print(f"win% = {float(win_cnt) / tot_cnt}, unk% = {float(unk_cnt) / tot_cnt}")
    print(len(to_save))
    torch.save(to_save, f"{main.output_dir}/all_labeled_{str(uuid.uuid4())}.pt")



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

        e_model = ExperimentModule(args.model_name, args.method)
        e_model.parallelize()
        e_model.model.eval()

        # dataset = main.get_data()
        # dataset = datasets.load_dataset("sst2", split="train").shuffle(seed=1034).select(range(int(args.num_examples)))
        all_data = main.get_mixed_data_w_names(args.output_dir + "/../", "data_good_")
        # all_examples = all_lows + all_highs

        all_examples, all_labels = zip(*all_data)
        all_examples_with_prompt = [
            f"Sentence: {sentence}\nQuestion: Does this sentence convey a negative or positive sentiment? Please answer with a single word. \nAnswer:"
            for sentence in all_examples]
        tokenized_Examples = e_model.tokenizer(all_examples_with_prompt)
        tokenized_Examples['text'] = all_examples
        my_data = main.MySplitDataset(tokenized_Examples, all_labels)
        # tokenized_examples = [tokenizer(example) for example in all_examples]
        # tokenized_examples['text'] = all_examples
        # print("FORDOR")
        # print(len(all_examples))
        # print(len(tokenized_examples))
        # tokenized_data = MyDataset(tokenized_examples)
        dataloader = DataLoader(my_data, shuffle=True, batch_size=1)
        # q5 = torch.load(f"{main.output_dir}/quantile-{args.qlow}.pt")
        # q95 = torch.load(f"{main.output_dir}/quantile-{args.qhigh}.pt")
        # print("quantiles:")
        # print(q5)
        # print(q95)
        qa_attn(e_model, dataloader)


if __name__ == '__main__':
    sys.exit(main1())
