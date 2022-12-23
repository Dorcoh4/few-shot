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
from sklearn import metrics

model_name = main.model_name

prompt_q = "What is the index of the word in the sentence which reveals the most sentiment?"
prompt_after = ""

def choose_word(e_model, dataloader, all_pairs):
    random.shuffle(all_pairs)
    def craft_prompt(example):

        # no_str = "yes."
        # yes_str = "no."
        # prompt_q = "Does this sentence perplex you?"
        post_example = "" if (prompt_after == "" or prompt_after is None) else prompt_after + "\n"
        used_examples = [example]
        few_shot = ""
        for i in range(main.shot):
            new_ex = example
            while new_ex in used_examples:
                new_ex = random.choice(all_pairs)
            used_examples.append(new_ex)

            few_shot += f"Question: {prompt_q}\nSentence: {new_ex[0]}\n{post_example}Answer: {new_ex[1]}\n###\n"
        return f"{few_shot}Question: {prompt_q}\nSentence: {example}\n{post_example}Answer:"


    win_cnt = 0
    tot_cnt = 0
    unk_cnt = 0
    rand_cnt = 0
    batch_cnt = 0
    predictions = []
    randoms = []
    truths = []
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

        target = int(batch['labels'])
        # elif outputs.loss.item() < low_bound:
        #     # curr_example = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]

        if curr_example is not None:
            # prompt = craft_prompt(curr_example)
            # prompt_tokens = e_model.tokenizer(in, return_tensors="pt").input_ids.cuda()
            # print("what is this ",prompt_tokens)
            predicted_token = e_model.get_word(input_ids)
            prompt = craft_prompt(curr_example)
            prompt_tokens = e_model.tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            # print("what is this ",prompt_tokens)
            generated_ids = e_model.model.generate(prompt_tokens, max_new_tokens=3)
            generated_text = e_model.get_answer(generated_ids, prompt_tokens)
            generated_text = generated_text.lstrip()
            j = 0
            res = 0
            while generated_text[j].isdigit():
                j += 1
            if j != 0:
                res = int(generated_text[:j])
                # if res == target:
                #     win_cnt += 1
                # else:
                #     print(f"WRONG! expected {target} got {generated_text} score: X prompt: {prompt}")
            else:
                res = -1
                print(f"unknown : {e_model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]}")
            rand_res = random.randrange(len(input_ids[0]-1))
                # rand_cnt += 1
            tot_cnt += 1
            truths.append(target)
            predictions.append(res)
            randoms.append(rand_res)

            # print(prompt_tokens.size())
            # print(f"FORDOR actual result::: {generated_text} -expected- ({target})")
        batch_cnt +=1
        progress_bar.update(1)
        # if batch_cnt % 100 == 0 and tot_cnt > 0:
        #     print(f"tot : {tot_cnt}, wins : {win_cnt}, unks: {unk_cnt}, rand: {rand_cnt}")
        #     print(f"win% = {float(win_cnt) / tot_cnt}, unk% = {float(unk_cnt) / tot_cnt}, e_win% = {float(win_cnt) / (tot_cnt - unk_cnt)}")
        #     print(f"random win% = {float(rand_cnt) / tot_cnt}")
    print("FINALLY:")
    # print(f"tot : {tot_cnt}, wins : {win_cnt}, unks: {unk_cnt}")
    # print(f"win% = {float(win_cnt) / tot_cnt}, unk% = {float(unk_cnt) / tot_cnt}")
    print(metrics.classification_report(truths, predictions, digits=4))
    print("For random:::")
    print(metrics.classification_report(truths, randoms, digits=4))




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
        all_pairs = main.get_mixed_data_w_names(args.output_dir, "all_labeled_",)

        all_examples, all_labels = zip(*all_pairs)
        # all_examples = [f"Sentence: {sentence}\nQuestion: Does this sentence convey a negative or positive sentiment? Please answer with a single word. \nAnswer:"
        #              for sentence in all_examples]
        tokenized_Examples = e_model.tokenizer(all_examples)
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
        choose_word(e_model, dataloader, all_pairs)


if __name__ == '__main__':
    sys.exit(main1())
