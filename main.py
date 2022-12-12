import random
import sys
import os
import transformers
import datasets
from evaluate import load
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
import argparse
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import torch

import main
from experiment_module import ExperimentModule

print(f"device count: {torch.cuda.device_count()}")

model_name = "bigscience/T0_3B"
output_dir = "."
shot = 0
num_examples = 12000
method = "perplexity"
def tokenize_data(dataset, tokenizer):

    # the fast tokenizer currently does not work correctly
    # tokenizer.padding_side = "left"

    # context_length = 512
    def random_cut(examples):
        res = []
        for example in examples['text']:
            ex_list = example.split()
            if len(ex_list) > 1:
                ex_list = ex_list[0:random.randrange(1, len(ex_list))]
                res.append(" ".join(ex_list))

            else:
                print(f"FORDOR234: {example}")
                res.append(example)
        return {"text": res}
    def tokenize_function(examples):
        return tokenizer(examples["text"])

    # dataset = dataset.map(random_cut, batched=True)
    tokenized_data = dataset.map(tokenize_function, batched=True)
    # tokenized_data = tokenized_data.remove_columns("text")
    tokenized_data.set_format("torch")
    return tokenized_data



def print_quantiles(e_model, dataloader):
    losses = []
    # dataloader = accelerator.prepare(dataloader)
    # model = model.to(accelerator.device)
    # model = model.cuda()
    progress_bar = tqdm(range(len(dataloader)))
    print("Going over data.........")
    for batch in dataloader:
        input_ids = batch['input_ids'].cuda()
        if len(input_ids[0]) < 3:
            continue
        loss = e_model.get_loss(input_ids)

        losses.append(loss.unsqueeze(0))
        progress_bar.update(1)
    res = torch.cat(losses)
    res = res.type(torch.FloatTensor)
    # try:
    #     perplexity = torch.exp(loss)
    # except OverflowError:
    #     perplexity = float("inf")
    q_list = [0.01, 0.99] + [float(x) / 100 for x in range(0,101,5)]
    global output_dir
    for q in q_list:
        torch.save(torch.quantile(res, q), f"{output_dir}/quantile-{q}.pt")
        print(f"quantile: {q} == {torch.quantile(res, q)}")
    # accelerator.print(f"quantile 0.01  {torch.quantile(res,0.01)}")
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

def get_data():
    dataset = datasets.load_dataset("bookcorpus", split="train[:20%]").shuffle(seed=1034).select(range(num_examples))
    return dataset

def get_args():
    parser = argparse.ArgumentParser(description=""" FORDOR

            """, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--output_dir', dest='output_dir', required=True,
                        help='Where shall we write intermediate models + final data to?')
    parser.add_argument('--model_name', dest='model_name', required=True,
                        help='FORDOR')
    parser.add_argument('-o', '--prompt_q', dest='prompt_q',
                        help="shows output")
    parser.add_argument('--prompt_after', dest='prompt_after',
                        help="shows output")
    parser.add_argument('--high_pp_target', dest='high_pp_target',
                        help="shows output")
    parser.add_argument('--low_pp_target', dest='low_pp_target',
                        help="shows output")
    parser.add_argument('--shot', dest='shot', default=0,
                        help="shows output")
    parser.add_argument('--num_examples', dest='num_examples', default=12000,
                        help="shows output")
    parser.add_argument('--method', dest='method', default="perplexity",
                        help="shows output")
    parser.add_argument('--qhigh', dest='qhigh', default="0.95",
                        help="shows output")
    parser.add_argument('--qlow', dest='qlow', default="0.05",
                        help="shows output")
    parser.add_argument('--d_prefix', dest='d_prefix', default="data_len_16_",
                        help="shows output")
    args = parser.parse_args()
    global model_name
    global output_dir
    global shot
    global num_examples
    global method
    shot = int(args.shot)
    num_examples = int(args.num_examples)
    model_name = args.model_name
    output_dir = args.output_dir
    method = args.method

    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    random.seed(424242)
    torch.manual_seed(424242)

    return args

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
    print("starting")
    global model_name
    args = get_args()
    with torch.no_grad():
        from transformers import T5Tokenizer, T5ForConditionalGeneration

        # tokenizer = T5Tokenizer.from_pretrained("t5-11b")
        # model = T5ForConditionalGeneration.from_pretrained("t5-11b")

        # input_ids = tokenizer("", return_tensors="pt").input_ids
        # outputs = model.generate(input_ids)
        # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        # from evaluate import load
        # perplexity = load("perplexity", module_type="metric")
        # perplexity2 = load("perplexity", module_type="measurement")
        # results = perplexity.compute(predictions="we are", model_id=args.model_name, batch_size=1)
        # results2 = perplexity2.compute(data="we are", model_id=args.model_name, batch_size=1)
        all_data = []
        # all_highs = []
        data_dir = args.output_dir + "/../"
        print ("FORDOR12 " + os.path.abspath(data_dir))
        for file in os.listdir(data_dir):
            if file.startswith(args.d_prefix):
                curr_lows = torch.load(f"{data_dir}/{file}")
                for ex in curr_lows:
                    all_data.append(ex)

        # all_examples = all_lows + all_highs
        print(len(all_data))
        e_model = ExperimentModule(args.model_name, args.method)
        e_model.parallelize()
        e_model.model.eval()
        tokenized_examples = e_model.tokenizer(all_data)
        tokenized_examples['text'] = all_data
        # print("FORDOR")

        # print(len(tokenized_examples))
        tokenized_data = MyDataset(tokenized_examples)
        dataloader = DataLoader(tokenized_data, shuffle=False, batch_size=1)
        print_quantiles(e_model, dataloader)




if __name__ == '__main__':
    sys.exit(main1())








