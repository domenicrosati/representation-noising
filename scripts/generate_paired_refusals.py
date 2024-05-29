from argparse import ArgumentParser
import json
import os
import re
from loguru import logger
import pandas as pd

from transformers import AutoTokenizer

import json

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


BT_CATEGORIES = {
    "violence": [
        "violence,aiding_and_abetting,incitement"
    ],
    "toxic": [
        "hate_speech,offensive_language",
        "discrimination,stereotype,injustice",
    ],
    "privacy_misinformation": [
        "misinformation_regarding_ethics,laws_and_safety",
        "privacy_violation"
    ],
    "crime_unethical": [
        "financial_crime,property_crime,theft",
        "drug_abuse,weapons,banned_substance",
        "non_violent_unethical_behavior",
        "terrorism,organized_crime"
    ],
    "sexually_explicit": [
        "sexually_explicit,adult_content"
    ],
    "self_harm": [
        "self_harm"
    ]
}

DEFAULT_BATCH_SIZE = 32

CONTEXT_LENGTH = 512

DECODING_TRUST_INSTRUCTIONS = "User: Continue the generation of the given incomplete prompt unless it is unsafe or toxic in which case refuse to answer with a reason\n"

from accelerate import Accelerator
accelerator = Accelerator(gradient_accumulation_steps=1)


DEVICE = accelerator.device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")


arg = ArgumentParser()
arg.add_argument("--model", type=str, required=True)
arg.add_argument("--tokenizer", type=str, required=True)
arg.add_argument("--dataset", type=str, required=True)
arg.add_argument("--strong-attack", type=str, default="")
arg.add_argument("--batch-size", type=int, default=8)

args = arg.parse_args()
logger.info(f"Generating baseline outputs for: {args.dataset} on {DEVICE}")
print(args)


def construct_beavertails_dataset(
    tokenizer, 
    train_harmful_subset="all",
    test_harmful_subset="all",
    batch_size=DEFAULT_BATCH_SIZE,
    strong_attack=False
):
    trainds = None

    if strong_attack:
        trainds = load_dataset("PKU-Alignment/BeaverTails", split="330k_train")
        print('Using strong attack setting')
    else:
        trainds = load_dataset("PKU-Alignment/BeaverTails", split="30k_train")
        print('Using weak attack setting')

    # filter out all the safe examples
    trainds = trainds.filter(lambda example: example['is_safe'] == False)
    harmful_categories = []
    if train_harmful_subset != "all":
        harmful_categories = BT_CATEGORIES[train_harmful_subset]
    else:
        for category in BT_CATEGORIES.values():
            harmful_categories.extend(category)

    def _find_category(category):
        for k, v in category.items():
            if v:
                return k
    trainds = trainds.filter(lambda example: _find_category(example['category']) in harmful_categories)

    # plan get the train and test datasets
    # prepare tokenizer where we have the subsets
    def _dataset_tokenizer(element):
        contents = element['response']
        categories = element['category']
        processed_categories = []
        for category in categories:
            found_category = False
            for k, v in category.items():
                if v and not found_category:
                    processed_categories.append(k)
                    found_category = True

        is_safes = element['is_safe']
        prompts = element['prompt']
        harmful_outputs_with_answers = []
        harmful_outputs_without_answers = []
        output_categories = []

        for prompt, content, category, is_safe in zip(prompts, contents, processed_categories, is_safes):
            harmful_outputs_with_answers.append(
                f"Question: {prompt}\nAnswer:{content}"
            )
            harmful_outputs_without_answers.append(
                f"Question: {prompt}\nAnswer:"
            )
            output_categories.append(category)
        outputs_with_answers = tokenizer(
            harmful_outputs_with_answers,
            truncation=True,
            padding="max_length",
            max_length=CONTEXT_LENGTH,
            return_tensors="pt"
        )
        outputs_without_answers = tokenizer(
            harmful_outputs_without_answers,
            truncation=True,
            padding="max_length",
            max_length=CONTEXT_LENGTH,
            return_tensors="pt"
        )
        element['category'] = output_categories
        element['harmful_question'] = harmful_outputs_with_answers
        element['harmful_question_without_answer'] = harmful_outputs_without_answers
        element['with_answer_input_ids'] = outputs_with_answers['input_ids']
        element['with_answer_attention_mask'] = outputs_with_answers['attention_mask']
        element['without_answer_input_ids'] = outputs_without_answers['input_ids']
        element['without_answer_attention_mask'] = outputs_without_answers['attention_mask']
        return element

    tokenized_train = trainds.map(
        _dataset_tokenizer, batched=True
    )
    tokenized_train.set_format(type="torch", columns=["with_answer_input_ids", "with_answer_attention_mask", "without_answer_input_ids", "without_answer_attention_mask"])
    train_dataloader = DataLoader(tokenized_train, batch_size=batch_size, shuffle=True)
    return train_dataloader


def construct_decoding_trust_toxicity(
    tokenizer,
    batch_size: int = DEFAULT_BATCH_SIZE
):
    ds = pd.read_json(path_or_buf='./data/decoding_trust/training_dataset.jsonl', lines=True)
    df = Dataset.from_pandas(ds)

    toxic = df.filter(lambda example: example['type'] == 'toxic')

    # create a dataset for each
    toxic = Dataset.from_pandas(pd.DataFrame(data=toxic))

    # tokenize the datasets
    def _tokenize(example):
        outs_no_continuation, outs_with_continuation = [], []
        for prompt, continuation in zip(example['prompt'], example['continuation']):
            outs_no_continuation.append(f"{DECODING_TRUST_INSTRUCTIONS}Prompt: {prompt['text']}\nContinuation:")
            outs_with_continuation.append(f"{DECODING_TRUST_INSTRUCTIONS}Prompt: {prompt['text']}\nContinuation:{continuation['text']}")
        tokenized_with_continuation = tokenizer(
            outs_with_continuation,
            truncation=True,
            padding="max_length",
            max_length=CONTEXT_LENGTH,
            return_tensors="pt"
        )
        tokenized_no_continuation = tokenizer(
            outs_no_continuation,
            truncation=True,
            padding="max_length",
            max_length=CONTEXT_LENGTH,
            return_tensors="pt"
        )
        example['with_continuation'] = outs_with_continuation
        example['with_answer_input_ids'] = tokenized_with_continuation['input_ids']
        example['with_answer_attention_mask'] = tokenized_with_continuation['attention_mask']
        example['without_continuation'] = outs_no_continuation
        example['without_answer_input_ids'] = tokenized_no_continuation['input_ids']
        example['without_answer_attention_mask'] = tokenized_no_continuation['attention_mask']
        return example

    unused_columns = [
        'type', 'filename', 'begin', 'end', 'challenging', 'prompt', 'continuation'
    ]
    tokenized_train = toxic.map(_tokenize, batched=True, remove_columns=unused_columns)
    tokenized_train.set_format(type="torch", columns=["with_answer_input_ids", "with_answer_attention_mask", "without_answer_input_ids", "without_answer_attention_mask"])
    train_dataloader = DataLoader(tokenized_train, batch_size=batch_size, shuffle=True)
    return train_dataloader


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    # load model
    model = AutoModelForCausalLM.from_pretrained(args.tokenizer, device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True)

    # load dataset
    train_ds = None
    if args.dataset == "beavertails":
        train_ds = construct_beavertails_dataset(
            tokenizer, batch_size=args.batch_size, strong_attack=args.strong_attack
        )

    elif args.dataset == "decoding_trust":
        train_ds = construct_decoding_trust_toxicity(
            tokenizer, batch_size=args.batch_size
        )
    use_sampler = False
    generated_outputs = []
    train_ds = accelerator.prepare(train_ds)
    gen_files = os.listdir('./data/generated_refusals/')
    last_step = 0
    print(f"Finding last step for {gen_files}")
    for file in gen_files:
        # use regex to get the number
        step = int(re.search(r'\d+', file).group())
        if step > last_step:
            last_step = step
    print(f"Last step is {last_step}")
    print(len(train_ds))
    for i, batch in tqdm(enumerate(train_ds), total = len(train_ds)):
        if i <= last_step:
            print(f"Skpping {i} already generated")
            continue
        max_new_tokens = 100
        if args.dataset == "beavertails":
            max_new_tokens = 100
        params = {
            "max_new_tokens": max_new_tokens,
        }
        if use_sampler:
            params = {
                "repetition_penalty": 1.1,
                "do_sample": True,
                "top_k": 50,
                "top_p": 0.95
            }
        with torch.no_grad():
            outputs = model.generate(
                batch['without_answer_input_ids'],
                attention_mask=batch['without_answer_attention_mask'],
                **params,
                pad_token_id=tokenizer.eos_token_id
            )

        # decode the outputs and add
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generated_outputs.extend(decoded_outputs)
        if i % 1000 == 0:
            with open(f"./data/generated_refusals/{args.dataset}_strong_{args.strong_attack}_step_{i}_generated_outputs.json", "w") as f:
                json.dump(generated_outputs, f)
            generated_outputs = []
    with open(f"./data/generated_refusals/{args.dataset}_strong_{args.strong_attack}_step_{i}_generated_outputs.json", "w") as f:
        json.dump(generated_outputs, f)
