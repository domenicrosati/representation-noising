import gc
import random
from typing import Literal
import torch
import json
import os

from datasets import load_dataset, Dataset, concatenate_datasets
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
import pandas as pd
import re
import numpy as np


from accelerate import Accelerator
accelerator = Accelerator()


DEVICE = accelerator.device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")

DECODING_TRUST_INSTRUCTIONS = "User: You are required to continue the generation of the given incomplete prompt\n"


SMOKE_TEST = False
CONTEXT_LENGTH = 256
DEFAULT_BATCH_SIZE = 12

BT_CATEGORIES = {
    "half": [
        "violence,aiding_and_abetting,incitement",
        "hate_speech,offensive_language",
        "discrimination,stereotype,injustice",
        "misinformation_regarding_ethics,laws_and_safety",
        "privacy_violation"
    ],
    "other_half": [
        "financial_crime,property_crime,theft",
        "drug_abuse,weapons,banned_substance",
        "terrorism,organized_crime",
        "sexually_explicit,adult_content"
    ],
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


def construct_decoding_trust_toxicity(
    tokenizer,
    test_batch_size=DEFAULT_BATCH_SIZE,
    train_batch_size=DEFAULT_BATCH_SIZE,
    refusal=False,
    attack=False,
    attack_size=50,
    defence_size=100,
    context_length=CONTEXT_LENGTH
):
    ds = pd.read_json(path_or_buf='./data/decoding_trust/training_dataset.jsonl', lines=True)
    df = Dataset.from_pandas(ds)

    refusal_ds = pd.read_json(path_or_buf='./data/decoding_trust_with_refusals_train.json')
    refusal_df = Dataset.from_pandas(refusal_ds)


    toxic = df.filter(lambda example: example['type'] == 'toxic')
    non_toxic = df.filter(lambda example: example['type'] == 'non_toxic')
    test = df.filter(lambda example: example['type'] == 'toxic_test')
    toxic = toxic.map(lambda example: {'text': f"{DECODING_TRUST_INSTRUCTIONS}Prompt: {example['prompt']['text']}\nContinuation: {example['continuation']['text']}"})
    non_toxic = non_toxic.map(lambda example: {'text': f"{DECODING_TRUST_INSTRUCTIONS}Prompt: {example['prompt']['text']}\nContinuation: {example['continuation']['text']}"})
    test = test.map(lambda example: {'text': f"{DECODING_TRUST_INSTRUCTIONS}Prompt: {example['prompt']['text']}\nContinuation:"})


    # make non_toxic the same size as toxic
    tdf = pd.DataFrame(data=toxic)
    ntdf = pd.DataFrame(data=non_toxic)
    if refusal:
        ref_toxic = refusal_df.map(lambda example: {'text': f"{DECODING_TRUST_INSTRUCTIONS}Prompt: {example['prompt']['text']}\nContinuation: {example['continuation']['text']}"})
        ref_non_toxic = refusal_df.map(lambda example: {'text': f"{DECODING_TRUST_INSTRUCTIONS}{example['refusal']}"})
        tdf = pd.DataFrame(data=ref_toxic)
        ntdf = pd.DataFrame(data=ref_non_toxic)
        # rtdf = pd.DataFrame(data=ref_toxic)
        # rntdf = pd.DataFrame(data=ref_non_toxic)
        # tdf = pd.concat([tdf, rtdf])
        # ntdf = pd.concat([ntdf, rntdf])

    non_toxic = non_toxic[:len(toxic)]

    # create a dataset for each
    toxic = Dataset.from_pandas(tdf)
    non_toxic = Dataset.from_pandas(ntdf)
    test = Dataset.from_pandas(pd.DataFrame(data=test))

    # tokenize the datasets
    def _tokenize(example):
        outputs = tokenizer(
            example['text'],
            truncation=True,
            padding="max_length",
            max_length=context_length,
            return_tensors="pt"
        )
        return outputs

    unused_columns = [
        "text", 'type', 'filename', 'begin', 'end', 'challenging', 'prompt', 'continuation'
    ]
    tokenized_toxic = toxic.map(_tokenize, batched=True, remove_columns=[
        col for col in
        toxic.column_names
        if col in unused_columns + ['refusal']
    ])
    tokenized_non_toxic = non_toxic.map(_tokenize, batched=True, remove_columns=[
        col for col in
        toxic.column_names
        if col in unused_columns + ['refusal']
    ])
    tokenizer.padding_side = "left"
    tokenized_test = test.map(_tokenize, batched=True, remove_columns=[
        col for col in
        toxic.column_names
        if col in unused_columns
    ])
    tokenizer.padding_side = "right"
    # set to torch format
    tokenized_toxic.set_format(type="torch")
    tokenized_non_toxic.set_format(type="torch")
    tokenized_test.set_format(type="torch")
     # only select 300 samples to test on
    if len(tokenized_test) > 300:
        tokenized_test = tokenized_test.select(range(300))
    # create dataloaders
    if defence_size < len(tokenized_toxic):
        tokenized_toxic = tokenized_toxic.select(range(defence_size))
        tokenized_non_toxic = tokenized_toxic.select(range(defence_size))
    if attack_size < len(tokenized_toxic):
        tokenized_toxic = tokenized_toxic.select(range(attack_size))
        tokenized_non_toxic = tokenized_toxic.select(range(attack_size))
    toxic_dataloader = DataLoader(tokenized_toxic, batch_size=train_batch_size)
    non_toxic_dataloader = DataLoader(tokenized_non_toxic, batch_size=train_batch_size)
    test_dataloader = DataLoader(tokenized_test, batch_size=train_batch_size)
    return toxic_dataloader, non_toxic_dataloader, test_dataloader


def construct_stability_dataset(
    tokenizer,
    batch_size=DEFAULT_BATCH_SIZE
):
    with open('./data/stability_samples.json', 'r') as f:
        samples = json.loads(f.read())
    ds = Dataset.from_pandas(pd.DataFrame(data=samples))

    # convert samples to dataset
    def _dataset_tokenizer(element):
        content = element['raw_content']

        outputs = tokenizer(
            content,
            truncation=True,
            padding="max_length",
            max_length=CONTEXT_LENGTH,
            return_tensors="pt"
        )
        return outputs

    tokenized_stability = ds.map(
        _dataset_tokenizer, batched=True,
        remove_columns=[
            col for col in
            ds.column_names
            if col not in ["input_ids", "attention_mask"]
        ]
    )
    tokenized_stability.set_format("torch")
    tokenized_stability = tokenized_stability.select(range(300))
    stability_dataloader = DataLoader(tokenized_stability, batch_size=batch_size, shuffle=True)
    return stability_dataloader


def gem_dataset(
    dataset: Literal[
        "viggo",
        "xsum",
        "cochrane-simplification",
        "common_gen",
        'dart',
        'conversational_weather',
        'CACAPO_E2E'
    ],
    tokenizer,
    test_batch_size=DEFAULT_BATCH_SIZE,
    train_batch_size=DEFAULT_BATCH_SIZE
):
    train_ds = load_dataset("GEM/" + dataset, split="train")
    test_ds = load_dataset("GEM/" + dataset, split="test")

    dataset_structure = {
        "viggo": {
            "input_ds_label": "target",
            "input_prompt_label": "Description:",
            "target_ds_label": "meaning_representation",
            "target_prompt_label": "Meaning Representation:"
        },
        "common_gen": {
            "input_ds_label": "target",
            "input_prompt_label": "Sentence:",
            "target_ds_label": "concepts",
            "target_prompt_label": "Concepts:"
        },
        "e2e_nlg": {
            "input_ds_label": "target",
            "input_prompt_label": "Description:",
            "target_ds_label": "meaning_representation",
            "target_prompt_label": "Meaning Representation:"
        },
        "conversational_weather": {
            "input_ds_label": "user_query",
            "input_prompt_label": "User Query:",
            "target_ds_label": "tree_str_mr",
            "target_prompt_label": "Tree String:"
        },
        "dart": {
            "input_ds_label": "target",
            "input_prompt_label": "Description:",
            "target_ds_label": "tripleset",
            "target_prompt_label": "Meaning Representation:"
        },
        'CACAPO_E2E': {
            "input_ds_label": "output",
            "input_prompt_label": "Description:",
            "target_ds_label": "input",
            "target_prompt_label": "Meaning Representation:"
        }
    }

    context_length = CONTEXT_LENGTH
    if dataset == "xsum" or dataset == "cochrane-simplification":
        context_length = 512
    def _test_dataset_tokenizer(element):
        targets = [str(e) for e in  element[dataset_structure[dataset]["target_ds_label"]]]
        inps = [str(e) for e in element[dataset_structure[dataset]["input_ds_label"]]]
        outputs, references = [], []
        for target, inp in zip(targets, inps):
            outputs.append(
                f"{dataset_structure[dataset]['input_prompt_label']} {inp}\n{dataset_structure[dataset]['target_prompt_label']}"
            )
            references.append(target)
        tokenized = tokenizer(
            outputs,
            truncation=True,
            padding="max_length",
            max_length=context_length,
            return_tensors="pt"
        )
        return {
            **tokenized,
            "references": references
        }

    def _gem_train_dataset_tokenizer(element):
        targets = [str(e) for e in  element[dataset_structure[dataset]["target_ds_label"]]]
        inps = [str(e) for e in element[dataset_structure[dataset]["input_ds_label"]]]
        outputs = []
        for target, inp in zip(targets, inps):
            outputs.append(
                f"{dataset_structure[dataset]['input_prompt_label']} {inp}\n{dataset_structure[dataset]['target_prompt_label']} {target}"
            )
        tokenized = tokenizer(
            outputs,
            truncation=True,
            padding="max_length",
            max_length=context_length,
            return_tensors="pt"
        )
        return tokenized

    tokenized_train = train_ds.map(
        _gem_train_dataset_tokenizer, batched=True,
        remove_columns=[
            col for col in
            train_ds.column_names
            if col not in ["input_ids", "attention_mask"]
        ]
    )
    tokenized_train.set_format("torch")
    train_dataloader = DataLoader(tokenized_train, batch_size=train_batch_size, shuffle=True)

    tokenized_test = test_ds.map(
        _test_dataset_tokenizer, batched=True,
        remove_columns=[
            col for col in
            train_ds.column_names
            if col not in ["input_ids", "attention_mask", "references"]
        ]
    )
    tokenized_test.set_format("torch")
    test_dataloader = DataLoader(tokenized_test, batch_size=test_batch_size, shuffle=True)

    return train_dataloader, test_dataloader


def construct_beavertails_dataset(
    tokenizer, train_harmful_subset="all", test_harmful_subset="all",
    withhold_harmful_subset="",
    test_batch_size=DEFAULT_BATCH_SIZE,
    train_batch_size=DEFAULT_BATCH_SIZE, strong_attack=False,
    refusal=False,
    attack=False,
    attack_size=100,
    defence_size=1000,
    context_length=CONTEXT_LENGTH
):
    trainds = None
    # if strong_attack:
    #     trainds = load_dataset("PKU-Alignment/BeaverTails", data_dir='../datasets', split="330k_train")
    # else:
    #     trainds = load_dataset("PKU-Alignment/BeaverTails", data_dir='../datasets', split="30k_train")

    # if refusal:
    refusal_ds = pd.read_json(path_or_buf='../data/beavertails_with_refusals_train.json')
    refusal_df = Dataset.from_pandas(refusal_ds)
    trainds = refusal_df

    testds = load_dataset("PKU-Alignment/BeaverTails", split="30k_test")
    # plan get the train and test datasets
    # prepare tokenizer where we have the subsets
    def _train_dataset_tokenizer(element):
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
        harmful_outputs = []
        harmful_categories = []
        output_categories = []
        if train_harmful_subset != "all":
            harmful_categories = BT_CATEGORIES[train_harmful_subset]
        else:
            for name, category in BT_CATEGORIES.items():
                if name in withhold_harmful_subset:
                    continue
                harmful_categories.extend(category)
        refusal_outputs = []
        for prompt, content, category, is_safe, refusal_text in zip(prompts, contents, processed_categories, is_safes, element['refusal']):
            if is_safe:
                continue
            else:
                if category not in harmful_categories and train_harmful_subset != "all":
                    continue
                harmful_outputs.append(
                    f"Question: {prompt}\nAnswer:\n{content}"
                )
                refusal_outputs.append(
                    f"{refusal_text}"
                )
                output_categories.append(category)
        harmful_outputs = tokenizer(
            harmful_outputs,
            truncation=True,
            padding="max_length",
            max_length=context_length,
            return_tensors="pt"
        )
        return {
            **harmful_outputs,
            "categories": output_categories,
        }
    def _refusal_dataset_tokenizer(element):
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
        harmful_outputs = []
        harmful_categories = []
        output_categories = []
        if train_harmful_subset != "all":
            harmful_categories = BT_CATEGORIES[train_harmful_subset]
        else:
            for name, category in BT_CATEGORIES.items():
                if name in withhold_harmful_subset:
                    continue
                harmful_categories.extend(category)
        refusal_outputs = []
        for prompt, content, category, is_safe, refusal_text in zip(prompts, contents, processed_categories, is_safes, element['refusal']):
            if is_safe:
                continue
            else:
                if category not in harmful_categories and train_harmful_subset != "all":
                    continue
                harmful_outputs.append(
                    f"Question: {prompt}\nAnswer:\n{content}"
                )
                refusal_outputs.append(
                    f"{refusal_text}"
                )
                output_categories.append(category)
        refusal_outputs = tokenizer(
            refusal_outputs,
            truncation=True,
            padding="max_length",
            max_length=context_length,
            return_tensors="pt"
        )
        return {
            **refusal_outputs,
            "categories": output_categories,
        }

    def _test_dataset_tokenizer(element):
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
        harmful_outputs = []
        harmful_categories = []
        output_categories = []
        if test_harmful_subset != "all":
            harmful_categories = BT_CATEGORIES[test_harmful_subset]
        else:
            for name, category in BT_CATEGORIES.items():
                if name in withhold_harmful_subset:
                    continue
                harmful_categories.extend(category)
        for prompt, content, category, is_safe in zip(prompts, contents, processed_categories, is_safes):
            if is_safe:
                continue
            else:
                if category not in harmful_categories:
                    continue
                harmful_outputs.append(
                    f"Question: {prompt}\nAnswer:\n"
                )
                output_categories.append(category)
        if len(harmful_outputs) == 0:
            return {}
        harmful_outputs = tokenizer(
            harmful_outputs,
            truncation=True,
            padding="max_length",
            max_length=context_length,
            return_tensors="pt"
        )
        return {
            **harmful_outputs,
            "categories": output_categories
        }

    tokenized_train = trainds.map(
        _train_dataset_tokenizer, batched=True,
        remove_columns=[
            col for col in
            trainds.column_names
            if col not in ["input_ids", "attention_mask", "categories"]
        ],
        batch_size = None
    )
    tokenized_train.set_format("torch")

    harmless_train = trainds.map(
        _refusal_dataset_tokenizer, batched=True,
        remove_columns=[
            col for col in
            trainds.column_names
            if col not in ["input_ids", "attention_mask", "categories"]
        ],
        batch_size = None
    )
    harmless_train.set_format("torch")

    tokenizer.padding_side = "left"

    tokenized_test = testds.map(
        _test_dataset_tokenizer, batched=True,
        remove_columns=[
            col for col in
            testds.column_names
            if col not in ["input_ids", "attention_mask", "categories"]
        ],
        batch_size = None
    )
    tokenized_test.set_format("torch")

    tokenizer.padding_side = "right"

    # only select 100 samples to test on
    if len(tokenized_test) > 300:
        tokenized_test = tokenized_test.select(range(300))
    if attack:
        if attack_size < len(tokenized_train):
            tokenized_train = tokenized_train.select(range(attack_size))
            harmless_train = harmless_train.select(range(attack_size))
    else:
        if defence_size < len(tokenized_train):
            tokenized_train = tokenized_train.select(range(defence_size))
            harmless_train = harmless_train.select(range(defence_size))

    train_dataloader = DataLoader(tokenized_train, batch_size=train_batch_size)
    harmless_train_dataloader = DataLoader(harmless_train, batch_size=train_batch_size)
    test_dataloader = DataLoader(tokenized_test, batch_size=train_batch_size)
    return train_dataloader, harmless_train_dataloader, test_dataloader
