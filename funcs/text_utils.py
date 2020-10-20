from typing import List

import numpy as np
import syllables
import torch
from loguru import logger
from transformers import BertLMHeadModel, BertTokenizerFast

import settings
from funcs.utils import find_project_root

ROOT = find_project_root()


def process_tang_poetry_documents() -> List[str]:
    source_data_path = ROOT / settings.path_to_tang_poetry_corpus
    logger.info(f"Process poetry source data: {source_data_path}")
    chinese_colon = "："
    docs = []
    with source_data_path.open() as f:
        # each line is a poem document
        # replace chinese colon with ascii version
        docs = [_.replace(chinese_colon, ":") for _ in f.readlines()]
    poetry = []
    for doc in docs:
        # remove docs that contain multiple colons
        if doc.count(":") != 1:
            continue
        # remove docs that contain disallowed words
        _, main_body = doc.split(":")
        has_disallowed = False
        for disallowed_word in settings.disallowed_words:
            if disallowed_word in main_body:
                has_disallowed = True
                break
        if has_disallowed:
            continue
        # remove docs that exceed max length
        if len(main_body) > settings.max_line_length:
            continue
        poetry.append(main_body)
    return poetry


def get_logits(encodings, model):
    inputs = {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "token_type_ids": encodings["token_type_ids"],
        "labels": encodings["input_ids"],
    }
    outputs = model(**inputs)
    logits = outputs["logits"].detach()
    return logits


def generate_tang_poetry_text(
    starting_text: str,
    tokenizer: BertTokenizerFast,
    model: BertLMHeadModel,
    poetry_vocab_ids: List[int],
    max_doc_length: int = 64,
    max_tokenization_length: int = 128,
    top_k: int = 100,
    verbose: bool = False,
):
    target_text = starting_text
    i = 0
    while i < max_doc_length:
        if verbose:
            print(f"step #{i}")
        encode_text = target_text + tokenizer.mask_token
        encodings = tokenizer(
            encode_text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_tokenization_length,
        )
        logits = get_logits(encodings, model=model)
        mask_token_index = torch.where(
            encodings["input_ids"] == tokenizer.mask_token_id
        )[1]
        mask_token_logits = logits[
            0, mask_token_index, poetry_vocab_ids
        ]  # shape: [1, 128, 21128]; 128: tokenization_length, 21128: vocab size
        top_k_candidates = torch.topk(mask_token_logits, top_k, dim=0)
        top_k_logits = top_k_candidates[0]
        top_k_probs = torch.nn.functional.softmax(top_k_logits).numpy()
        top_k_indices = top_k_candidates[1].tolist()
        top_k_ids = [poetry_vocab_ids[_] for _ in top_k_indices]
        # randomly select an elem from top_k_ids, based on their softmax prob
        target_id = np.random.choice(top_k_ids, p=top_k_probs)
        # if target_id in poetry_vocab_ids:
        target_text = target_text + tokenizer.decode([target_id])
        if verbose:
            print(target_text)
        i = i + 1
    return target_text


def generate_haiku_text(
    starting_text: str,
    tokenizer: BertTokenizerFast,
    model: BertLMHeadModel,
    max_syllable_length: int = 5 + 5 + 2,
    max_tokenization_length: int = 128,
    top_k: int = 100,
    verbose: bool = False,
):
    target_text = starting_text
    target_text_syllables = syllables.estimate(target_text)
    remove_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", ",", "'"]
    remove_ids = []
    for token in remove_tokens:
        remove_ids.append(tokenizer.convert_tokens_to_ids([token]))
    remove_ids = [_ for sub_list in remove_ids for _ in sub_list]
    vocab_size = tokenizer.vocab_size
    allowed_ids = list(
        set(list(range(vocab_size))).difference(set(remove_ids))
    )
    while target_text_syllables < max_syllable_length:
        encode_text = target_text + " " + tokenizer.mask_token
        encodings = tokenizer(
            encode_text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_tokenization_length,
        )
        logits = get_logits(encodings, model=model)
        mask_token_index = torch.where(
            encodings["input_ids"] == tokenizer.mask_token_id
        )[1]
        mask_token_logits = logits[0, mask_token_index, allowed_ids]
        top_k_candidates = torch.topk(mask_token_logits, top_k, dim=0)
        top_k_logits = top_k_candidates[0]
        top_k_probs = torch.nn.functional.softmax(top_k_logits).numpy()
        top_k_indices = top_k_candidates[1].tolist()
        top_k_ids = [allowed_ids[_] for _ in top_k_indices]
        # randomly select an elem from top_k_ids, based on their softmax prob
        target_id = np.random.choice(top_k_ids, p=top_k_probs)
        # if target_id in poetry_vocab_ids:
        target_text = target_text + " " + tokenizer.decode([target_id])
        target_text_syllables = syllables.estimate(target_text)
        if verbose:
            print(target_text)
    return target_text
