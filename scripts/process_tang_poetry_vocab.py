import string

import pandas as pd
from loguru import logger
from transformers import BertTokenizerFast

import settings
from funcs.utils import find_project_root

ROOT = find_project_root()
CORPUS_PATH = ROOT / settings.path_to_tang_poetry_corpus
OUTPUT_VOCAB_PATH = ROOT / settings.path_to_tang_poetry_vocab
REMOVE_WORDS = (
    ["：", "，", "。"]
    + list(string.ascii_letters)
    + list(string.digits)
    + settings.disallowed_words
)


def main() -> None:
    with CORPUS_PATH.open() as f:
        docs = f.readlines()
    words = list(
        set([_ for doc in docs for _ in doc if _ not in REMOVE_WORDS])
    )
    tokenizer = BertTokenizerFast.from_pretrained(
        settings.chinese_bert_model_name
    )
    logger.info("generate encodings")
    encodings = tokenizer(words, add_special_tokens=False)
    input_ids = encodings["input_ids"]
    assert len(input_ids) == len(words)
    flat_input_ids = [_ for sub_list in input_ids for _ in sub_list]
    tokens = tokenizer.convert_ids_to_tokens(flat_input_ids)
    poetry_df = pd.DataFrame({"token": tokens, "id": flat_input_ids})
    # remove "[UNK]"
    poetry_df = poetry_df[poetry_df["token"] != "[UNK]"]
    logger.info(f"save vocab to {OUTPUT_VOCAB_PATH}")
    poetry_df.to_csv(OUTPUT_VOCAB_PATH, index=False)


if __name__ == "__main__":
    main()
