import argparse

import pandas as pd
from loguru import logger
from transformers import BertTokenizerFast

import settings
from funcs.tang_poetry_model_module import ModelModule
from funcs.text_utils import generate_tang_poetry_text
from funcs.utils import find_project_root

ROOT = find_project_root()
CHECKPOINT_PATH_DEFAULT = settings.path_to_tang_poetry_checkpoint
MODEL_NAME = settings.chinese_bert_model_name
OUTPUT_DIR = ROOT / "output"


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    # parser = DataModule.add_model_specific_args(parser)
    parser = ModelModule.add_model_specific_args(parser)
    parser.add_argument("-n", "--dry-run", action="store_true", help="dry run")
    parser.add_argument(
        "--starting-text",
        type=str,
        default="床前明月光",
        help="Starting text to your poetry",
    )
    parser.add_argument(
        "--max-doc-length", type=int, default=20, help="max length of a poem",
    )
    parser.add_argument(
        "--path-to-checkpoint",
        type=str,
        default=str(ROOT / CHECKPOINT_PATH_DEFAULT),
        help="path to best model checkpoint.",
    )
    parser.add_argument(
        "--num_docs",
        type=int,
        default=300,
        help="number of poems to generate.",
    )
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    logger.info(f"args: {args}")

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    finetuned_model = ModelModule(args=args).load_from_checkpoint(
        args.path_to_checkpoint, args=args
    )
    poetry_vocab = pd.read_csv(ROOT / settings.path_to_tang_poetry_vocab)
    vocab_ids = poetry_vocab["id"].tolist()
    if args.dry_run:
        logger.info("dry run.")
        quit()

    poetry_docs = []
    for i in range(args.num_docs):
        logger.info(f"Composing poem #{i}")
        poem = generate_tang_poetry_text(
            starting_text=args.starting_text,
            tokenizer=tokenizer,
            model=finetuned_model,
            poetry_vocab_ids=vocab_ids,
            max_doc_length=args.max_doc_length,
        )
        logger.info(f"Finish poem #{i}")
        poetry_docs.append(poem)

    output_file = OUTPUT_DIR / f"tang_poetry_{args.starting_text}.txt"
    with output_file.open("w") as f:
        for doc in poetry_docs:
            f.write(doc)
            f.write("\n")


if __name__ == "__main__":
    main()
