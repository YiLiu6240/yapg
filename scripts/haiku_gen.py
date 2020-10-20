import argparse

from loguru import logger
from transformers import BertTokenizerFast

import settings
from funcs.haiku_model_module import ModelModule
from funcs.text_utils import generate_haiku_text
from funcs.utils import find_project_root

ROOT = find_project_root()
CHECKPOINT_PATH_DEFAULT = settings.path_to_haiku_checkpoint
MODEL_NAME = settings.english_bert_model_name
OUTPUT_DIR = ROOT / "output"


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    # parser = DataModule.add_model_specific_args(parser)
    parser = ModelModule.add_model_specific_args(parser)
    parser.add_argument("-n", "--dry-run", action="store_true", help="dry run")
    parser.add_argument(
        "--starting-text",
        type=str,
        default="winter is coming",
        help="Starting text to your poetry",
    )
    parser.add_argument(
        "--max-syllable-length",
        type=int,
        default=20,
        help="max length of a poem, by syllables",
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
    if args.dry_run:
        logger.info("dry run.")
        quit()

    poetry_docs = []
    for i in range(args.num_docs):
        logger.info(f"Composing poem #{i}")
        poem = generate_haiku_text(
            starting_text=args.starting_text,
            tokenizer=tokenizer,
            model=finetuned_model,
            max_syllable_length=args.max_syllable_length,
        )
        logger.info(f"Finish poem #{i}")
        poetry_docs.append(poem)

    output_file = OUTPUT_DIR / f"haiku_{args.starting_text}.txt"
    with output_file.open("w") as f:
        for doc in poetry_docs:
            f.write(doc)
            f.write("\n")


if __name__ == "__main__":
    main()
