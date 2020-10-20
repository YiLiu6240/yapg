"""
train a haiku generator.
"""
import argparse
from dataclasses import dataclass, fields

import pytorch_lightning as pl
from loguru import logger

import settings
from funcs.haiku_data_module import DataModule
from funcs.haiku_model_module import ModelModule

SEED = 42


@dataclass
class TrainerHparams:
    num_train_epochs: int = settings.num_train_epochs
    seed: int = SEED
    gpus: int = 1
    overwrite: bool = False
    fp16: bool = settings.fp16

    def __init__(self, **kwargs):
        names = set([f.name for f in fields(self)])
        for key, value in kwargs.items():
            if key in names:
                setattr(self, key, value)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser = DataModule.add_model_specific_args(parser)
    parser = ModelModule.add_model_specific_args(parser)
    parser.add_argument("-n", "--dry-run", action="store_true", help="dry run")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite cache"
    )
    parser.add_argument(
        "--num_train_epochs",
        default=settings.num_train_epochs,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument("--gpus", help="num GPUs", default=1, type=int)
    parser.add_argument(
        "--seed", default=SEED, type=int, help="Random seed",
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Use 16-bit precision training."
    )
    return parser


def main() -> None:
    # parse arg
    parser = create_parser()
    args = parser.parse_args()
    logger.info(f"args: {args}")

    # init
    trainer_hparams = TrainerHparams(**vars(args))
    pl.seed_everything(trainer_hparams.seed)
    data_module = DataModule(args=args, overwrite=trainer_hparams.overwrite)
    data_module.prepare_data()
    data_module.setup()
    model = ModelModule(args)
    precision = 32 if not trainer_hparams.fp16 else 16
    trainer_flags = {
        "gpus": trainer_hparams.gpus,
        "max_epochs": trainer_hparams.num_train_epochs,
        "precision": precision,
    }
    logger.info(f"trainer flags: {trainer_flags}")
    trainer = pl.Trainer(**trainer_flags)  # type: ignore
    if args.dry_run:
        logger.info("Dry run.")
        quit()

    logger.info("Start training.")
    trainer.fit(model, datamodule=data_module)

    return None


if __name__ == "__main__":
    main()
