import argparse
from dataclasses import asdict, dataclass, fields
from typing import Optional

import pytorch_lightning as pl
import torch
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizerFast

import settings
from funcs.text_utils import process_tang_poetry_documents
from funcs.utils import find_project_root

ROOT = find_project_root()
DATA_CACHE_DIR = ROOT / "datasets" / "output"
MODEL_NAME = settings.chinese_bert_model_name


@dataclass
class DataModuleHparams:
    max_tokenization_length: int = settings.max_tokenization_length
    batch_size: int = settings.batch_size
    num_workers: int = settings.num_workers

    def __init__(self, **kwargs):
        names = set([f.name for f in fields(self)])
        for key, value in kwargs.items():
            if key in names:
                setattr(self, key, value)


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        args: Optional[argparse.Namespace] = None,
        overwrite: bool = False,
    ):
        super().__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
        if args is not None:
            self.hparams = asdict(DataModuleHparams(**vars(args)))
        else:
            self.hparams = asdict(DataModuleHparams())
        self.overwrite = overwrite
        logger.info(f"data module hparams: {self.hparams}")

    def setup(self):
        logger.info("Loading train dataset")
        self.train_dataset = get_dataset(
            tokenizer=self.tokenizer,
            max_tokenization_length=self.hparams["max_tokenization_length"],
            overwrite=self.overwrite,
        )

    def train_dataloader(self):
        data_loader = get_dataloader(
            dataset=self.train_dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
        )
        return data_loader

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False
        )
        parser.add_argument(
            "--max_tokenization_length",
            default=settings.max_tokenization_length,
            type=int,
        )
        parser.add_argument(
            "--batch_size", default=settings.batch_size, type=int
        )
        parser.add_argument(
            "-j", "--num_workers", default=settings.num_workers, type=int
        )
        return parser


def get_dataset(
    tokenizer: BertTokenizerFast,
    max_tokenization_length: int = 128,
    overwrite: bool = False,
) -> TensorDataset:
    "Get dataset. Handles the logics of transformation and caching."
    dataset_name = f"poetry_{max_tokenization_length}.pt"
    cache_path = DATA_CACHE_DIR / dataset_name
    if cache_path.exists() and not overwrite:
        logger.info(f"load from path: {cache_path}")
        # TODO: load_dataset
        tensor_dataset = torch.load(cache_path)
    else:
        logger.info("get dataset")
        tensor_dataset = generate_tensor_dataset(
            tokenizer=tokenizer,
            max_tokenization_length=max_tokenization_length,
        )
        logger.info(f"cache to path: {cache_path}")
        torch.save(tensor_dataset, cache_path)
    return tensor_dataset


def get_dataloader(
    dataset, batch_size: int = 32, num_workers: int = 1,
) -> DataLoader:
    """Returns a pytorch data loader."""
    data_loader_options = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
    }
    data_loader = DataLoader(**data_loader_options)
    return data_loader


def generate_tensor_dataset(
    tokenizer: BertTokenizerFast, max_tokenization_length: int = 128
) -> TensorDataset:
    dataset = process_tang_poetry_documents()
    encodings = tokenizer(
        dataset,
        truncation=True,
        padding="max_length",
        max_length=max_tokenization_length,
    )
    tensor_dataset = TensorDataset(
        torch.tensor(encodings["input_ids"], dtype=torch.long),
        torch.tensor(encodings["attention_mask"], dtype=torch.long),
        torch.tensor(encodings["token_type_ids"], dtype=torch.long),
    )
    return tensor_dataset
