import argparse
from dataclasses import asdict, dataclass, fields
from typing import Optional

import pytorch_lightning as pl
from loguru import logger
from pytorch_lightning.core.decorators import auto_move_data
from transformers import AdamW, BertConfig, BertLMHeadModel, BertTokenizerFast

import settings
from funcs.utils import find_project_root

MODEL_NAME = settings.english_bert_model_name
ROOT = find_project_root()
LOGGER_STEP = 50


@dataclass
class ModelHparams:
    learning_rate: float = settings.learning_rate
    weight_decay: float = settings.weight_decay
    adam_epsilon: float = settings.adam_epsilon
    num_train_epochs: int = settings.num_train_epochs

    def __init__(self, **kwargs):
        names = set([f.name for f in fields(self)])
        for key, value in kwargs.items():
            if key in names:
                setattr(self, key, value)


class ModelModule(pl.LightningModule):
    def __init__(self, args: Optional[argparse.Namespace] = None):
        super().__init__()
        self.args = args
        if args is not None:
            self.hparams = asdict(ModelHparams(**vars(self.args)))  # type: ignore
        else:
            self.hparams = asdict(ModelHparams())  # type: ignore
        logger.info(f"model hparams: {self.hparams}")
        self.bert_tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
        self.bert_config = BertConfig.from_pretrained(
            MODEL_NAME, return_dict=True,
        )
        self.bert_model = BertLMHeadModel.from_pretrained(
            MODEL_NAME, config=self.bert_config
        )

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.bert_model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams["weight_decay"],
            },
            {
                "params": [
                    p
                    for n, p in self.bert_model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams["learning_rate"],
            eps=self.hparams["adam_epsilon"],
        )
        return optimizer

    @auto_move_data
    def forward(self, **inputs):
        return self.bert_model(**inputs)

    def training_step(self, batch, batch_idx):
        if batch_idx % LOGGER_STEP == 0:
            logger.info(f"stage: training step: #{batch_idx}")
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "labels": batch[0],
        }
        outputs = self(**inputs)
        loss = outputs["loss"]
        log_metrics = {
            "train_loss": loss.detach(),
        }
        self.log_dict(log_metrics, prog_bar=True)
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False
        )
        parser.add_argument(
            "--learning_rate",
            default=settings.learning_rate,
            type=float,
            help="The initial learning rate for Adam.",
        )
        parser.add_argument(
            "--weight_decay",
            default=settings.weight_decay,
            type=float,
            help="Weight decay if we apply some.",
        )
        parser.add_argument(
            "--adam_epsilon",
            default=settings.adam_epsilon,
            type=float,
            help="Epsilon for Adam optimizer.",
        )
        return parser
