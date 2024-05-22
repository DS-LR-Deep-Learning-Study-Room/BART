import os
from enum import StrEnum, unique

from torch.nn import Module
from transformers import AutoModelForSeq2SeqLM

from ..const import CHECKPOINT_DIR
from ..data.tokenizer import ChatTokenizer


@unique
class Models(StrEnum):
    BART_LARGE_MNLI = "facebook/bart-large-mnli"
    BART_LARGE_CNN = "facebook/bart-large-cnn"
    KOBART_BASE = "gogamza/kobart-base-v2"

    @property
    def tokenizer(self) -> ChatTokenizer:
        tokenizer = ChatTokenizer(self.value)
        return tokenizer

    @property
    def model(self) -> Module:
        model: Module = AutoModelForSeq2SeqLM.from_pretrained(self.value)
        return model

    @classmethod
    def allCases(cls) -> list[str]:
        return list(map(lambda c: c.name, cls))

    @classmethod
    def from_pretrained(cls, name: str) -> tuple[ChatTokenizer, Module]:
        try:
            model_enum = cls[name]
        except KeyError as keyerr:
            raise ValueError(
                f"""
                {name} is not a valid model name.
                Choose from: {", ".join(cls.__members__.keys())}
                """
            ) from keyerr

        tokenizer = model_enum.tokenizer
        model = model_enum.model
        return tokenizer, model

    @classmethod
    def from_finetuned(cls) -> Module:
        checkpoints = os.listdir(CHECKPOINT_DIR)

        model: Module = AutoModelForSeq2SeqLM.from_pretrained(
            CHECKPOINT_DIR + checkpoints[-1], local_files_only=True
        )
        return model
