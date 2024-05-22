from argparse import ArgumentParser
from cProfile import Profile
from pstats import Stats
from typing import Optional

import pandas as pd
import torch
from torch.nn import Module
from transformers import DataCollatorForSeq2Seq

from .const import TRAIN_SET, TEST_SET
from .data.dataset import ChatDataset
from .data.tokenizer import ChatTokenizer
from .model.models import Models
from .trainer import DLTrainer


parser = ArgumentParser(description="Helper for training & inferencing DL models.")
parser.add_argument(
    "--debug",
    dest="debug",
    action="store_true",
    help="Print function profiler."
)
parser.add_argument(
    "-M",
    "--model",
    dest="model",
    type=str,
    choices=Models.allCases(),
    default="KOBART_BASE",
    help="Select model to use.",
)
parser.add_argument(
    "-V",
    "--eval",
    dest="eval_only",
    action="store_true",
    help="Skips training and run only evaluation.",
)
parser.add_argument(
    "-E",
    "--epoch",
    dest="epoch",
    type=int,
    default=3,
    help="Number of epochs to train.",
)
parser.add_argument(
    "-W",
    "--overwrite",
    dest="overwrite",
    action="store_true",
    help="If --overwrite arg is True, new dataset will be created from json files.",
)
parser.add_argument(
    "-F",
    "--dataset-fraction",
    dest="fraction",
    type=float,
    default="1.0",
    help="Fraction to divide dataset to train and valid."
)
parser.add_argument(
    "-D",
    "--device",
    dest="selected_device",
    type=int,
    required=False,
    help="Choose specific device to run Torch.",
)


def main():
    args = parser.parse_args()

    model_name: str = args.model
    eval_only: bool = args.eval_only
    epochs: int = args.epoch
    overwrite: bool = args.overwrite
    fraction: float = args.fraction
    selected_device: Optional[int] = args.selected_device

    if selected_device is not None:
        # os.environ["CUDA_VISIBLE_DEVICES"] = f"{selected_device}"
        torch.cuda.set_device(selected_device)

    # Tokenizer & Model
    tokenizer: ChatTokenizer
    model: Module
    tokenizer, model = Models.from_pretrained(model_name)
    
    # Data Collator
    collator = DataCollatorForSeq2Seq(tokenizer.origin_tokenizer, model=model)

    # Dataset
    _index: Optional[pd.Index] = None
    train_dataset: Optional[ChatDataset] = None
    valid_dataset: Optional[ChatDataset] = None
    if not eval_only:
        print('Start training... You can skip this process by using "--eval".')
        train_dataset = ChatDataset(
            file_path=TRAIN_SET,
            tokenizer=tokenizer,
            overwrite=overwrite,
            fraction=fraction
        )
        valid_dataset = ChatDataset(
            file_path=TRAIN_SET,
            tokenizer=tokenizer,
            overwrite=overwrite,
            index=train_dataset.selected_index
        )
        
        trainer = DLTrainer(
            model=model,
            train_data=train_dataset,
            eval_data=valid_dataset,
            epochs=epochs,
            data_collator=collator,
            tokenizer=tokenizer.origin_tokenizer,
        )
        trainer.train()

    print("Start evaluating...")
    test_dataset = ChatDataset(
        file_path=TEST_SET, tokenizer=tokenizer, overwrite=overwrite
    )
    trainer = DLTrainer(
        model=Models.from_finetuned(),
        eval_data=test_dataset,
        data_collator=collator,
        tokenizer=tokenizer.origin_tokenizer,
    )
    eval_result = trainer.evaluate(valid_dataset)
    print(eval_result)


if __name__ == "__main__":
    args = parser.parse_args()
    
    debug: bool = args.debug
    
    if debug:
        profiler = Profile()
        profiler.run("main()")

        stats = Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats("cumulative")
        stats.print_stats()
    else:
        main()
