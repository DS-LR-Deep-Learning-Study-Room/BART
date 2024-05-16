from argparse import ArgumentParser

from torch.nn import Module

from .const import TRAIN_SET, VALID_SET
from .data.dataset import ChatDataset
from .model.models import Models, Tokenizer
from .trainer import DLTrainer

parser = ArgumentParser(description="Helper for training & inferencing DL models.")
parser.add_argument(
    "-M", "--model", dest="model",
    type=str, choices=Models.allCases(),
    required=True, help="Select model to use."
)
parser.add_argument(
    "-E", "--epoch", dest="epoch",
    type=int, default=5,
    help="Number of epochs to train."
)

if __name__ == "__main__":
    args = parser.parse_args()
    
    model_name: str = args.model
    epochs: int = args.epoch
    
    # Tokenizer & Model
    tokenizer: Tokenizer
    model: Module
    tokenizer, model = Models.from_pretrained(model_name)
    
    # Dataset
    train_dataset = ChatDataset(file_path=TRAIN_SET, tokenizer=tokenizer)
    valid_dataset = ChatDataset(file_path=VALID_SET, tokenizer=tokenizer)
    
    # print(next(iter(train_dataset)))
    
    trainer = DLTrainer(
        model=model,
        train_data=train_dataset,
        eval_data=valid_dataset,
        epochs=epochs
    )
    trainer.train()
    