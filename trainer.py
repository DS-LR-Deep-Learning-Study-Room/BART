import contextlib
import os
from typing import Optional

import evaluate
import torch.nn as nn
from torch.utils.data import Dataset
from transformers.trainer import Trainer, TrainingArguments

from .const import CHECKPOINT_DIR, MODEL_DIR


class DLTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_data: Optional[Dataset] = None,
        eval_data: Optional[Dataset] = None,
        epochs: float = 3,
        batch_size: int = 8,
        label_names: Optional[list[str]] = None
    ):
        training_args = TrainingArguments(
            output_dir=CHECKPOINT_DIR,
            overwrite_output_dir=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            label_names=label_names,
            load_best_model_at_end=True
        )
        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            compute_metrics=self.compute_metrics
        )
        
        self.metric = evaluate.load("accuracy")
    
    def compute_metrics(self, eval_pred) -> Optional[dict]:
        logits, labels = eval_pred
        predictions = logits.argmax(-1)
        return self.metric.compute(predictions=predictions, references=labels)
    
    def train(self):
        self.trainer.train()
        
        with contextlib.suppress(OSError):
            os.remove(MODEL_DIR)
        self.trainer.save_model(MODEL_DIR)

    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> dict[str, float]:
        return self.trainer.evaluate(eval_dataset=eval_dataset)
