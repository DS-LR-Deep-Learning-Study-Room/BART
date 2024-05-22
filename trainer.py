import contextlib
import os
from typing import Optional

import evaluate
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from .const import CHECKPOINT_DIR, MODEL_DIR
from .data.tokenizer import Tokenizer


class DLTrainer:
    def __init__(
        self,
        model: nn.Module,
        data_collator,
        tokenizer: Tokenizer,
        train_data: Optional[Dataset] = None,
        eval_data: Optional[Dataset] = None,
        epochs: float = 3,
        batch_size: int = 4,
        device: Optional[torch.device] = None,
    ):
        _use_fp16 = device is not None and device == torch.device("cuda")

        training_args = Seq2SeqTrainingArguments(
            output_dir=CHECKPOINT_DIR,
            overwrite_output_dir=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            fp16=_use_fp16,
            predict_with_generate=True,
            load_best_model_at_end=True,
        )
        self.trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=self.compute_rouge,
        )

        self.tokenizer = tokenizer
        self.rouge = evaluate.load("rouge")

    def compute_rouge(self, eval_pred):
        predictions, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = self.rouge.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )

        res = self.rouge.compute(predictions=decoded_preds, references=decoded_labels)
        res = {key: value * 100 for key, value in res.items()}

        prediction_lens = [
            np.count_nonzero(pred != self.tokenizer.pad_token_id)
            for pred in predictions
        ]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    def train(self):
        self.trainer.train()

        with contextlib.suppress(OSError):
            os.remove(MODEL_DIR)
        self.trainer.save_model(MODEL_DIR)

    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> dict[str, float]:
        return self.trainer.evaluate(eval_dataset=eval_dataset)
