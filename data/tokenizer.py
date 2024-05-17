from typing import TypeAlias

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from .model_input import ModelInput

Tokenizer: TypeAlias = AutoTokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast

class ChatTokenizer:
    def __init__(
        self,
        checkpoint: str,
        max_length: int = 512,
        max_target: int = 128
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        
        self.max_length = max_length
        self.max_target = max_target
    
    def tokenize(
        self,
        text_input: str,
        summary_target: str
    ) -> ModelInput:
        """
        Dialogue(`text_input`)과 Summary(`summary_target`)을 받아 토큰화를 시켜
        반환합니다.
        
        ##### Returns
        ```json
        {
            "input_ids": torch.Tensor, // 토큰화 된 입력
            "attention_masks": torch.Tensor,
            "labels": torch.Tensor // 토큰화 된 Summary
        }
        ```
        """
        model_input = self.tokenizer(
            text_input,
            max_length=self.max_length,
            padding="max_length",
            truncation=True
        )
        
        with self.tokenizer.as_target_tokenizer():
            targets = self.tokenizer(
                summary_target,
                max_length=self.max_target,
                padding="max_length",
                truncation=True
            )
        model_input["labels"] = targets["input_ids"]
        
        return ModelInput.model_validate(model_input)