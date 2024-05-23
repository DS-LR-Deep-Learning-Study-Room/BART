import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


class Runner:
    def __init__(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer
    ):
        self.model = model
        self.tokenizer = tokenizer

    def run(self, text: str):
        inputs = self.tokenizer(text, return_token_type_ids=False, return_tensors="pt")

        # Generate the output
        with torch.no_grad():
            outputs = self.model.generate(**inputs)

        # Decode and print the output
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Output: {output_text}")
