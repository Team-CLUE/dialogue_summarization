import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Tuple, Dict


class SummarizationDataset(Dataset):
    """
    A PyTorch Dataset for summarization task.

    Args:
        data (List(Tuple[str,str])]): List of dictionaries containing 'document' and 'summary' keys.
        tokenizer_name (str): The name of the tokenizer to use. This should be a string identifier that can be passed to `AutoTokenizer.from_pretrained()`
        max_input_length (int): Maximum length of the input sequence. Default is 512.
        max_output_length (int): Maximum length of the output sequence. Default is 128.
     Attributes:
        tokenizer (AutoTokenizer): The tokenizer instance used to tokenize the input.
    """

    def __init__(
        self,
        data: List[Tuple[str, str]],
        tokenizer_name: str,
        max_input_length: int = 512,
        max_output_length: int = 128,
    ) -> None:
        self.data: List[Tuple[str, str]] = data
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_input_length: int = max_input_length
        self.max_output_length: int = max_output_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # Get input text and target summary from the dataset
        input_text, target_summary = self.data[index]

        # Tokenize the input and output text
        encoded_input = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encoded_output = self.tokenizer(
            target_summary,
            max_length=self.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Return a dictionary containing the input and target encodings
        return {
            "input_ids": encoded_input["input_ids"].flatten(),
            "attention_mask": encoded_input["attention_mask"].flatten(),
            "target_ids": encoded_output["input_ids"].flatten(),
            "target_attention_mask": encoded_output["attention_mask"].flatten(),
        }


class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(self, tokenizer_name: str, data, block_size: int):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        batch_encoding = tokenizer(
            data, add_special_tokens=True, truncation=True, max_length=block_size
        )
        self.examples = batch_encoding["input_ids"]
        self.examples = [
            {"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]
