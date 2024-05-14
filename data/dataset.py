import os

import numpy as np
import orjson
from pydantic import TypeAdapter, ValidationError
from torch.utils.data import Dataset
from tqdm import tqdm

from ..model.models import Tokenizer
from .chat_data import ChatData

_BASE_PATH: str = "./data/Korean Chat/"

class QuestionDataset(Dataset):
    """
    Questions Parquet 파일로부터 Dataset을 생성합니다.
    `def __init__(self, filename: str)`
    """

    def __init__(
        self,
        file_path: str,
        tokenizer: Tokenizer,
        max_length: int = 512
    ):
        """
        `filename` : Parquet 파일 이름
        `tokenizer` : Tokenizer 인스턴스
        `max_length` : Token 최대 길이
        """
        super().__init__()
        
        dataset_dir: str = os.path.join(_BASE_PATH, file_path)
        data_files: np.ndarray = [
            f for f in os.listdir(dataset_dir)
            if os.path.isfile(os.path.join(dataset_dir, f))
            and os.path.splitext(f)[-1] == ".json"
        ]
        data_files = [
            os.path.join(dataset_dir, f) for f in data_files
        ]
        data_files: np.ndarray = np.array(data_files)
        
        progress_bar = tqdm(data_files, desc="Preparing dataset: ")
        chat_adapter = TypeAdapter(ChatData)
        chat_data = ChatData(numberOfItems=0, data=[])
        for data_file in progress_bar:
            progress_bar.set_postfix(file=data_file)
            with open(data_file) as json_file:
                _data = orjson.loads(json_file.read())
                try:
                    _chat_data = chat_adapter.validate_python(_data)
                    chat_data.merge_(_chat_data)
                except ValidationError as e:
                    print(e)
        progress_bar.close()

        self.chat_data = chat_data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def num_labels(self) -> int:
        nunique = self.dataframe["label"].nunique()
        if isinstance(nunique, int):
            return nunique
        else:
            return 0

    def __len__(self):
        return self.chat_data.num_chats

    def __getitem__(self, index: int) -> list[int]:
        chat = self.chat_data[index]
        
        encoded_chat = self.tokenizer.encode(
            chat.dialogues,
            padding="max_length", truncation=True, max_length=self.max_length,
            return_tensors="pt"
        )
        print(chat.dialogues)
        print(encoded_chat)
        return encoded_chat