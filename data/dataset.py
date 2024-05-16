import asyncio
import os
from typing import Optional

import numpy as np
import orjson
import torch
from pydantic import TypeAdapter, ValidationError
from torch.utils.data import Dataset

from ..model.models import Tokenizer
from .chat_data import ChatData

_BASE_PATH: str = "./data/Korean Chat/"
_TORCH_DATASET_FILENAME: str = "dataset.pt"

class ChatDataset(Dataset):
    """
    Questions Parquet 파일로부터 Dataset을 생성합니다.
    `def __init__(self, filename: str)`
    """

    def __init__(
        self,
        file_path: str,
        tokenizer: Tokenizer,
        max_length: int = 512,
        overwrite: bool = False
    ):
        """
        `filename` : Parquet 파일 이름
        `tokenizer` : Tokenizer 인스턴스
        `max_length` : Token 최대 길이
        """
        super().__init__()
        
        dataset_dir: str = os.path.join(_BASE_PATH, file_path)
        dataset_file_path: str = os.path.join(dataset_dir, _TORCH_DATASET_FILENAME)
        
        if overwrite is False and os.path.exists(dataset_file_path):
            print("Pre-saved dataset file detected. Loading it...")
            chat_data = torch.load(dataset_file_path)
            print(f"Dataset loaded from pre-saved file: {chat_data.num_chats} chats")
            
            self.chat_data = chat_data
        else:
            print(
                """
Cannot found pre-saved dataset file or overwrite flag is True.
Saving new one...
                """
            )
            data_files = [
                f for f in os.listdir(dataset_dir)
                if os.path.isfile(os.path.join(dataset_dir, f))
                and os.path.splitext(f)[-1] == ".json"
            ]
            data_files = [
                os.path.join(dataset_dir, f) for f in data_files
            ]
            data_files: np.ndarray = np.array(data_files)
            
            chat_data = asyncio.run(
                self.load_json_data(data_files=data_files)
            )
            torch.save(chat_data, dataset_file_path)
            print(f"Dataset loaded and saved: {chat_data.num_chats} chats")

            self.chat_data = chat_data
        
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    async def load_json_data(self, data_files: np.ndarray[str]) -> ChatData:
        chat_adapter = TypeAdapter(ChatData)
        
        _chat_data = ChatData()
        
        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(self._load_json_data(file, adapter=chat_adapter))
                for file in data_files
            ]
            
            gather_future = asyncio.gather(*tasks)
            
            results = await gather_future
            for result in results:
                if result is not None:
                    _chat_data.merge_(result)
        
        return _chat_data
    
    async def _load_json_data(
        self,
        data_file: str,
        adapter: TypeAdapter[ChatData]
    ) -> Optional[ChatData]:
        try:
            with open(data_file) as json_file:
                _data = orjson.loads(json_file.read())
                return adapter.validate_python(_data)
        except ValidationError as e:
            print(f"Validation error in file {data_file}: {e}")
        except FileNotFoundError as e:
            print(f"File not found: {e}")
        return None

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
            chat,
            padding="max_length", truncation=True, max_length=self.max_length,
            return_tensors="pt"
        )
        print(chat)
        return encoded_chat