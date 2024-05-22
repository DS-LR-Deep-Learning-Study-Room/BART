import asyncio
import os
from typing import Any, Optional

import numpy as np
import orjson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas import DataFrame
from pydantic import TypeAdapter, ValidationError
from torch.utils.data import Dataset

from .chat_data import Chat, ChatData
from .const import JSON_DATASET_FILENAME, PARQUET_DATASET_FILENAME
from .tokenizer import ChatTokenizer

_BASE_PATH: str = "./data/Korean Chat/"


class ChatDataset(Dataset):
    """
    Questions Parquet 파일로부터 Dataset을 생성합니다.
    `def __init__(self, filename: str)`
    """

    dataframe: DataFrame

    def __init__(
        self,
        file_path: str,
        tokenizer: ChatTokenizer,
        max_length: int = 512,
        overwrite: bool = False,
        fraction: Optional[float] = 1.0,
        index: Optional[pd.Index] = None,
    ):
        """
        `filename` : Parquet 파일 이름
        `tokenizer` : ChatTokenizer 인스턴스
        `max_length` : Token 최대 길이
        `overwrite` : dataset의 존재 여부와 상관없이 덮어씁니다.
        `fraction` : dataset에서 사용할 데이터의 비율 (0.0 ~ 1.0)
        `index` : dataset에서 사용할 데이터의 index

        `fraction`과 `index` 중 하나의 값은 주어져야 합니다. 주어지지 않을 경우,
        1.0 `fraction`을 사용해 학습됩니다.
        """
        super().__init__()

        _df: DataFrame
        self.chat_adapter = TypeAdapter(ChatData)

        dataset_dir: str = os.path.join(_BASE_PATH, file_path)
        dataset_file_path: str = os.path.join(dataset_dir, PARQUET_DATASET_FILENAME)

        # Parquet 파일이 있을 경우 or overwrite arg가 True일 경우
        if overwrite is False and os.path.exists(dataset_file_path):
            print("Pre-saved dataset parquet file detected. Loading it...")
            _df = pd.read_parquet(dataset_file_path)
            print(f"Dataset loaded from pre-saved file: {len(_df)} chats")
        else:  # Parquet 파일이 없을 경우
            print(
                """
Cannot found pre-saved dataset file or overwrite flag is True.
Saving new one...
                """
            )
            chat_data: ChatData

            # 병합 JSON 파일이 없을 경우
            json_file_path: str = os.path.join(dataset_dir, JSON_DATASET_FILENAME)
            if not os.path.exists(json_file_path):
                print("JSON file not found. Creating new one...")
                data_files = [
                    f
                    for f in os.listdir(dataset_dir)
                    if os.path.isfile(os.path.join(dataset_dir, f))
                    and os.path.splitext(f)[-1] == ".json"
                ]
                data_files = [os.path.join(dataset_dir, f) for f in data_files]
                data_files: np.ndarray = np.array(data_files)

                chat_data = asyncio.run(self.load_json_data(data_files=data_files))
                _json = chat_data.model_dump_json(indent=2, by_alias=True)

                # JSON 파일 저장
                with open(json_file_path, "w") as json_file:
                    json_file.write(_json)

                    print("Saved dataset as json format.")

            # JSON으로부터 dataframe을 만들고 parquet 파일 저장
            with open(json_file_path) as json_file:
                json_data = orjson.loads(json_file.read())
                _df = pd.DataFrame(json_data)

            table = pa.Table.from_pandas(_df)
            pq.write_table(table, dataset_file_path)

        if index is None:  # Using index
            self.dataframe = _df.drop(index=index)
            self.selected_index = self.dataframe.index
        else:  # Using fraction (default : 1.0)
            self.dataframe = _df.sample(frac=fraction)

        self.tokenizer = tokenizer
        self.max_length = max_length

    @property
    def selected_index(self) -> pd.Index:
        return self.selected_index

    async def load_json_data(self, data_files: np.ndarray[str]) -> ChatData:
        _chat_data = ChatData()

        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(self._load_json_data(file, adapter=self.chat_adapter))
                for file in data_files
            ]

            gather_future = asyncio.gather(*tasks)

            results = await gather_future
            for result in results:
                if result is not None:
                    _chat_data.merge_(result)

        return _chat_data

    async def _load_json_data(
        self, data_file: str, adapter: TypeAdapter[ChatData]
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

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, Any]:
        _chat_adapter = TypeAdapter(Chat)

        data: dict[Any, Any] = self.dataframe.iloc[index].to_dict()["data"]

        chat_data = _chat_adapter.validate_python(data)

        model_input = self.tokenizer.tokenize(
            text_input=chat_data.dialogues, summary_target=chat_data.body.summary
        )

        return model_input.model_dump(exclude={"token_type_ids"})
