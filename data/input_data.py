from typing import Optional

from pydantic import BaseModel, ConfigDict
from torch import Tensor


class InputData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    input_ids: Tensor
    attention_mask: Optional[Tensor]
    labels: int