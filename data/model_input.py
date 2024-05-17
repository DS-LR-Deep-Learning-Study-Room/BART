from pydantic import BaseModel, ConfigDict
from torch import Tensor


class ModelInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    input_ids: Tensor
    attention_masks: Tensor
    labels: Tensor