from typing import Optional

from pydantic import BaseModel, ConfigDict
from torch import Tensor


class ModelInput(BaseModel):
	model_config = ConfigDict(arbitrary_types_allowed=True)

	input_ids: Tensor
	token_type_ids: Optional[Tensor]
	attention_mask: Tensor
	labels: Tensor
