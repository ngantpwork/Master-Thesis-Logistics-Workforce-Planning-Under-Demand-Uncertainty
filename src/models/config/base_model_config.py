from pydantic import BaseModel, Field
from typing import Optional

class BaseModelConfig(BaseModel):
    rng_seed: int = 42
    time_limit: Optional[int] = 3600
    mip_gap: Optional[float] = 0.001