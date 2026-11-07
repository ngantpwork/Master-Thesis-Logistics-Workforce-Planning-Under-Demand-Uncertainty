from typing import Dict
from pydantic import BaseModel, Field
from .base_model_config import BaseModelConfig

class PartToDispatcherWorkloadBalancingModelConfig(BaseModelConfig):
    cost_deviation: float = 1.0
    cost_coord : float = 10.0
    k_max: int = 2
    pareto_mode: bool = False
    max_deviation_objective: float = None