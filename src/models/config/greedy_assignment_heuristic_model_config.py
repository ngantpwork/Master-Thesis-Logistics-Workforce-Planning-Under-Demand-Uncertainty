from .base_model_config import BaseModelConfig

class GreedyAssignmentHeuristicModelConfig(BaseModelConfig):
    capacity_multiplier: float = 20.0

    cost_reassign: float = 20.0
    cost_deviation: float = 10.0
    cost_spike: float = 10.0
    base_target_minutes: float = 480.0