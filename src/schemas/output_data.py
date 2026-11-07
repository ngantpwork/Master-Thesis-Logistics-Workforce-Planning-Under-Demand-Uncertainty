import os
import json
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from enum import Enum
from uuid import uuid4
class AssignmentType(Enum):
    SUPPLIER_TO_DISPATCHER = "supplier_to_dispatcher"
    PART_TO_DISPATCHER = "part_to_dispatcher"

class OutputData(BaseModel):
    """
    Kết quả đầu ra của model: phân công supplier → dispatcher.

    Attributes
    ----------
    assignment : dict {dispatcher_id → list of supplier_id} hoặc dict {dispatcher_id → list of (part_id, supplier_id)}
    assignment_type: AssignmentType
    """
    run_id: Optional[str] = Field(default_factory=lambda: str(uuid4()))
    assignment: Dict[str, List]
    assignment_type: AssignmentType
    solve_time: Optional[float] = None
    status: Optional[int] = None
    create_at: Optional[str] = None
    # -------------------------

    # -------------------------
    # IO
    # -------------------------
    def save(self, path: str):
        output_dir = os.path.dirname(path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(path, "w") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2)
    
    @staticmethod
    def load(path: str) -> "OutputData":
        with open(path, "r") as f:
            raw = json.load(f)

        assignment_type = AssignmentType(raw["assignment_type"])

        if assignment_type == AssignmentType.SUPPLIER_TO_DISPATCHER:
            return SupplierToDispatcherWorkloadBalancingOutputData.model_validate(raw)

        if assignment_type == AssignmentType.PART_TO_DISPATCHER:
            return PartToDispatcherWorkloadBalancingOutputData.model_validate(raw)

        return OutputData.model_validate(raw)
class GreedyAssignmentHeuristicOutputData(OutputData):
    pass

class SupplierToDispatcherWorkloadBalancingOutputData(OutputData):
    reassigned: Optional[List[str]] = None
    objective_value: Optional[float] = None
    mip_gap: Optional[float] = None
    workload_stats: Optional[Dict[str, Dict[str, float]]] = None
    avg_spike: Optional[float] = None
    W_targets: Optional[Dict[str, float]] = None

class PartToDispatcherWorkloadBalancingOutputData(OutputData):
    part_assignment: Optional[Dict[str, List[Dict[str, str]]]] = None  # dispatcher_id → list of {part_id, supplier_id}
    reassigned: Optional[List[str]] = None
    objective_value: Optional[float] = None
    mip_gap: Optional[float] = None
    workload_stats: Optional[Dict[str, Dict[str, float]]] = None
    avg_spike: Optional[float] = None
    W_targets: Optional[Dict[str, float]] = None