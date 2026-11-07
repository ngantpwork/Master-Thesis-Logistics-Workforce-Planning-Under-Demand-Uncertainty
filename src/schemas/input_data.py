import os
import json
import time
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from uuid import uuid4
from ..schemas import Part, Supplier, Dispatcher

class InputData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    parts: Dict[str, Part]
    suppliers: Dict[str, Supplier]
    dispatchers: Dict[str, Dispatcher]
    rep_days: List[str]
    prev_assignment: Optional[Dict] = None
    created_at: Optional[str] = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()))

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
    def load(path: str) -> "InputData":
        with open(path, "r") as f:
            raw = json.load(f)

        return InputData.model_validate(raw)