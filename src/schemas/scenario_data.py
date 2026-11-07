from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field, model_validator

class ScenarioGeneratorConfig(BaseModel):
    """
    Tham số điều khiển quá trình sinh scenario SAA.

    Tách khỏi DataGeneratorConfig (cấu trúc dữ liệu) và ModelConfig (solver)
    vì đây là tham số của bước stochastic modelling độc lập.

    Attributes
    ----------
    rng_seed      : seed riêng, không dùng chung với DataGenerator hay Model
    n_scenarios   : số scenario omega được sample (SAA sample size)
    alpha0, beta0 : hyperparameter của Beta prior (trước khi cộng observations)
    handling_time : phút xử lý mỗi 1 critical delivery  (T^C)
    """

    rng_seed: int = 9999
    n_scenarios: int = 200
    alpha0: float = 1.0
    beta0: float = 1.0
    handling_time: float = 80.0

    @model_validator(mode="after")
    def validate_values(self) -> "ScenarioGeneratorConfig":
        if self.n_scenarios <= 0:
            raise ValueError("n_scenarios phải > 0")
        if self.alpha0 <= 0 or self.beta0 <= 0:
            raise ValueError("alpha0 và beta0 phải > 0")
        if self.handling_time <= 0:
            raise ValueError("handling_time phải > 0")
        return self

class PartScenarioStats(BaseModel):
    """
    Thống kê tổng hợp của critical_count theo từng (part, day) qua tất cả scenario.
    Dùng để lưu vào OLAP FACT_SCENARIO_PART_CRITICAL ở dạng aggregated.
    """

    part_id: str
    supplier_id: str
    day: str
    n_deliveries: int
    mean_critical: float
    std_critical: float
    p5_critical: float
    p50_critical: float
    p95_critical: float
    mean_workload_min: float


class SupplierScenarioStats(BaseModel):
    """
    Thống kê tổng hợp workload của supplier theo từng day qua tất cả scenario.
    Dùng để lưu vào OLAP FACT_SCENARIO_WORKLOAD ở dạng aggregated.
    """

    supplier_id: str
    day: str
    mean_workload_min: float
    std_workload_min: float
    p5_workload_min: float
    p50_workload_min: float
    p95_workload_min: float


class ScenarioSet(BaseModel):
    """
    Kết quả sinh scenario SAA cho một InputData cụ thể.

    Đây là artifact độc lập, được sinh 1 lần và dùng chung
    cho mọi model chạy trên cùng data_id.

    Raw data (theta, critical_count) được lưu dạng flat list để
    Pydantic có thể serialize/deserialize dễ dàng thay vì dùng
    Dict với tuple key (không JSON-serializable).

    Attributes
    ----------
    scenario_set_id : uuid định danh
    data_id         : FK về InputData.id
    config          : ScenarioGeneratorConfig dùng để sinh
    created_at    : timestamp UTC
    n_parts         : số parts thực tế được sinh scenario
    n_rep_days      : số ngày đại diện

    theta_records      : list of {part_id, scenario_index, posterior_alpha, posterior_beta, theta_sampled, true_theta}
    critical_records   : list of {part_id, supplier_id, day, scenario_index,
                                  n_deliveries, critical_count, workload_contribution_min}
    part_stats         : aggregated stats theo (part, day)
    supplier_stats     : aggregated stats theo (supplier, day)
    """

    scenario_set_id: str = Field(default_factory=lambda: str(uuid4()))
    data_id: str
    config: ScenarioGeneratorConfig
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    n_parts: int = 0
    n_rep_days: int = 0

    # Raw records — flat list, JSON-safe
    theta_records: List[Dict] = Field(default_factory=list)
    # {part_id, scenario_index, posterior_alpha, posterior_beta, theta_sampled, true_theta}

    critical_records: List[Dict] = Field(default_factory=list)
    # {part_id, supplier_id, day, scenario_index,
    #  n_deliveries, critical_count, workload_contribution_min}

    # Aggregated stats
    part_stats: List[PartScenarioStats] = Field(default_factory=list)
    supplier_stats: List[SupplierScenarioStats] = Field(default_factory=list)

    # ===========================================================
    # Lookup helpers — được 2 model dùng để build M_scenario
    # ===========================================================

    def get_part_workload_matrix(
        self,
    ) -> Dict[Tuple[str, str, int], float]:
        """
        Trả về dict dạng {(part_id, day, omega) → workload_minutes}.
        PTD model dùng trực tiếp dict này.
        """
        return {
            (r["part_id"], r["day"], r["scenario_index"]): r["workload_contribution_min"]
            for r in self.critical_records
        }

    def get_supplier_workload_matrix(
        self,
    ) -> Dict[Tuple[str, str, int], float]:
        """
        Trả về dict dạng {(supplier_id, day, omega) → workload_minutes}.
        STP model dùng trực tiếp dict này.

        Aggregate bằng cách sum workload của tất cả parts trong supplier —
        đảm bảo hoàn toàn nhất quán với get_part_workload_matrix().
        """
        result: Dict[Tuple[str, str, int], float] = {}
        for r in self.critical_records:
            key = (r["supplier_id"], r["day"], r["scenario_index"])
            result[key] = result.get(key, 0.0) + r["workload_contribution_min"]
        return result

    # ===========================================================
    # IO
    # ===========================================================

    def save(self, path: str) -> None:
        output_dir = os.path.dirname(path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2, ensure_ascii=False)

    @staticmethod
    def load(path: str) -> "ScenarioSet":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return ScenarioSet.model_validate(raw)

    def summary(self) -> Dict:
        n_records = len(self.critical_records)
        return {
            "scenario_set_id": self.scenario_set_id,
            "data_id": self.data_id,
            "created_at": self.created_at,
            "n_scenarios": self.config.n_scenarios,
            "n_parts": self.n_parts,
            "n_rep_days": self.n_rep_days,
            "n_critical_records": n_records,
            "n_theta_records": len(self.theta_records),
            "rng_seed": self.config.rng_seed,
        }