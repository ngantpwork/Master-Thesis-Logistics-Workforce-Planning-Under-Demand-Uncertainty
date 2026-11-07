from pydantic import BaseModel, Field
from typing import Any, Dict, List

# ==============================================================================
# 2. DATA CLASSES
# ==============================================================================

class Part(BaseModel):
    """
    Một linh kiện (part) được giao từ supplier.

    Attributes
    ----------
    part_id      : định danh duy nhất
    supplier_id  : supplier sở hữu linh kiện này
    n_obs        : tổng số lần giao hàng quan sát được trong kỳ t  (nₚ)
    k_obs        : số lần giao hàng critical trong kỳ t             (kₚ)
    deliveries   : dict {ngày_đại_diện → số_lần_giao_dự_kiến} trong kỳ t+1  (aₚτ)
    """
    part_id     : str
    supplier_id : str
    n_obs       : int
    k_obs       : int
    deliveries  : Dict[str, int]   # {tau -> a_{p,tau}}
    base_true_theta: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "part_id"    : self.part_id,
            "supplier_id": self.supplier_id,
            "n_obs"      : self.n_obs,
            "k_obs"      : self.k_obs,
            "deliveries" : self.deliveries,
            "base_true_theta": self.base_true_theta
        }


class Supplier(BaseModel):
    """
    Nhà cung cấp (supplier). Toàn bộ parts của một supplier
    được giao cho cùng một dispatcher.
    """
    supplier_id   : str
    parts : List[str] = Field(default_factory=list)
    base_true_theta: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "supplier_id"  : self.supplier_id,
            "parts"        : self.parts,
            "base_true_theta": self.base_true_theta
        }


class Dispatcher(BaseModel):
    """
    Nhân viên dispatcher.

    Attributes
    ----------
    dispatcher_id : định danh
    fte_factor    : hệ số FTE fₐ ∈ (0,1] — 1.0 full-time, 0.5 half-time
    target_workload : {tau -> W̄_{d,tau}} workload mục tiêu full-time (phút)
    """
    dispatcher_id   : str
    fte_factor      : float = 1.0
    target_workload : Dict[str, float] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dispatcher_id"  : self.dispatcher_id,
            "fte_factor"     : self.fte_factor,
            "target_workload": self.target_workload,
        }