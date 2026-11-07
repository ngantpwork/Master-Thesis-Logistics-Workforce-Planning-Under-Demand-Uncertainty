"""
Stochastic MIP Part-Level Dispatcher Workload Balancing
========================================================

Extended stochastic MIP model (SAA + Bayesian Learning) for assigning
**parts** (not suppliers) to dispatchers with workload balancing and
coordination cost penalisation.

Coordination cost modes
-----------------------

``use_coord_in_objective=True``  (default — existing behaviour)
    Coordination cost is included in the MIP objective.
    Setting to ``False`` decouples coordination from the optimiser:
    the split variables and constraints are still built (so coordination
    is measurable), but the objective term ``cost_coord × split_v[s]``
    is omitted.  Use this to benchmark how much "free" splitting hurts
    versus penalised splitting.

``n_split_exact=False``  (default — existing behaviour)
    When ``True``, replaces the
    k_max upper-bound constraint with an *equality*:
        Σ_d x[s,d] == k_max  ∀s   (legacy mode)
        Σ_s b_split[s] == k_max   (binary mode)
    Forces the model to use *exactly* k_max as the split count / number
    of split suppliers, rather than treating it as a ceiling.

Objective component options
---------------------------

``use_underload_penalty=True``  (default)
    When ``False``: the deviation variable ``u[d,τ,ω]`` only captures
    *overload* (``u >= L - target``).  The symmetric under-target
    constraint ``u >= target - L`` is dropped, so under-utilisation is
    not penalised.

Changes (single-period refactor)
---------------------------------
* Internal scenario generation has been **removed**; a ``ScenarioSet`` is
  required. Passing ``None`` raises ``ValueError``.
* ``use_scenario_based_target`` (default ``False``) — when True the workload
  target for each (dispatcher, day, scenario) is computed from the scenario's
  own fair share.
"""

from __future__ import annotations

import csv
import json
import os
import time
from enum import Enum
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

from ..schemas import (
    InputData, AssignmentType,
    PartToDispatcherWorkloadBalancingOutputData, ScenarioSet,
)
from .config import PartToDispatcherWorkloadBalancingModelConfig


# ==============================================================================
# Internal result containers
# ==============================================================================

class ScenarioEntry(BaseModel):
    part_id:  str
    day:      str
    mean_min: float
    std_min:  float
    p5_min:   float
    p50_min:  float
    p95_min:  float


class ScenarioCostBreakdown(BaseModel):
    """Cost components attributed to a single scenario ω.

    ``coord_cost`` is 0.0 when the respective option is disabled.

    """
    omega: int

    coord_cost:    float   # 0 for M1/M2

    deviation_raw:          float
    total_raw:              float

    deviation_contribution: float
    total_contribution:     float

    # Raw variable values (un-weighted, un-normalized, un-multiplied by cost coeff)
    f_dev_omega:      float = 0.0   # Σ_{d,τ} u[d,τ,ω]          (single scenario)
    f_coord_omega:    float = 0.0   # Σ_s split_v[s]  (fixed across ω)


class CostStats(BaseModel):
    mean: float
    std:  float
    p5:   float
    p50:  float
    p95:  float
    min:  float
    max:  float

class OptimizeProgressPoint(BaseModel):
    elapsed_sec:   float
    incumbent:     Optional[float] = None
    best_bound:    float
    gap_pct:       Optional[float] = None
    node_count:    float
    solution_count: int


class SolveResult(BaseModel):
    assignment:        Dict[str, str]
    obj_value:         Optional[float]
    solve_time:        float
    mip_gap:           Optional[float]
    status:            int
    workload_stats:    Dict[str, Dict[str, float]]
    W_targets:         Dict[str, float]
    n_split_suppliers: int
    total_coord_cost:  float

    scenario_costs: List[ScenarioCostBreakdown] = Field(default_factory=list)
    cost_stats:     Dict[str, CostStats]         = Field(default_factory=dict)

    # Raw (un-normalized, un-weighted) objective component values
    # Keys: "deviation" (only present when solved)
    raw_objective_components: Dict[str, float] = Field(default_factory=dict)
    optimize_progress: List[OptimizeProgressPoint] = Field(default_factory=list)

# ==============================================================================
# Main model
# ==============================================================================

class PartToDispatcherWorkloadBalancingModel:
    """
    Extended stochastic MIP for part-to-dispatcher assignment.

    Parameters
    ----------
    use_scenario_based_target : bool
        Compute W_{d,τ}^ω from scenario total part load.  Default ``False``.
    use_underload_penalty : bool
        Penalise under-utilisation in the deviation term.  Default ``True``.

        When ``False``: the constraint ``u[d,τ,ω] >= target - L`` (under-target)
        is dropped.  ``u[d,τ,ω]`` then only captures overload
        (``u >= L - target``), so ``deviation_raw`` reflects total overload only.
    """

    def __init__(
        self,
        input_data_file_path:        str = None,
        input_data:                  InputData = None,
        model_config:                PartToDispatcherWorkloadBalancingModelConfig = None,
        scenario_set:                ScenarioSet = None,
        # ── single-period options ──────────────────────────────────────────
        use_scenario_based_target:   bool = False,
        # ── coordination objective control ────────────────────────────────
        use_coord_in_objective:      bool = True,
        # ── exact split count (M4 only) ───────────────────────────────────
        n_split_exact:               bool = False,
        # ── objective component options ───────────────────────────────────
        use_underload_penalty:       bool = True,\
        # ── warm start state ─────────────────────────────────────────────
        warm_start_assignment : Optional[Dict[str, str]] = None,
        # ── cpu usage ratio ─────────────────────────────────────────────
        cpu_usage_ratio: float = 0.9
    ):
        if input_data is not None:
            self.input_data = input_data
        else:
            self.input_data = InputData.load(input_data_file_path)

        if scenario_set is None:
            raise ValueError(
                "A ScenarioSet is required. Internal scenario generation has "
                "been removed. Generate scenarios externally and pass them in."
            )

        self.model_config              = model_config
        self.scenario_set              = scenario_set
        self.use_scenario_based_target = use_scenario_based_target
        self.use_coord_in_objective    = use_coord_in_objective
        self.n_split_exact             = n_split_exact
        self.use_underload_penalty     = use_underload_penalty
        
        self.warm_start_assignment     = warm_start_assignment
        self.cpu_usage_ratio = cpu_usage_ratio
        
        self.cost_deviation      = self.model_config.cost_deviation
        self.cost_coord          = self.model_config.cost_coord
        self.k_max               = self.model_config.k_max
        self.time_limit          = self.model_config.time_limit
        self.mip_gap             = self.model_config.mip_gap
        self.pareto_mode         = self.model_config.pareto_mode
        self.max_deviation_objective   = self.model_config.max_deviation_objective
        self.rng_seed            = self.model_config.rng_seed

        if self.pareto_mode:
            if self.max_deviation_objective is None:
                raise ValueError(
                    "max_deviation_objective must be set when pareto_mode=True."
                )
            print(
                f"[INFO] Pareto mode enabled. Max deviation objective: "
                f"{self.max_deviation_objective:.2f} minutes."
            )

        self.suppliers_active   = self.input_data.suppliers
        self.dispatchers_active = self.input_data.dispatchers
        
        self.suppliers   = self.input_data.suppliers
        self.dispatchers = self.input_data.dispatchers
        self.parts            = self.input_data.parts
        self.rep_days: List[str] = self.input_data.rep_days

        # Part → supplier mapping for active period
        self.part_to_supplier: Dict[str, str] = {
            pid: sid
            for sid, supplier in self.suppliers_active.items()
            for pid in supplier.parts
        }



        # P_next = all parts in active period
        self.P_next: Set[str] = {
            pid
            for sid in self.suppliers_active
            for pid in self.suppliers_active[sid].parts
        }


        # ── runtime state ─────────────────────────────────────────────────────
        self.M_scenario: Dict[Tuple[str, str, int], float] = {}
        self.solve_result: SolveResult | None = None
        self.output_data:  PartToDispatcherWorkloadBalancingOutputData | None = None
        self._gurobi_model: gp.Model | None = None


    # =========================================================
    # Public entry point
    # =========================================================

    def solve(self) -> PartToDispatcherWorkloadBalancingOutputData:
        """Execute full pipeline: load scenarios → MIP → OutputData."""

        create_time = time.time()

        obj_mode = (
         "raw weighted-sum"
        )
        print(
            f"[INFO] "
            f"scenario_target={self.use_scenario_based_target}  "
            f"coord_in_obj={self.use_coord_in_objective}  "
            f"n_split_exact={self.n_split_exact}  "
            f"underload_penalty={self.use_underload_penalty}  "
            f"objective={obj_mode}"
        )
        print(
            "[INFO] Loading part-level scenarios from ScenarioSet "
            f"{self.scenario_set.scenario_set_id[:8]}..."
        )
        self.M_scenario  = self.scenario_set.get_part_workload_matrix()
        self.n_scenarios = self.scenario_set.config.n_scenarios
        print(f"[INFO] Loaded {len(self.M_scenario)} scenario entries.")

        print("[INFO] Building and solving MIP…")
        self.solve_result = self._build_and_solve()

        # Supplier-level grouping for OutputData (majority-vote per supplier)
        dispatcher_to_suppliers: Dict[str, List[str]] = {
            d: [] for d in self.dispatchers_active
        }
        sup_dispatcher: Dict[str, str] = {}
        for pid, did in self.solve_result.assignment.items():
            sid = self.part_to_supplier.get(pid)
            if sid and sid not in sup_dispatcher:
                sup_dispatcher[sid] = did
        for sid, did in sup_dispatcher.items():
            dispatcher_to_suppliers.setdefault(did, []).append(sid)

        part_assignment: Dict[str, List[Dict[str, str]]] = {}
        for pid, did in self.solve_result.assignment.items():
            sid = self.part_to_supplier.get(pid)
            if sid:
                part_assignment.setdefault(did, []).append(
                    {"part_id": pid, "supplier_id": sid}
                )

        self.output_data = PartToDispatcherWorkloadBalancingOutputData(
            create_at=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(create_time)),
            assignment=dispatcher_to_suppliers,
            part_assignment=part_assignment,
            assignment_type=AssignmentType.PART_TO_DISPATCHER,
            objective_value=self.solve_result.obj_value,
            solve_time=self.solve_result.solve_time,
            mip_gap=self.solve_result.mip_gap,
            status=self.solve_result.status,
            workload_stats=self.solve_result.workload_stats,
            W_targets=self.solve_result.W_targets,
        )
        return self.output_data

    # =========================================================
    # MIP build & solve
    # =========================================================

    def _build_and_solve(self) -> SolveResult:
        if not GUROBI_AVAILABLE:
            raise RuntimeError("gurobipy is not installed. Cannot solve MIP.")

        P_next   = self.P_next
        D_active = set(self.dispatchers_active)
        S_active = set(self.suppliers_active)
        OMEGA    = list(range(self.n_scenarios))

        sup_parts: Dict[str, List[str]] = {
            sid: [p for p in self.suppliers_active[sid].parts if p in P_next]
            for sid in S_active
        }

        print(
            f"S_active={len(S_active)}, D_active={len(D_active)}, "
            f"OMEGA={len(OMEGA)}, "
            f"underload_penalty={self.use_underload_penalty}"
        )

        model = gp.Model("PartDispatcherWorkloadBalancing")
        if self.time_limit is not None:
            model.setParam("TimeLimit", self.time_limit)
        else:
            model.setParam("TimeLimit", 3600 * 24 * 30)  # default 30 days
        if self.mip_gap is not None:
            model.setParam("MIPGap",    self.mip_gap)
        else:
            print("[INFO] No MIP gap limit set; using Gurobi default (1e-4).")
        model.setParam("Symmetry", 2)
        model.setParam("Presolve", 2)
        model.setParam("OutputFlag", 1)
        model.setParam("LogFile","gurobi.log")  

        number_of_threads = max(1, int(os.cpu_count() * self.cpu_usage_ratio))
        print("[DEBUG] Number of threads", number_of_threads)
        model.setParam("Threads", number_of_threads)

        # Optimize
        # model.setParam("MIPFocus", 3)     # tập trung chứng minh optimality/bound thay vì tìm thêm incumbent
        # model.setParam("Cuts", 3)         # cuts tích cực hơn (Aggressive)
        # model.setParam("Heuristics", 0.02)  # giảm thời gian cho heuristic tìm UB (đã có sẵn UB tốt)
        # model.setParam("VarBranch", 3)    # strong branching — chậm hơn mỗi node nhưng bound tăng nhanh hơn    

        # ------------------------------------------------------------------
        # Decision variables
        # ------------------------------------------------------------------
        print("[DEBUG] Adding z (part assignment) variables…")
        z = model.addVars(
            [(p, d) for p in P_next for d in D_active],
            vtype=GRB.BINARY, name="z",
        )

        if self.warm_start_assignment:
            n_applied = 0
            for p in P_next:
                d_assigned = self.warm_start_assignment.get(p)
                if d_assigned is None or d_assigned not in D_active:
                    continue
                for d in D_active:
                    z[p, d].Start = 1.0 if d == d_assigned else 0.0
                n_applied += 1
            print(
                f"[INFO] MIP Start applied to z for {n_applied}/{len(P_next)} "
                f"parts from stage-1 warm start."
            )


        print("[DEBUG] Adding u (deviation) variables…")
        u = model.addVars(
            [(d, tau, omega)
             for d in D_active for tau in self.rep_days for omega in OMEGA],
            lb=0.0, name="u",
        )

        print("[DEBUG] Adding x (dispatcher-supplier link) variables…")
        x = model.addVars(
            [(s, d) for s in S_active for d in D_active],
            vtype=GRB.BINARY, name="x",
        )

            
        print("[DEBUG] Adding b_split (binary split indicator) variables…")
        split_v = model.addVars(
            list(S_active), vtype=GRB.BINARY, name="b_split",
        )
            
        # ------------------------------------------------------------------
        # Workload targets
        # ------------------------------------------------------------------
        W_fixed = {}
        for d in D_active:
            for tau in self.rep_days:
                W_fixed[(d, tau)] = self.dispatchers_active[d].target_workload[tau]
                print(f"[DEBUG] Fixed workload for dispatcher {d}, day {tau}: {W_fixed[(d, tau)]}")

        if self.use_scenario_based_target:
            total_fte = sum(
                self.dispatchers_active[d].fte_factor for d in D_active
            )
            W_scenario: Dict[Tuple[str, str, int], float] = {}
            for omega in OMEGA:
                for tau in self.rep_days:
                    total_load = sum(
                        self.M_scenario.get((p, tau, omega), 0.0)
                        for p in P_next
                    )
                    for d in D_active:
                        fte_share = self.dispatchers_active[d].fte_factor / total_fte
                        W_scenario[(d, tau, omega)] = total_load * fte_share

        def _target(d: str, tau: str, omega: int) -> float:
            if self.use_scenario_based_target:
                return W_scenario[(d, tau, omega)]
            return W_fixed.get((d, tau), 0.0)

        # ------------------------------------------------------------------
        # Objective
        # ------------------------------------------------------------------

        
        print(
            f"[DEBUG] Building objective "
            f"underload_penalty={self.use_underload_penalty}…"
            f"pareto_mode={self.pareto_mode}  "
        )

        deviation_term = (
            (1.0 / self.n_scenarios) * (
                self.cost_deviation * gp.quicksum(
                    u[d, tau, omega]
                    for d in D_active for tau in self.rep_days for omega in OMEGA
                )
            )
        )

        if split_v is not None and self.use_coord_in_objective:
            coord_term = self.cost_coord * gp.quicksum(split_v[s] for s in S_active)
        else:
            coord_term = 0.0

        if self.pareto_mode:
            # In Pareto mode, we only include the coordination term in the objective.
            # The deviation term is constrained to be below max_deviation_objective.
            obj = coord_term
        else:
            obj = deviation_term + coord_term
        model.setObjective(obj, GRB.MINIMIZE)
        

        # ------------------------------------------------------------------
        # Constraints
        # ------------------------------------------------------------------
        # (C1) Each part assigned to exactly one dispatcher
        print("[DEBUG] (C1) part assignment constraints…")
        for p in P_next:
            model.addConstr(
                gp.quicksum(z[p, d] for d in D_active) == 1,
                name=f"assign_part_{p}",
            )


        # (C4) Workload deviation
        # Over-target half is always added.
        # Under-target half is added only when use_underload_penalty=True.
        print(
            f"[DEBUG] (C4) deviation constraints  "
            f"(underload_penalty={self.use_underload_penalty})…"
        )
        for d in D_active:
            for tau in self.rep_days:
                for omega in OMEGA:
                    L = gp.quicksum(
                        self.M_scenario.get((p, tau, omega), 0.0) * z[p, d]
                        for p in P_next
                    )
                    target = _target(d, tau, omega)
                    # Over-target (overload): always present
                    model.addConstr(
                        u[d, tau, omega] >= L - target,
                        name=f"dev_over_{d}_{tau}_{omega}",
                    )
                    # Under-target (underload): conditional
                    if self.use_underload_penalty:
                        model.addConstr(
                            u[d, tau, omega] >= target - L,
                            name=f"dev_under_{d}_{tau}_{omega}",
                        )

        # (C6/C7) Dispatcher-supplier linking
        if x is not None:
            print("[DEBUG] (C6/C7) dispatcher-supplier linking…")
            for s in S_active:
                parts_s = sup_parts[s]
                for d in D_active:
                    for p in parts_s:
                        model.addConstr(
                            x[s, d] >= z[p, d],
                            name=f"link_lb_{s}_{d}_{p}",
                        )
                    if parts_s:
                        model.addConstr(
                            x[s, d] <= gp.quicksum(z[p, d] for p in parts_s),
                            name=f"link_ub_{s}_{d}",
                        )

        # (C8) Fragmentation / split cost variable linking
        if split_v is not None and x is not None:
            print("[DEBUG] (C8-binary) b_split upper-bound constraints…")
            D_count = len(D_active)
            for s in S_active:
                model.addConstr(
                    gp.quicksum(x[s, d] for d in D_active)
                    <= 1 + (D_count - 1) * split_v[s],
                    name=f"bsplit_lb_{s}",
                )
                model.addConstr(
                    split_v[s] + 1 <= gp.quicksum(x[s, d] for d in D_active),
                    name=f"bsplit_ub_{s}",
                )

        # (C9) k_max cap
        if x is not None:
            op_label = "==" if self.n_split_exact else "<="
            print(
                f"[DEBUG] (C9-binary) global split cap k_max={self.k_max} "
                f"(op={op_label})…"
            )
            lhs = gp.quicksum(split_v[s] for s in S_active)
            if self.n_split_exact:
                model.addConstr(lhs == self.k_max, name="kmax_global_split_eq")
            else:
                model.addConstr(lhs <= self.k_max, name="kmax_global_split")

        if self.pareto_mode:
            # In Pareto mode, we only include the coordination term in the objective.
            # The deviation term is constrained to be below max_deviation_objective.
            model.addConstr(
                deviation_term <= self.max_deviation_objective,
                name="pareto_max_deviation_constraint"
            )

        # Optimizer constraint

        fte_groups: Dict[float, List[str]] = defaultdict(list)
        for d in D_active:
            fte_groups[round(self.dispatchers_active[d].fte_factor, 6)].append(d)

        for fte_val, group in fte_groups.items():
            if len(group) < 2:
                continue
            group_sorted = sorted(group)
            # Lexicographic ordering: dispatcher đứng trước trong group phải nhận
            # "part_id nhỏ nhất được gán" nhỏ hơn hoặc bằng dispatcher đứng sau.
            # Cách đơn giản & hiệu quả: ép tổng workload kỳ vọng giảm dần theo thứ tự group.
            for d1, d2 in zip(group_sorted, group_sorted[1:]):
                model.addConstr(
                    gp.quicksum(
                        self.M_scenario.get((p, tau, 0), 0.0) * z[p, d1]
                        for p in P_next for tau in self.rep_days
                    )
                    >= gp.quicksum(
                        self.M_scenario.get((p, tau, 0), 0.0) * z[p, d2]
                        for p in P_next for tau in self.rep_days
                    ),
                    name=f"symbreak_{fte_val}_{d1}_{d2}",
                )
        
        print("[DEBUG] Computing per-(day,scenario) exact lower-bound cuts…")
        for tau in self.rep_days:
            for omega in OMEGA:
                lb = self._solve_single_day_scenario_lb(tau, omega, list(D_active))
                if lb > 1e-6:
                    model.addConstr(
                        gp.quicksum(u[d, tau, omega] for d in D_active) >= lb,
                        name=f"binpack_lb_{tau}_{omega}",
                    )


        print(f"[DEBUG] Model: {model.NumVars} vars, {model.NumConstrs} constrs")
        self._gurobi_model = model

        # ------------------------------------------------------------------
        # Solve
        # ------------------------------------------------------------------
        print("[INFO] Calling model.optimize()…")
        t_start = time.time()
        progress_callback, progress_log = self._make_progress_callback()
        model.optimize(progress_callback)
        solve_time = time.time() - t_start
        status = model.Status
        print(f"[DEBUG] status={status}, time={solve_time:.2f}s")

        if status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
            return SolveResult(
                assignment={},
                obj_value=None, solve_time=solve_time,
                mip_gap=None, status=status,
                workload_stats={}, W_targets={},
                n_split_suppliers=0, total_coord_cost=0.0,
                scenario_costs=[], cost_stats={},
                raw_objective_components={},
                optimize_progress=progress_log,
            )

        # ------------------------------------------------------------------
        # Extract solution
        # ------------------------------------------------------------------
        assignment: Dict[str, str] = {}
        for p in P_next:
            for d in D_active:
                if z[p, d].X > 0.5:
                    assignment[p] = d
                    break



        workload_stats: Dict[str, Dict[str, float]] = {}
        for d in D_active:
            workload_stats[d] = {}
            for tau in self.rep_days:
                avg_wl = float(np.mean([
                    sum(
                        self.M_scenario.get((p, tau, omega), 0.0)
                        for p in P_next
                        if assignment.get(p) == d
                    )
                    for omega in OMEGA
                ]))
                workload_stats[d][tau] = avg_wl

        # ------------------------------------------------------------------
        # Raw objective components (un-normalized, un-weighted)
        # ------------------------------------------------------------------
        raw_objective_components: Dict[str, float] = {}

        # f_dev = (1/Ω) Σ_{d,τ,ω} u[d,τ,ω]
        raw_objective_components["deviation"] = float(
            (1.0 / self.n_scenarios) * sum(
                u[d, tau, omega].X
                for d in D_active for tau in self.rep_days for omega in OMEGA
            )
        )

        # f_coord = Σ_s split_v[s]
        if split_v is not None:
            raw_objective_components["coord"] = float(
                sum(split_v[s].X for s in S_active)
            )

        n_split_suppliers = 0
        total_coord_cost  = 0.0
        if split_v is not None:
            for s in S_active:
                if split_v[s].X > 0.5:
                    n_split_suppliers += 1
                    total_coord_cost  += self.cost_coord
           

        # W_targets for reporting
        if self.use_scenario_based_target:
            W_targets_report: Dict[str, float] = {
                f"{d}|{tau}": float(
                    np.mean([W_scenario[(d, tau, omega)] for omega in OMEGA])
                )
                for d in D_active for tau in self.rep_days
            }
        else:
            W_targets_report = {
                f"{d}|{tau}": v for (d, tau), v in W_fixed.items()
            }



        scenario_costs: List[ScenarioCostBreakdown] = []

        f_coord_fixed    = float(sum(split_v[s].X for s in S_active)) if split_v is not None else 0.0

        for omega in OMEGA:
            f_dev_omega   = float(sum(u[d, tau, omega].X for d in D_active for tau in self.rep_days))

            dev_raw = self.cost_deviation * f_dev_omega
            dev_contribution = dev_raw / self.n_scenarios

            scenario_costs.append(ScenarioCostBreakdown(
                omega=omega,
                coord_cost=total_coord_cost,
                deviation_raw=dev_raw,
                total_raw=(
                    dev_raw + total_coord_cost
                ),
                deviation_contribution=dev_contribution,
                total_contribution=(
                    dev_contribution
                    + total_coord_cost
                ),
                f_dev_omega=f_dev_omega,
                f_coord_omega=f_coord_fixed,
            ))

        def _stats(values: List[float]) -> CostStats:
            arr = np.array(values, dtype=float)
            return CostStats(
                mean=float(np.mean(arr)), std=float(np.std(arr)),
                p5=float(np.percentile(arr, 5)),
                p50=float(np.percentile(arr, 50)),
                p95=float(np.percentile(arr, 95)),
                min=float(np.min(arr)), max=float(np.max(arr)),
            )

        cost_stats: Dict[str, CostStats] = {
            "coord_cost":    _stats([sc.coord_cost    for sc in scenario_costs]),
            "deviation_raw": _stats([sc.deviation_raw for sc in scenario_costs]),
            "total_raw":     _stats([sc.total_raw     for sc in scenario_costs]),
        }

        return SolveResult(
            assignment=assignment,
            obj_value=float(model.ObjVal),
            solve_time=solve_time,
            mip_gap=float(model.MIPGap),
            status=status,
            workload_stats=workload_stats,
            W_targets=W_targets_report,
            n_split_suppliers=n_split_suppliers,
            total_coord_cost=total_coord_cost,
            scenario_costs=scenario_costs,
            raw_objective_components=raw_objective_components,
            cost_stats=cost_stats,
            optimize_progress=progress_log
        )

    def _solve_single_day_scenario_lb(
        self, tau: str, omega: int, D_active: List[str],
    ) -> float:
        """Giải chính xác min Σ_d u[d,τ,ω] cho 1 (τ,ω) — dùng làm RHS cut."""
        # print(f"[DEBUG] Solving single-day scenario lower bound for τ={tau}, ω={omega}")
        sub = gp.Model("sub_lb")
        sub.setParam("OutputFlag", 0)
        sub.setParam("TimeLimit", 5)

        zz = sub.addVars([(p, d) for p in self.P_next for d in D_active], vtype=GRB.BINARY)
        uu = sub.addVars(D_active, lb=0.0)

        for p in self.P_next:
            sub.addConstr(gp.quicksum(zz[p, d] for d in D_active) == 1)

        for d in D_active:
            target = self.dispatchers_active[d].target_workload[tau]
            L = gp.quicksum(self.M_scenario.get((p, tau, omega), 0.0) * zz[p, d] for p in self.P_next)
            sub.addConstr(uu[d] >= L - target)
            if self.use_underload_penalty:
                sub.addConstr(uu[d] >= target - L)

        sub.setObjective(gp.quicksum(uu[d] for d in D_active), GRB.MINIMIZE)
        sub.optimize()

        return float(sub.ObjVal) if sub.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) else 0.0
        
    def _make_progress_callback(self):
        """Trả về callback ghi lại incumbent/bound/gap theo thời gian để vẽ biểu đồ."""
        progress: List[OptimizeProgressPoint] = []
        t_start = time.time()
        last_log_ts = [0.0]
        LOG_INTERVAL = 5.0  # giây — throttle để không ghi quá dày, chỉnh theo nhu cầu

        def callback(model, where):
            if where == GRB.Callback.MIP:
                now = time.time()
                if now - last_log_ts[0] < LOG_INTERVAL:
                    return
                last_log_ts[0] = now

                obj_best = model.cbGet(GRB.Callback.MIP_OBJBST)
                obj_bnd  = model.cbGet(GRB.Callback.MIP_OBJBND)
                node_cnt = model.cbGet(GRB.Callback.MIP_NODCNT)
                sol_cnt  = model.cbGet(GRB.Callback.MIP_SOLCNT)

                incumbent = obj_best if obj_best < GRB.INFINITY else None
                gap_pct = (
                    abs(obj_best - obj_bnd) / max(abs(obj_best), 1e-9) * 100
                    if incumbent is not None else None
                )

                progress.append(OptimizeProgressPoint(
                    elapsed_sec=now - t_start,
                    incumbent=incumbent,
                    best_bound=obj_bnd,
                    gap_pct=gap_pct,
                    node_count=node_cnt,
                    solution_count=sol_cnt,
                ))

            elif where == GRB.Callback.MIPSOL:
                # Ghi thêm điểm ngay khi có incumbent mới (không đợi interval)
                now = time.time()
                obj_best = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                obj_bnd  = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
                node_cnt = model.cbGet(GRB.Callback.MIPSOL_NODCNT)
                sol_cnt  = model.cbGet(GRB.Callback.MIPSOL_SOLCNT)
                gap_pct = abs(obj_best - obj_bnd) / max(abs(obj_best), 1e-9) * 100

                progress.append(OptimizeProgressPoint(
                    elapsed_sec=now - t_start,
                    incumbent=obj_best,
                    best_bound=obj_bnd,
                    gap_pct=gap_pct,
                    node_count=node_cnt,
                    solution_count=sol_cnt,
                ))
                last_log_ts[0] = now

        return callback, progress

    # =========================================================
    # Save
    # =========================================================

    def save(self, folder_path: str) -> None:
        os.makedirs(folder_path, exist_ok=True)

        self.input_data.save(os.path.join(folder_path, "input_data.json"))

        model_config_json = self.model_config.model_dump()
        model_config_json["use_scenario_based_target"] = self.use_scenario_based_target
        model_config_json["use_coord_in_objective"]    = self.use_coord_in_objective
        model_config_json["n_split_exact"]             = self.n_split_exact
        model_config_json["use_underload_penalty"]     = self.use_underload_penalty

        with open(
            os.path.join(folder_path, "model_config.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(model_config_json, f, indent=2)

        self.output_data.save(os.path.join(folder_path, "output_data.json"))

        if self.scenario_set is not None:
            self.scenario_set.save(os.path.join(folder_path, "scenario_set.json"))

        self._save_scenario_summary(folder_path)

        if self.solve_result is not None:
            result_clean = {
                "use_coord_in_objective": self.use_coord_in_objective,
                "n_split_exact":          self.n_split_exact,
                "use_underload_penalty":  self.use_underload_penalty,
                "assignment":             self.solve_result.assignment,
                "obj_value":              self.solve_result.obj_value,
                "solve_time":             self.solve_result.solve_time,
                "mip_gap":                self.solve_result.mip_gap,
                "status":                 self.solve_result.status,
                "workload_stats":         self.solve_result.workload_stats,
                "W_targets":              self.solve_result.W_targets,
                "n_split_suppliers":      self.solve_result.n_split_suppliers,
                "total_coord_cost":       self.solve_result.total_coord_cost,
                "raw_objective_components": self.solve_result.raw_objective_components,
                "scenario_costs": [
                    e.model_dump() for e in self.solve_result.scenario_costs
                ],
                "cost_stats": {
                    k: v.model_dump()
                    for k, v in self.solve_result.cost_stats.items()
                },
            }
            with open(
                os.path.join(folder_path, "solve_result.json"),
                "w", encoding="utf-8",
            ) as f:
                json.dump(result_clean, f, indent=2, ensure_ascii=False)

            self._save_assignment_csv(folder_path)
            self._save_scenario_costs_csv(folder_path)
            self._save_cost_stats_csv(folder_path)
            self._save_optimize_progress_csv(folder_path)   

        self._save_model_files(folder_path)

    def _save_scenario_summary(self, folder_path: str) -> None:
        rows = []
        for pid in sorted(self.P_next):
            for tau in self.rep_days:
                vals = [
                    self.M_scenario.get((pid, tau, omega), 0.0)
                    for omega in range(self.n_scenarios)
                ]
                rows.append({
                    "part_id":  pid,
                    "day":      tau,
                    "mean_min": round(float(np.mean(vals)), 2),
                    "std_min":  round(float(np.std(vals)), 2),
                    "p5_min":   round(float(np.percentile(vals, 5)), 2),
                    "p50_min":  round(float(np.percentile(vals, 50)), 2),
                    "p95_min":  round(float(np.percentile(vals, 95)), 2),
                })
        if rows:
            with open(
                os.path.join(folder_path, "scenarios_summary.csv"),
                "w", newline="", encoding="utf-8",
            ) as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

    def _save_assignment_csv(self, folder_path: str) -> None:
        if not self.solve_result or not self.solve_result.assignment:
            return
        rows = []
        for pid, did in sorted(self.solve_result.assignment.items()):
            sid = self.part_to_supplier.get(pid, "UNKNOWN")
            rows.append({
                "part_id":         pid,
                "supplier_id":     sid,
                "dispatcher":      did,
            })
        with open(
            os.path.join(folder_path, "assignment.csv"),
            "w", newline="", encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def _save_scenario_costs_csv(self, folder_path: str) -> None:
        if not self.solve_result or not self.solve_result.scenario_costs:
            return
        rows = [sc.model_dump() for sc in self.solve_result.scenario_costs]
        with open(
            os.path.join(folder_path, "scenario_costs.csv"),
            "w", newline="", encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"[INFO] Saved scenario_costs.csv ({len(rows)} rows)")

    def _save_cost_stats_csv(self, folder_path: str) -> None:
        if not self.solve_result or not self.solve_result.cost_stats:
            return
        rows = []
        for component, stats in self.solve_result.cost_stats.items():
            row = {"cost_component": component}
            row.update(stats.model_dump())
            rows.append(row)
        with open(
            os.path.join(folder_path, "cost_stats.csv"),
            "w", newline="", encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"[INFO] Saved cost_stats.csv ({len(rows)} components)")

    def _save_model_files(self, folder_path: str) -> None:
        if self._gurobi_model is None:
            print("[WARN] _gurobi_model not set – skipping model file export.")
            return
        lp_path  = os.path.join(folder_path, "model.lp")
        mps_path = os.path.join(folder_path, "model.mps")
        try:
            self._gurobi_model.write(lp_path)
            print(f"[INFO] Gurobi LP  written → {lp_path}")
            self._gurobi_model.write(mps_path)
            print(f"[INFO] Gurobi MPS written → {mps_path}")
        except gp.GurobiError as exc:
            print(f"[WARN] Could not write model files: {exc}")
    
    def _save_optimize_progress_csv(self, folder_path: str) -> None:
        if not self.solve_result or not self.solve_result.optimize_progress:
            return
        rows = [p.model_dump() for p in self.solve_result.optimize_progress]
        with open(
            os.path.join(folder_path, "optimize_progress.csv"),
            "w", newline="", encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"[INFO] Saved optimize_progress.csv ({len(rows)} points)")

    # =========================================================
    # Reporting
    # =========================================================

    def print_summary(self) -> None:
        r = self.solve_result

        coord_mode_label = (
            "binary (1 meeting/split supplier)"
        )

        print("=" * 70)
        print("   PART → DISPATCHER ASSIGNMENT  (STOCHASTIC MIP)")
        print(f"   Options : "
              f"scenario_target={self.use_scenario_based_target}  ")
        coord_obj_note = "" if self.use_coord_in_objective else "  ⚠ NOT in objective"
        exact_note     = f"  n_split_exact={self.n_split_exact}" if self.n_split_exact else ""
        print(f"   Coord   : {coord_mode_label}{coord_obj_note}{exact_note}")
        underload_note     = "" if self.use_underload_penalty  else "  ⚠ underload NOT penalised (overload only)"
        print(f"   Deviation: {underload_note if underload_note else 'over + under target'}")
        print("=" * 70)

        print("\n[INPUT] PARTS")
        self._print_separator()
        print(f"  Total (active) : {len(self.P_next)}  ")

        print("\n[INPUT] DISPATCHERS")
        self._print_separator()
        for did, disp in self.dispatchers_active.items():
            print(f"  {did}  FTE={disp.fte_factor:.2f}")

        print("\n[SCENARIOS]")
        self._print_separator()
        print(f"  Scenarios : {self.n_scenarios}")
        print(f"  Rep days  : {self.rep_days}")

        if r is None:
            print("\n  (solve() has not been called yet)")
            return

        print("\n[SOLVER]")
        self._print_separator()
        print(f"  Status       : {r.status}")
        if r.obj_value is not None:
            obj_label = "weighted-sum"
            print(f"  Objective ({obj_label}) : {r.obj_value:,.6f}")
        if r.mip_gap is not None:
            print(f"  MIP gap      : {r.mip_gap * 100:.2f}%")
        print(f"  Solve time   : {r.solve_time:.2f}s")

        if r.raw_objective_components:
            print("\n[RAW OBJECTIVE COMPONENTS  (un-normalized, un-weighted)]")
            self._print_separator()
            print(f"  {'Component':<12}  {'f_i(x)':>12}  {'in objective':>14}")
            self._print_separator(char="·")
            obj_flags = {
                "deviation": True,
                "coord":     self.use_coord_in_objective,
            }
            for k, v in r.raw_objective_components.items():
                in_obj = obj_flags.get(k, True)
                flag   = "yes" if in_obj else "no  [reported only]"
                print(f"  {k:<12}  {v:>12.4f}  {flag:>14}")

        obj_note = "" if self.use_coord_in_objective else "  [not in objective]"
        print(f"  Split suppls : {r.n_split_suppliers}  [{coord_mode_label}]{obj_note}")
        print(f"  Coord cost   : {r.total_coord_cost:,.2f}{obj_note}")

        kmax_label = (
            f"exactly {self.k_max} suppliers split (global)"
            if self.n_split_exact
            else f"max {self.k_max} suppliers split (global)"
        )

        print(f"  k_max cap    : {kmax_label}")

        if r.cost_stats:
            print("\n[COST BREAKDOWN  (raw values across scenarios)]")
            self._print_separator()
            note_fixed = {"coord_cost"}
            hdr = (
                f"  {'Component':<18}  {'mean':>10}  {'std':>8}  "
                f"{'p5':>8}  {'p50':>8}  {'p95':>8}  {'max':>8}"
            )
            print(hdr)
            self._print_separator(char="·")
            for component, cs in r.cost_stats.items():
                tag = "  (fixed)" if component in note_fixed else ""
                print(
                    f"  {component:<18}  {cs.mean:>10.2f}  {cs.std:>8.2f}"
                    f"  {cs.p5:>8.2f}  {cs.p50:>8.2f}  {cs.p95:>8.2f}"
                    f"  {cs.max:>8.2f}{tag}"
                )
            if r.obj_value is not None:
                contrib_sum = (
                   (r.cost_stats["coord_cost"].mean
                       if "coord_cost" in r.cost_stats else 0.0)
                    + sum(
                        r.cost_stats[k].mean / self.n_scenarios
                        for k in ("deviation_raw")
                        if k in r.cost_stats
                    )
                )
                print(f"\n  obj_value (solver)  : {r.obj_value:>10.4f}")
                print(
                    f"  Σ contributions     : {contrib_sum:>10.4f}  "
                    f"(diff={abs(r.obj_value - contrib_sum):.2e})"
                )

        print("\n[RESULT]")
        self._print_separator()
        d_parts: Dict[str, List[str]] = {}
        for p, d in r.assignment.items():
            d_parts.setdefault(d, []).append(p)

        for did in sorted(self.dispatchers_active):
            parts_assigned = d_parts.get(did, [])
            disp           = self.dispatchers_active[did]
            sups = {
                self.part_to_supplier[p]
                for p in parts_assigned
                if p in self.part_to_supplier
            }

            print(f"\n  {did}  (FTE={disp.fte_factor:.2f})")
            print(f"    Parts      : {len(parts_assigned)}")
            print(f"    Suppliers  : {len(sups)}")

            for tau in self.rep_days:
                avg_wl = r.workload_stats.get(did, {}).get(tau, 0.0)
                target = r.W_targets.get(f"{did}|{tau}", 0.0)
                util   = avg_wl / target * 100 if target > 0 else 0.0
                bar    = "█" * int(util / 5)
                lbl    = "(mean)" if self.use_scenario_based_target else "(fixed)"
                print(
                    f"    {tau:<5} : {avg_wl:>6.0f} / {target:.0f} min  "
                    f"({util:>5.1f}%)  {bar} {lbl}"
                )

        print("\n[SUMMARY]")
        self._print_separator()
        print(f"  Total parts       : {len(r.assignment)}")
        print(f"  Total dispatchers : {len(self.dispatchers_active)}")
        if r.obj_value is not None:
            print(f"  Objective value   : {r.obj_value:,.4f}")
        self._print_separator()
        print("Completed.")

    @staticmethod
    def _print_separator(char: str = "-", width: int = 70) -> None:
        print(char * width)