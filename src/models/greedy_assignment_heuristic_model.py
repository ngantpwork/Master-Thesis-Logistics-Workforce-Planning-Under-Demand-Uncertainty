"""
Greedy Supplier → Dispatcher Assignment Heuristic
================================================

Baseline heuristic model for assigning suppliers to dispatchers.

Changes (single-period refactor)
---------------------------------
* Scenarios must be provided via a ``ScenarioSet``; passing ``None`` when
  scenario evaluation is desired raises ``ValueError``.
* ``use_scenario_based_target`` (default ``False``) — when True, per-scenario
  targets are used:

      W_{d,τ}^ω = (Σ_s M_{s,τ}^ω) × (f_d / Σ_{d'} f_{d'})

  so u_{d,τ}^ω = |L_{d,τ}^ω − W_{d,τ}^ω|.

Notes
-----
* Greedy assigns at **supplier level** → a supplier is never split, so
  coordination cost is always 0.
* Scenarios come from ``ScenarioSet``; internal generation is intentionally
  **not** supported.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from pydantic import BaseModel, Field

from ..schemas import InputData, OutputData, AssignmentType, ScenarioSet
from .config import GreedyAssignmentHeuristicModelConfig


# ==============================================================================
# Cost data containers
# ==============================================================================

class ScenarioCostBreakdown(BaseModel):
    """Cost components for a single scenario ω.

    ``coord_cost`` is always 0 (greedy never splits a supplier).
    """
    omega: int

    coord_cost:    float   # always 0

    deviation_raw:          float
    spike_raw:              float

    deviation_contribution: float
    spike_contribution:     float

    total_raw:          float
    total_contribution: float


class CostStats(BaseModel):
    mean: float
    std:  float
    p5:   float
    p50:  float
    p95:  float
    min:  float
    max:  float


# ==============================================================================
# Existing containers (unchanged)
# ==============================================================================

class AssignmentStep(BaseModel):
    step:             int
    supplier:         str
    parts:            int
    dispatcher:       str
    capacity_before:  int
    capacity_after:   int


class AssignmentResult(BaseModel):
    assignment:         Dict[str, List[Tuple[str, int]]]
    remaining_capacity: Dict[str, int]
    log:                List[AssignmentStep]


# ==============================================================================
# Main model
# ==============================================================================

class GreedyAssignmentHeuristicModel:
    """
    Greedy baseline model for supplier-dispatcher assignment.

    Strategy
    --------
    1. Sort suppliers by descending part count.
    2. Pick dispatcher with largest remaining capacity.
    3. Assign supplier; update capacity.
    4. Repeat until all suppliers are assigned.

    Parameters
    ----------
    use_scenario_based_target : bool
        Compute W_{d,τ}^ω from scenario total load instead of fixed target.
        Default ``False``.

    """

    def __init__(
        self,
        input_data_file_path:        str = None,
        input_data:                  InputData = None,
        model_config:                GreedyAssignmentHeuristicModelConfig = None,
        scenario_set:                ScenarioSet = None,
        # ── new single-period options ──────────────────────────────────────
        use_scenario_based_target:   bool = False,
    ):
        if input_data is not None:
            self.input_data = input_data
        else:
            self.input_data = InputData.load(input_data_file_path)

        self.model_config              = model_config
        self.scenario_set              = scenario_set
        self.use_scenario_based_target = use_scenario_based_target

        capacity_multiplier = self.model_config.capacity_multiplier


        process_suppliers   = self.input_data.suppliers
        process_dispatchers = self.input_data.dispatchers

        self.suppliers_active   = process_suppliers
        self.dispatchers_active = process_dispatchers

        self.suppliers: Dict[str, int] = {
            supplier_id: len(supplier.parts)
            for supplier_id, supplier in process_suppliers.items()
        }

        self.assignment: Dict[str, List[Tuple[str, int]]] = {
            dispatcher_id: []
            for dispatcher_id in process_dispatchers.keys()
        }

        self.capacity: Dict[str, int] = {
            dispatcher_id: int(dispatcher.fte_factor * capacity_multiplier)
            for dispatcher_id, dispatcher in process_dispatchers.items()
        }
        self.log: List[AssignmentStep] = []

        self.assignment_result: AssignmentResult | None = None
        self.output_data: OutputData | None = None

        # ── scenario cost state ───────────────────────────────────────────────
        self.n_scenarios: int = (
            scenario_set.config.n_scenarios if scenario_set is not None else 0
        )
        self.rep_days: List[str] = self.input_data.rep_days

        self._M_scenario: Dict[Tuple[str, str, int], float] = {}

        # Part → supplier mapping
        self._part_to_supplier: Dict[str, str] = {
            pid: sid
            for sid, supplier in process_suppliers.items()
            for pid in supplier.parts
        }


        # Cost parameters
        self._cost_reassign:       float = getattr(model_config, "cost_reassign", 0.0)
        self._cost_deviation:      float = getattr(model_config, "cost_deviation", 1.0)
        self._cost_spike:          float = getattr(model_config, "cost_spike", 1.0)
        self._base_target_minutes: float = getattr(model_config, "base_target_minutes", 480.0)

        # Results populated after solve()
        self.scenario_costs: List[ScenarioCostBreakdown] = []
        self.cost_stats:     Dict[str, CostStats] = {}
        self.workload_stats: Dict[str, Dict[str, float]] = {}
        self.avg_spike:      float = 0.0
        self.obj_value:      Optional[float] = None
        self.W_targets:      Dict[str, float] = {}

    # =========================================================
    # Public entry point
    # =========================================================

    def solve(self) -> OutputData:
        """Execute greedy assignment, then evaluate scenario costs."""

        # ── 1. greedy assignment ──────────────────────────────────────────────
        sorted_suppliers = self._sort_suppliers()
        step = 1

        for supplier, parts in sorted_suppliers:
            dispatcher = self._select_best_dispatcher()
            capacity_before = self.capacity[dispatcher]

            self.assignment[dispatcher].append((supplier, parts))
            self.capacity[dispatcher] -= parts

            self.log.append(
                AssignmentStep(
                    step=step,
                    supplier=supplier,
                    parts=parts,
                    dispatcher=dispatcher,
                    capacity_before=capacity_before,
                    capacity_after=self.capacity[dispatcher],
                )
            )
            step += 1

        self.assignment_result = AssignmentResult(
            assignment=self.assignment,
            remaining_capacity=self.capacity,
            log=self.log,
        )
        self.output_data = OutputData(
            assignment={
                dispatcher: [supplier for supplier, _ in assigned]
                for dispatcher, assigned in self.assignment.items()
            },
            assignment_type=AssignmentType.SUPPLIER_TO_DISPATCHER,
        )

        # ── 2. scenario cost evaluation (optional) ───────────────────────────
        if self.scenario_set is not None:
            print(
                "[INFO] Loading part-level scenarios from ScenarioSet "
                f"{self.scenario_set.scenario_set_id[:8]}…"
            )
            self._M_scenario = self.scenario_set.get_part_workload_matrix()
            self.n_scenarios = self.scenario_set.config.n_scenarios
            print(f"[INFO] Loaded {len(self._M_scenario)} scenario entries.")
            self._evaluate_scenario_costs()
        else:
            print("[INFO] No ScenarioSet provided — skipping scenario cost evaluation.")

        return self.output_data

    # =========================================================
    # Scenario cost evaluation
    # =========================================================

    def _evaluate_scenario_costs(self) -> None:
        """Evaluate greedy assignment quality against all scenarios."""

        # Build flat part → dispatcher mapping from greedy result
        part_assignment: Dict[str, str] = {}
        for dispatcher, supplier_list in self.assignment.items():
            for supplier_id, _ in supplier_list:
                supplier = self.suppliers_active.get(supplier_id)
                if supplier is None:
                    continue
                for pid in supplier.parts:
                    part_assignment[pid] = dispatcher

        dispatchers_active = self.dispatchers_active
        D_active = set(dispatchers_active)
        OMEGA    = list(range(self.n_scenarios))

        # ── fixed workload targets W_{d,τ} ───────────────────────────────────
        W_fixed: Dict[Tuple[str, str], float] = {
            (d, tau): (
                dispatchers_active[d].fte_factor
                * dispatchers_active[d].target_workload.get(
                    tau, self._base_target_minutes
                )
            )
            for d in D_active
            for tau in self.rep_days
        }

        # ── scenario-based targets (optional) ────────────────────────────────
        W_scenario: Dict[Tuple[str, str, int], float] = {}
        if self.use_scenario_based_target:
            total_fte = sum(dispatchers_active[d].fte_factor for d in D_active)
            for omega in OMEGA:
                for tau in self.rep_days:
                    # total load across all parts assigned to any dispatcher
                    total_load = sum(
                        self._M_scenario.get((pid, tau, omega), 0.0)
                        for pid in part_assignment
                    )
                    for d in D_active:
                        fte_share = dispatchers_active[d].fte_factor / total_fte
                        W_scenario[(d, tau, omega)] = total_load * fte_share

        def _target(d: str, tau: str, omega: int) -> float:
            if self.use_scenario_based_target:
                return W_scenario.get((d, tau, omega), 0.0)
            return W_fixed.get((d, tau), 0.0)

        # W_targets for reporting
        if self.use_scenario_based_target:
            self.W_targets = {
                f"{d}|{tau}": float(
                    np.mean([W_scenario.get((d, tau, omega), 0.0) for omega in OMEGA])
                )
                for d in D_active for tau in self.rep_days
            }
        else:
            self.W_targets = {
                f"{d}|{tau}": v for (d, tau), v in W_fixed.items()
            }

        # ── group parts by dispatcher for efficiency ──────────────────────────
        dispatcher_parts: Dict[str, List[str]] = {d: [] for d in D_active}
        for pid, did in part_assignment.items():
            if did in dispatcher_parts:
                dispatcher_parts[did].append(pid)

        # ── per-scenario evaluation ───────────────────────────────────────────
        scenario_costs: List[ScenarioCostBreakdown] = []
        wl_accumulator: Dict[str, Dict[str, List[float]]] = {
            d: {tau: [] for tau in self.rep_days} for d in D_active
        }
        spike_values: List[float] = []

        for omega in OMEGA:
            L: Dict[Tuple[str, str], float] = {}
            for d in D_active:
                for tau in self.rep_days:
                    total_wl = sum(
                        self._M_scenario.get((pid, tau, omega), 0.0)
                        for pid in dispatcher_parts[d]
                    )
                    L[(d, tau)] = total_wl
                    wl_accumulator[d][tau].append(total_wl)

            u: Dict[Tuple[str, str], float] = {
                (d, tau): abs(L[(d, tau)] - _target(d, tau, omega))
                for d in D_active
                for tau in self.rep_days
            }

            sigma_omega = max(u.values()) if u else 0.0
            spike_values.append(sigma_omega)

            dev_raw = self._cost_deviation * sum(u.values())
            spk_raw = self._cost_spike * sigma_omega

            dev_contribution = dev_raw / self.n_scenarios
            spk_contribution = spk_raw / self.n_scenarios

            total_raw         = dev_raw + spk_raw
            total_contribution = dev_contribution + spk_contribution

            scenario_costs.append(
                ScenarioCostBreakdown(
                    omega=omega,
                    coord_cost=0.0,
                    deviation_raw=dev_raw,
                    spike_raw=spk_raw,
                    deviation_contribution=dev_contribution,
                    spike_contribution=spk_contribution,
                    total_raw=total_raw,
                    total_contribution=total_contribution,
                )
            )

        # ── aggregate workload stats ──────────────────────────────────────────
        self.workload_stats = {
            d: {
                tau: float(np.mean(wl_accumulator[d][tau]))
                for tau in self.rep_days
            }
            for d in D_active
        }
        self.avg_spike = float(np.mean(spike_values))

        def _stats(values: List[float]) -> CostStats:
            arr = np.array(values, dtype=float)
            return CostStats(
                mean=float(np.mean(arr)),   std=float(np.std(arr)),
                p5=float(np.percentile(arr, 5)),
                p50=float(np.percentile(arr, 50)),
                p95=float(np.percentile(arr, 95)),
                min=float(np.min(arr)),     max=float(np.max(arr)),
            )

        self.cost_stats = {
            "reassign_cost": _stats([sc.reassign_cost for sc in scenario_costs]),
            "coord_cost":    _stats([sc.coord_cost    for sc in scenario_costs]),
            "deviation_raw": _stats([sc.deviation_raw for sc in scenario_costs]),
            "spike_raw":     _stats([sc.spike_raw     for sc in scenario_costs]),
            "total_raw":     _stats([sc.total_raw     for sc in scenario_costs]),
        }

        self.obj_value = float(
            sum(sc.total_contribution for sc in scenario_costs)
        )
        self.scenario_costs = scenario_costs

        print(
            f"[INFO] Scenario cost evaluation complete. "
            f"obj_value={self.obj_value:,.4f}, "
            f"avg_spike={self.avg_spike:.1f} min"
        )

    # =========================================================
    # Helpers
    # =========================================================

    def _sort_suppliers(self) -> List[Tuple[str, int]]:
        return sorted(self.suppliers.items(), key=lambda x: x[1], reverse=True)

    def _select_best_dispatcher(self) -> str:
        return max(self.capacity, key=lambda d: self.capacity[d])

    # =========================================================
    # Save
    # =========================================================

    def save(self, folder_path: str) -> None:
        os.makedirs(folder_path, exist_ok=True)

        self.input_data.save(os.path.join(folder_path, "input_data.json"))
        self.output_data.save(os.path.join(folder_path, "output_data.json"))

        model_config_json = self.model_config.model_dump()
        model_config_json["use_scenario_based_target"] = self.use_scenario_based_target
        with open(
            os.path.join(folder_path, "model_config.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(model_config_json, f, indent=2)

        if self.scenario_set is not None:
            self.scenario_set.save(os.path.join(folder_path, "scenario_set.json"))

        if self.scenario_costs:
            self._save_scenario_costs_csv(folder_path)
            self._save_cost_stats_csv(folder_path)
            self._save_workload_stats_csv(folder_path)

            result_summary = {
                "use_scenario_based_target": self.use_scenario_based_target,
                "obj_value":                 self.obj_value,
                "avg_spike":                 self.avg_spike,
                "W_targets":                 self.W_targets,
                "workload_stats":            self.workload_stats,
                "cost_stats": {
                    k: v.model_dump() for k, v in self.cost_stats.items()
                },
            }
            with open(
                os.path.join(folder_path, "scenario_result.json"),
                "w", encoding="utf-8",
            ) as f:
                json.dump(result_summary, f, indent=2, ensure_ascii=False)
            print("[INFO] Saved scenario_result.json")

    def _save_scenario_costs_csv(self, folder_path: str) -> None:
        rows = [sc.model_dump() for sc in self.scenario_costs]
        with open(
            os.path.join(folder_path, "scenario_costs.csv"),
            "w", newline="", encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"[INFO] Saved scenario_costs.csv ({len(rows)} rows)")

    def _save_cost_stats_csv(self, folder_path: str) -> None:
        rows = []
        for component, stats in self.cost_stats.items():
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

    def _save_workload_stats_csv(self, folder_path: str) -> None:
        rows = []
        for d, day_stats in self.workload_stats.items():
            for tau, avg_wl in day_stats.items():
                target = self.W_targets.get(f"{d}|{tau}", 0.0)
                rows.append({
                    "dispatcher":      d,
                    "day":             tau,
                    "avg_workload_min": round(avg_wl, 2),
                    "target_min":      round(target, 2),
                    "target_type":     "scenario_mean" if self.use_scenario_based_target else "fixed",
                    "utilization_pct": round(avg_wl / target * 100 if target > 0 else 0.0, 1),
                })
        with open(
            os.path.join(folder_path, "workload_stats.csv"),
            "w", newline="", encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"[INFO] Saved workload_stats.csv ({len(rows)} rows)")

    # =========================================================
    # Reporting
    # =========================================================

    def print_summary(self) -> None:
        print("=" * 70)
        print("   SUPPLIER → DISPATCHER ASSIGNMENT HEURISTIC")
        print(
            f"scenario_target={self.use_scenario_based_target}  "
        )
        print("=" * 70)

        print("\n[INPUT] SUPPLIERS")
        self._print_separator()
        for supplier, parts in self._sort_suppliers():
            bar = "█" * parts
            print(f"  {supplier:<6} {parts:>3} parts  {bar}")

        print("\n[INPUT] DISPATCHERS")
        self._print_separator()
        for dispatcher, cap in self.capacity.items():
            print(f"  {dispatcher}: {cap}")

        print("\n[STEPS]")
        self._print_separator()
        for step in self.log:
            print(
                f"  Step {step.step:<2} | "
                f"{step.supplier:<5} ({step.parts:<2}) "
                f"→ {step.dispatcher:<3} | "
                f"{step.capacity_before:<3} -> {step.capacity_after:<3}"
            )

        print("\n[RESULT]")
        self._print_separator()
        for dispatcher, assigned in self.assignment.items():
            total_parts  = sum(parts for _, parts in assigned)
            supplier_txt = ", ".join(
                f"{supplier}({parts})" for supplier, parts in assigned
            )
            init_cap     = self.capacity[dispatcher]
            utilization  = total_parts / init_cap * 100 if init_cap > 0 else 0.0
            print(f"\n  {dispatcher} (capacity={init_cap})")
            print(f"    Suppliers : {supplier_txt}")
            print(f"    Used      : {total_parts}")
            print(f"    Remaining : {self.capacity[dispatcher]}")
            print(f"    Utilized  : {utilization:.1f}%")

        if self.scenario_costs:
            print("\n[SCENARIOS]")
            self._print_separator()
            print(f"  Scenarios : {self.n_scenarios}")
            print(f"  Rep days  : {self.rep_days}")

            if self.obj_value is not None:
                print(f"\n  Expected obj value : {self.obj_value:,.4f}")
            print(f"  Avg spike σ̄        : {self.avg_spike:.1f} min")

            print("\n[WORKLOAD  (avg over scenarios)]")
            self._print_separator()
            for d, day_stats in sorted(self.workload_stats.items()):
                disp = self.dispatchers_active.get(d)
                fte  = disp.fte_factor if disp else 1.0
                print(f"\n  {d}  (FTE={fte:.2f})")
                for tau in self.rep_days:
                    avg_wl = day_stats.get(tau, 0.0)
                    target = self.W_targets.get(f"{d}|{tau}", 0.0)
                    util   = avg_wl / target * 100 if target > 0 else 0.0
                    bar    = "█" * int(util / 5)
                    lbl    = "(mean)" if self.use_scenario_based_target else "(fixed)"
                    print(
                        f"    {tau:<5} : {avg_wl:>6.0f} / {target:.0f} min  "
                        f"({util:>5.1f}%)  {bar} {lbl}"
                    )

            print("\n[COST BREAKDOWN  (raw values across scenarios)]")
            self._print_separator()
            note_fixed = {"reassign_cost", "coord_cost"}
            hdr = (
                f"  {'Component':<18}  {'mean':>10}  {'std':>8}  "
                f"{'p5':>8}  {'p50':>8}  {'p95':>8}  {'max':>8}"
            )
            print(hdr)
            self._print_separator(char="·")
            for component, cs in self.cost_stats.items():
                tag = "  (fixed)" if component in note_fixed else ""
                print(
                    f"  {component:<18}  {cs.mean:>10.2f}  {cs.std:>8.2f}"
                    f"  {cs.p5:>8.2f}  {cs.p50:>8.2f}  {cs.p95:>8.2f}"
                    f"  {cs.max:>8.2f}{tag}"
                )

            if self.obj_value is not None:
                contrib_sum = (
                    self.cost_stats.get(
                        "reassign_cost",
                        CostStats(mean=0, std=0, p5=0, p50=0, p95=0, min=0, max=0)
                    ).mean
                    + self.cost_stats.get(
                        "coord_cost",
                        CostStats(mean=0, std=0, p5=0, p50=0, p95=0, min=0, max=0)
                    ).mean
                    + sum(
                        self.cost_stats[k].mean / self.n_scenarios
                        for k in ("deviation_raw", "spike_raw")
                        if k in self.cost_stats
                    )
                )
                print(f"\n  obj_value (greedy eval) : {self.obj_value:>10.4f}")
                print(
                    f"  Σ contributions         : {contrib_sum:>10.4f}  "
                    f"(diff={abs(self.obj_value - contrib_sum):.2e})"
                )

        self._print_global_summary()

    def _print_global_summary(self) -> None:
        print("\n[SUMMARY]")
        self._print_separator()
        total_parts    = sum(self.suppliers.values())
        total_capacity = sum(self.capacity.values())
        utilization    = total_parts / total_capacity * 100 if total_capacity > 0 else 0.0
        print(f"  Total suppliers   : {len(self.suppliers)}")
        print(f"  Total parts       : {total_parts}")
        print(f"  Total capacity    : {total_capacity}")
        print(f"  Overall util (%)  : {utilization:.1f}%")
        print(f"  Remaining cap     : {sum(self.capacity.values())}")
        self._print_separator()
        print("Completed.")

    @staticmethod
    def _print_separator(char: str = "-", width: int = 70) -> None:
        print(char * width)