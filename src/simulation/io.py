"""
world_io.py
===========
Persistence and schema-bridge utilities for the simulation World.

Three capabilities
------------------
1. Save / Load World
   Save the full World state to a JSON file and restore it exactly.
   All fields — settings, periods, suppliers, parts, dispatchers,
   observations, period states, trajectories — are serialised.

   API:
       save_world(world, path)        → writes  <path>.json
       load_world(path)               → returns World

2. World  →  InputData  (bridge to DataGenerator / ScenarioGenerator)
   Convert a realized World (one or two consecutive periods) into the
   ``InputData`` dict format expected by ``DataGenerator`` and
   ``ScenarioGenerator`` from the second codebase.

   The mapping is:
       World period  t   →  suppliers  /  dispatchers
       Changeset is inferred by diffing the two periods.

   API:
       world_to_input_data(world, period_t, period_t1, rep_days, handling_time)
           → InputData (plain dict, matches DataGenerator.generate() output)

3. InputData  →  ScenarioGenerator-ready dict
   Thin pass-through helper that validates the InputData dict and returns
   the exact structure expected by ScenarioGenerator.

   API:
       to_scenario_generator_input(input_data)  → same dict (validated)

Usage example
-------------
    from world_io import save_world, load_world, world_to_input_data

    # ── persist ────────────────────────────────────────────────────
    save_world(world, "checkpoints/world_period_12")
    world2 = load_world("checkpoints/world_period_12.json")

    # ── bridge ─────────────────────────────────────────────────────
    input_data = world_to_input_data(world, period_t=10, period_t1=11)

    # Use directly with ScenarioGenerator (second codebase)
    from scenario_generator_v2 import ScenarioGenerator, ScenarioGeneratorConfig
    sg = ScenarioGenerator(input_data=input_data)
    scenario_set = sg.generate()
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime, timezone
from uuid import uuid4

from .simulation_entities import (
    DeliveryObservation,
    Dispatcher,
    Part,
    PartPeriodState,
    PartTrueThetaTrajectory,
    Period,
    Supplier,
    World,
    WorldSettings,
    WEEKDAY_TO_OFFSET
)


# ---------------------------------------------------------------------------
# Internal serialisation helpers
# ---------------------------------------------------------------------------

def _date_to_str(d: date) -> str:
    return d.isoformat()


def _str_to_date(s: str) -> date:
    return date.fromisoformat(s)


def _settings_to_dict(s: WorldSettings) -> dict:
    """Serialise WorldSettings to a plain dict (JSON-safe)."""
    return {
        "n_simulation_periods":  s.n_simulation_periods,
        "current_period":        s.current_period,
        "simulation_start_date": _date_to_str(s.simulation_start_date),
        "weeks_per_period":      s.weeks_per_period,
        "n_init_suppliers":      s.n_init_suppliers,
        "n_dispatchers":         s.n_dispatchers,
        "n_new_suppliers_per_period":   s.n_new_suppliers_per_period,
        "n_new_dispatchers_per_period": s.n_new_dispatchers_per_period,
        "target_workload_dispatcher_fulltime_per_day":
            s.target_workload_dispatcher_fulltime_per_day,
        "target_workload_dispatcher_parttime_per_day":
            s.target_workload_dispatcher_parttime_per_day,
        "shortage_concentration_rho":   s.shortage_concentration_rho,
        "freeze_structure_across_horizon": s.freeze_structure_across_horizon,
        "non_stationary_theta_share":   s.non_stationary_theta_share,
        "allowed_non_stationary_theta_modes": list(s.allowed_non_stationary_theta_modes),
        # Hierarchical parameters
        "use_hierarchical_theta": s.use_hierarchical_theta,
        "m_global":   s.m_global,
        "kappa_s":    s.kappa_s,
        "kappa_p":    s.kappa_p,
        # Non-stationary regime params (scalar only, callables are skipped)
        "regime_change_min_period":          s.regime_change_min_period,
        "regime_change_max_period":          s.regime_change_max_period,
        "temporary_shock_min_start_period":  s.temporary_shock_min_start_period,
        "temporary_shock_max_start_period":  s.temporary_shock_max_start_period,
        "temporary_shock_min_duration_periods": s.temporary_shock_min_duration_periods,
        "temporary_shock_max_duration_periods": s.temporary_shock_max_duration_periods,
        "linear_trend_final_period":         s.linear_trend_final_period,
    }


def _settings_from_dict(d: dict) -> WorldSettings:
    """Restore WorldSettings from a plain dict (callables are None)."""
    return WorldSettings(
        n_simulation_periods  = d["n_simulation_periods"],
        current_period        = d["current_period"],
        simulation_start_date = _str_to_date(d["simulation_start_date"]),
        weeks_per_period      = d["weeks_per_period"],
        n_init_suppliers      = d["n_init_suppliers"],
        n_dispatchers         = d["n_dispatchers"],
        n_new_suppliers_per_period   = d.get("n_new_suppliers_per_period", 0),
        n_new_dispatchers_per_period = d.get("n_new_dispatchers_per_period", 0),
        target_workload_dispatcher_fulltime_per_day =
            d.get("target_workload_dispatcher_fulltime_per_day", 300.0),
        target_workload_dispatcher_parttime_per_day =
            d.get("target_workload_dispatcher_parttime_per_day", 180.0),
        shortage_concentration_rho     = d.get("shortage_concentration_rho", 0.0),
        freeze_structure_across_horizon= d.get("freeze_structure_across_horizon", False),
        non_stationary_theta_share     = d.get("non_stationary_theta_share", 0.0),
        allowed_non_stationary_theta_modes =
            tuple(d.get("allowed_non_stationary_theta_modes",
                        ["regime_change", "temporary_shock", "linear_trend"])),
        use_hierarchical_theta = d.get("use_hierarchical_theta", False),
        m_global               = d.get("m_global", 0.10),
        kappa_s                = d.get("kappa_s", 30.0),
        kappa_p                = d.get("kappa_p", 20.0),
        regime_change_min_period         = d.get("regime_change_min_period", 1),
        regime_change_max_period         = d.get("regime_change_max_period"),
        temporary_shock_min_start_period = d.get("temporary_shock_min_start_period", 1),
        temporary_shock_max_start_period = d.get("temporary_shock_max_start_period"),
        temporary_shock_min_duration_periods =
            d.get("temporary_shock_min_duration_periods", 1),
        temporary_shock_max_duration_periods =
            d.get("temporary_shock_max_duration_periods", 1),
        linear_trend_final_period = d.get("linear_trend_final_period"),
        # Callable samplers are not serialisable — restored as None.
        # Callers must re-attach them after load_world() if needed.
        supplier_part_count_sampler       = None,
        delivery_day_pattern_sampler      = None,
        base_shortage_rate_sampler        = None,
        delivery_call_off_rate_sampler    = None,
        regime_change_target_sampler      = None,
        temporary_shock_value_sampler     = None,
        linear_trend_target_sampler       = None,
    )


def _observation_to_dict(obs: DeliveryObservation) -> dict:
    return {
        "period_id":       obs.period_id,
        "delivery_day":    obs.delivery_day,
        "observation_date": _date_to_str(obs.observation_date),
        "is_critical":     obs.is_critical,
    }


def _observation_from_dict(d: dict) -> DeliveryObservation:
    return DeliveryObservation(
        period_id       = d["period_id"],
        delivery_day    = d["delivery_day"],
        observation_date = _str_to_date(d["observation_date"]),
        is_critical     = d["is_critical"],
    )


def _period_state_to_dict(ps: PartPeriodState) -> dict:
    return {
        "period_id":                  ps.period_id,
        "true_part_theta_in_period":  ps.true_part_theta_in_period,
        "n_possible_delivery_calloffs": ps.n_possible_delivery_calloffs,
        "n_actual_delivery_calloffs":   ps.n_actual_delivery_calloffs,
        "n_critical_delivery_calloffs": ps.n_critical_delivery_calloffs,
        "n_non_critical_delivery_calloffs": ps.n_non_critical_delivery_calloffs,
        "observed_theta_in_period":   ps.observed_theta_in_period,
        "posterior_alpha":            ps.posterior_alpha,
        "posterior_beta":             ps.posterior_beta,
        "posterior_mean":             ps.posterior_mean,
        "posterior_variance":         ps.posterior_variance,
    }


def _period_state_from_dict(d: dict) -> PartPeriodState:
    return PartPeriodState(
        period_id                       = d["period_id"],
        true_part_theta_in_period       = d["true_part_theta_in_period"],
        n_possible_delivery_calloffs    = d.get("n_possible_delivery_calloffs", 0),
        n_actual_delivery_calloffs      = d.get("n_actual_delivery_calloffs", 0),
        n_critical_delivery_calloffs    = d.get("n_critical_delivery_calloffs", 0),
        n_non_critical_delivery_calloffs= d.get("n_non_critical_delivery_calloffs", 0),
        observed_theta_in_period        = d.get("observed_theta_in_period"),
        posterior_alpha                 = d.get("posterior_alpha", 1.0),
        posterior_beta                  = d.get("posterior_beta", 1.0),
        posterior_mean                  = d.get("posterior_mean", 0.5),
        posterior_variance              = d.get("posterior_variance", 1.0 / 12.0),
    )


def _part_to_dict(part: Part) -> dict:
    return {
        "part_id":                part.part_id,
        "start_period":           part.start_period,
        "end_period":             part.end_period,
        "delivery_day_pattern":   part.delivery_day_pattern,
        "delivery_call_off_rate": part.delivery_call_off_rate,
        "base_true_part_theta":   part.base_true_part_theta,
        "is_true_theta_stationary": part.is_true_theta_stationary,
        "delivery_observations":  [_observation_to_dict(o) for o in part.delivery_observations],
        "period_state":           [_period_state_to_dict(ps) for ps in part.period_state],
    }


def _part_from_dict(d: dict) -> Part:
    p = Part(
        part_id                = d["part_id"],
        start_period           = d["start_period"],
        end_period             = d["end_period"],
        delivery_day_pattern   = d.get("delivery_day_pattern", []),
        delivery_call_off_rate = d.get("delivery_call_off_rate", 1.0),
        base_true_part_theta   = d.get("base_true_part_theta", 0.0),
        is_true_theta_stationary = d.get("is_true_theta_stationary", True),
    )
    p.delivery_observations = [_observation_from_dict(o) for o in d.get("delivery_observations", [])]
    p.period_state          = [_period_state_from_dict(ps) for ps in d.get("period_state", [])]
    return p


def _supplier_to_dict(sup: Supplier) -> dict:
    return {
        "supplier_id":        sup.supplier_id,
        "supplier_risk_factor": sup.supplier_risk_factor,
        "supplier_theta":     sup.supplier_theta,
        "part":               [_part_to_dict(p) for p in sup.part],
    }


def _supplier_from_dict(d: dict) -> Supplier:
    sup = Supplier(
        supplier_id          = d["supplier_id"],
        supplier_risk_factor = d.get("supplier_risk_factor", 0.0),
        supplier_theta       = d.get("supplier_theta"),
    )
    sup.part = [_part_from_dict(p) for p in d.get("part", [])]
    return sup


def _dispatcher_to_dict(disp: Dispatcher) -> dict:
    return {
        "dispatcher_id":           disp.dispatcher_id,
        "employment_type":         disp.employment_type,
        "target_workload_per_day": disp.target_workload_per_day,
        "assigned_supplier_ids":   list(disp.assigned_supplier_ids),
    }


def _dispatcher_from_dict(d: dict) -> Dispatcher:
    disp = Dispatcher(
        dispatcher_id          = d["dispatcher_id"],
        employment_type        = d.get("employment_type", "full_time"),
        target_workload_per_day= d.get("target_workload_per_day", 300.0),
    )
    disp.assigned_supplier_ids = d.get("assigned_supplier_ids", [])
    return disp


def _period_to_dict(period: Period) -> dict:
    return {
        "period_id":   period.period_id,
        "start_date":  _date_to_str(period.start_date),
        "supplier":    [_supplier_to_dict(s) for s in period.supplier],
        "dispatcher":  [_dispatcher_to_dict(d) for d in period.dispatcher],
        "assignment":  {str(k): v for k, v in period.assignment.items()},
    }


def _period_from_dict(d: dict) -> Period:
    period = Period(
        period_id  = d["period_id"],
        start_date = _str_to_date(d["start_date"]),
    )
    period.supplier   = [_supplier_from_dict(s) for s in d.get("supplier", [])]
    period.dispatcher = [_dispatcher_from_dict(disp) for disp in d.get("dispatcher", [])]
    period.assignment = {int(k): v for k, v in d.get("assignment", {}).items()}
    return period


def _trajectory_to_dict(traj: PartTrueThetaTrajectory) -> dict:
    return {
        "part_id":       traj.part_id,
        "theta_by_period": {str(k): v for k, v in traj.theta_by_period.items()},
    }


def _trajectory_from_dict(d: dict) -> PartTrueThetaTrajectory:
    traj = PartTrueThetaTrajectory(
        part_id = d["part_id"],
    )
    traj.theta_by_period = {int(k): v for k, v in d.get("theta_by_period", {}).items()}
    return traj


# ---------------------------------------------------------------------------
# Public API — Save / Load
# ---------------------------------------------------------------------------

def save_world(world: World, path: str) -> str:
    """
    Serialise the full World state to a JSON file.

    Note: ``WorldSettings`` callable samplers (e.g. base_shortage_rate_sampler)
    are not serialisable and are stored as ``null``.  Re-attach them after
    ``load_world()`` if the loaded world needs to be stepped forward.

    Parameters
    ----------
    world : World
    path  : str
        Target file path.  The ``.json`` extension is appended automatically
        if not already present.

    Returns
    -------
    str — absolute path to the written file.
    """
    if not path.endswith(".json"):
        path = path + ".json"

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    payload = {
        "schema_version": "1.1",
        "settings":       _settings_to_dict(world.settings),
        "current_period": world.current_period,
        "next_supplier_id":  world.next_supplier_id,
        "next_dispatcher_id": world.next_dispatcher_id,
        "next_part_id":      world.next_part_id,
        "rng_state":  world.rng.getstate()[1],   # list of ints — JSON-safe
        "period":     [_period_to_dict(p) for p in world.period],
        "true_theta_trajectories": [
            _trajectory_to_dict(traj)
            for traj in world.true_theta_trajectory_by_part_id.values()
        ],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    abs_path = os.path.abspath(path)
    print(f"[world_io] Saved world → {abs_path}  "
          f"({os.path.getsize(abs_path) / 1024:.1f} KB)")
    return abs_path


def load_world(path: str) -> World:
    """
    Restore a World from a JSON file written by ``save_world()``.

    Callable samplers in WorldSettings are restored as ``None``.
    Re-attach them manually before calling ``step_world_forward()`` again.

    Parameters
    ----------
    path : str
        Path to the ``.json`` file (with or without extension).

    Returns
    -------
    World — fully restored, ready to query or continue stepping.
    """
    if not path.endswith(".json"):
        path = path + ".json"

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    settings = _settings_from_dict(payload["settings"])

    world = World(
        settings         = settings,
        current_period   = payload["current_period"],
        next_supplier_id = payload.get("next_supplier_id", 1),
        next_dispatcher_id= payload.get("next_dispatcher_id", 1),
        next_part_id     = payload.get("next_part_id", 1),
    )

    # Restore RNG state
    if "rng_state" in payload:
        rng = random.Random()
        # random.Random internal state: (version, internalstate_tuple, None)
        rng.setstate((3, tuple(payload["rng_state"]), None))
        world.rng = rng

    # Restore periods
    world.period = [_period_from_dict(p) for p in payload.get("period", [])]

    # Restore trajectories
    for traj_dict in payload.get("true_theta_trajectories", []):
        traj = _trajectory_from_dict(traj_dict)
        world.true_theta_trajectory_by_part_id[traj.part_id] = traj

    print(f"[world_io] Loaded world ← {os.path.abspath(path)}  "
          f"(current_period={world.current_period}, "
          f"periods={len(world.period)}, "
          f"trajectories={len(world.true_theta_trajectory_by_part_id)})")
    return world


# ---------------------------------------------------------------------------
# Public API — World → InputData  (bridge to DataGenerator / ScenarioGenerator)
# ---------------------------------------------------------------------------

# Rep days used in the second codebase
_DEFAULT_REP_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]


def _count_deliveries_per_day(
    part: Part,
    period_id: int,
    rep_days: List[str],
) -> Dict[str, int]:
    """
    Count actual delivery call-offs per representative day in a given period.
    Used as the ``deliveries`` field in the DataGenerator Part schema.
    """
    counts: Dict[str, int] = {d: 0 for d in rep_days}
    for obs in part.delivery_observations:
        if obs.period_id == period_id and obs.delivery_day in counts:
            counts[obs.delivery_day] += 1

    # If no observations recorded for this period (e.g. new part), use
    # the part's delivery_call_off_rate × 1 as a unit estimate per day.
    if all(v == 0 for v in counts.values()):
        rate = max(1, int(round(part.delivery_call_off_rate)))
        for day in part.delivery_day_pattern:
            if day in counts:
                counts[day] = rate

    return counts


def _get_posterior_counts(part: Part, period_id: int) -> Tuple[int, int]:
    """
    Return (n_obs, k_obs) from the Bayesian posterior at the END of period_id.
    n_obs = total call-offs seen up to and including period_id.
    k_obs = total critical call-offs seen up to and including period_id.
    """
    n_obs, k_obs = 0, 0
    for obs in part.delivery_observations:
        if obs.period_id <= period_id:
            n_obs += 1
            if obs.is_critical:
                k_obs += 1
    return n_obs, k_obs


def _infer_fte_factor(dispatcher: Dispatcher, settings: WorldSettings) -> float:
    """
    Back-calculate the FTE factor from target_workload_per_day and settings.
    """
    fulltime = settings.target_workload_dispatcher_fulltime_per_day
    if fulltime <= 0:
        return 1.0
    return round(dispatcher.target_workload_per_day / fulltime, 4)


def world_to_input_data(
    world: World,
    period_t: int,
    period_t1: int,
    rep_days: Optional[List[str]] = None,
    handling_time: float = 30.0,
) -> Dict[str, Any]:
    """
    Convert two consecutive World periods into the ``InputData`` dict format
    expected by ``DataGenerator.generate()`` and ``ScenarioGenerator``.

    The produced dict has the following top-level keys (identical to what
    ``DataGenerator.generate()`` returns via ``InputData``):

        parts              Dict[str, Part-like dict]
        suppliers     Dict[str, Supplier-like dict]   (period t)
        dispatchers   Dict[str, Dispatcher-like dict] (period t)
        rep_days           List[str]
        changeset          Dict with supplier_changes / dispatcher_changes
        meta               Dict  ← extra info (world settings, thetas, etc.)

    Part schema (matches DataGenerator._make_part output)
    ------------------------------------------------------
        part_id    : str   (e.g. "S1_P1")
        supplier_id: str
        n_obs      : int   cumulative critical + non-critical call-offs up to period_t
        k_obs      : int   cumulative critical call-offs up to period_t
        deliveries : Dict[str, int]  {day: expected_deliveries}

    Supplier schema
    ---------------
        supplier_id: str
        parts      : List[str]

    Dispatcher schema
    -----------------
        dispatcher_id  : str
        fte_factor     : float
        target_workload: Dict[str, float]  {day: minutes}

    Parameters
    ----------
    world     : World  (must have both period_t and period_t1 realized)
    period_t  : int    source period  (t)
    period_t1 : int    forecast period (t+1)
    rep_days  : list of day-name strings, default ["Mon","Tue","Wed","Thu","Fri"]
    handling_time : float  minutes per critical call-off (stored in meta)

    Returns
    -------
    dict — InputData-compatible structure.

    Raises
    ------
    ValueError if either period is not found in the world.
    """
    if rep_days is None:
        rep_days = _DEFAULT_REP_DAYS

    p_t  = world.get_period(period_t)
    p_t1 = world.get_period(period_t1)

    if p_t is None:
        raise ValueError(f"period_t={period_t} not found in world.")
    if p_t1 is None:
        raise ValueError(f"period_t1={period_t1} not found in world.")

    settings = world.settings
    base_target = {d: settings.target_workload_dispatcher_fulltime_per_day
                   for d in rep_days}

    # ── Parts (union of both periods, keyed by str(part_id)) ──────────────
    parts: Dict[str, Dict] = {}

    def _register_part(part: Part, supplier_id: int, obs_up_to_period: int) -> None:
        pid_str = str(part.part_id)
        if pid_str in parts:
            return  # already registered
        n_obs, k_obs = _get_posterior_counts(part, obs_up_to_period)

        # No need deliveries (using for generating scenario when solving model)
        # deliveries   = _count_deliveries_per_day(part, obs_up_to_period, rep_days)
        deliveries = {}
        parts[pid_str] = {
            "part_id":     pid_str,
            "supplier_id": str(supplier_id),
            "n_obs":       n_obs,
            "k_obs":       k_obs,
            "deliveries":  deliveries,
            "base_true_theta":  part.base_true_part_theta,
            # Extra fields kept for traceability (not used by ScenarioGenerator)
            "_supplier_theta":   None,  # filled below
            "_is_stationary":    part.is_true_theta_stationary,
        }

    # ── suppliers  (period t) ────────────────────────────────────────
    suppliers: Dict[str, Dict] = {}
    for sup in p_t.supplier:
        sid_str   = str(sup.supplier_id)
        part_ids  = []
        for part in sup.part:
            if part.is_active_in_period(period_t):
                _register_part(part, sup.supplier_id, period_t)
                pid_str = str(part.part_id)
                parts[pid_str]["_supplier_theta"] = sup.supplier_theta
                part_ids.append(pid_str)
        suppliers[sid_str] = {
            "supplier_id":    sid_str,
            "parts":          part_ids,
            "base_true_theta": sup.supplier_theta,
            "_risk_factor":   sup.supplier_risk_factor,
        }

    # ── dispatchers  (period t) ─────────────────────────────────────
    dispatchers: Dict[str, Dict] = {}
    for disp in p_t.dispatcher:
        did_str = str(disp.dispatcher_id)
        dispatchers[did_str] = {
            "dispatcher_id":  did_str,
            "fte_factor":     _infer_fte_factor(disp, settings),
            "target_workload": {d: disp.target_workload_per_day for d in rep_days},
            "_employment_type": disp.employment_type,
        }


    # ── Metadata (extra context not in the original DataGenerator schema) ──
    meta = {
        "source_world_period_t":  period_t,
        "source_world_period_t1": period_t1,
        "handling_time_minutes":  handling_time,
        "use_hierarchical_theta": settings.use_hierarchical_theta,
        "m_global":  settings.m_global,
        "kappa_s":   settings.kappa_s,
        "kappa_p":   settings.kappa_p,
    }

    return {
        "parts":            parts,
        "suppliers":   suppliers,
        "dispatchers": dispatchers,
        "rep_days":         rep_days,
        "meta":             meta,
    }


# ---------------------------------------------------------------------------
# Public API — Validate / pass-through for ScenarioGenerator
# ---------------------------------------------------------------------------

def to_scenario_generator_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and return an InputData dict ready for ScenarioGenerator.

    ScenarioGenerator only uses:
        - input_data["parts"]          Dict[str, {part_id, n_obs, k_obs, deliveries}]
        - input_data["suppliers"] Dict[str, {supplier_id, parts: List[str]}]
        - input_data["rep_days"]       List[str]

    This function verifies those keys exist and that every part referenced by
    suppliers is present in parts.  It returns the same dict unchanged
    (no copy) so it can be passed directly to ScenarioGenerator().

    Parameters
    ----------
    input_data : dict  as returned by world_to_input_data() or DataGenerator.generate()

    Returns
    -------
    dict — the same input_data (validated in-place).

    Raises
    ------
    KeyError / ValueError on validation failures.
    """
    required_top = {"parts", "suppliers", "rep_days"}
    missing = required_top - set(input_data.keys())
    if missing:
        raise KeyError(f"InputData missing required keys: {missing}")

    parts          = input_data["parts"]
    suppliers = input_data["suppliers"]
    rep_days       = input_data["rep_days"]

    if not rep_days:
        raise ValueError("rep_days must not be empty.")

    missing_parts: List[str] = []
    for sid, sup in suppliers.items():
        for pid in sup.get("parts", []):
            if pid not in parts:
                missing_parts.append(pid)

    if missing_parts:
        raise ValueError(
            f"suppliers references {len(missing_parts)} part(s) not found in parts dict. "
            f"First missing: {missing_parts[:5]}"
        )

    # Verify each part has the fields ScenarioGenerator._compute_posterior needs
    for pid, part in parts.items():
        if "n_obs" not in part or "k_obs" not in part:
            raise ValueError(f"Part '{pid}' missing 'n_obs' or 'k_obs'.")
        if "deliveries" not in part:
            raise ValueError(f"Part '{pid}' missing 'deliveries' dict.")

    n_parts = sum(len(s.get("parts", [])) for s in suppliers.values())
    print(
        f"[world_io] InputData validated: "
        f"{len(suppliers)} suppliers, "
        f"{n_parts} parts in scope, "
        f"rep_days={rep_days}"
    )
    return input_data

# ---------------------------------------------------------------------------
# Scenario mapping: TP3 list[dict]  ↔  ScenarioSet (codebase mới)
# ---------------------------------------------------------------------------
 
 
 
def tp3_scenarios_to_scenario_set(
    tp3_scenarios: List[Dict[str, Any]],
    input_data: Dict[str, Any],
    handling_time: float = 30.0,
    alpha0: float = 1.0,
    beta0: float = 1.0,
    rng_seed: int = 0,
) -> Dict[str, Any]:
    """
    Map output của ``create_scenarios_for_current_period()`` (TP3) sang cấu
    trúc ``ScenarioSet`` (codebase mới).
 
    TP3 output layout
    -----------------
    tp3_scenarios[omega] = {
        scenario_id, source_period_id, forecast_period_id,
        forecast_period_start_date, forecast_period_dates, days,
        supplier_ids, dispatcher_ids,
        delivery_performance_by_part: { part_id(int) → {
            sampled_theta, posterior_alpha, posterior_beta,
            n_critical_delivery_calloffs, n_actual_delivery_calloffs,
            delivery_observations: [{delivery_day, is_critical, ...}]
        }},
        workload_by_part:     { part_id(int) → {daily_workload_minutes, ...} },
        workload_by_supplier: { supplier_id(int) → {daily_workload_minutes, ...} },
    }
 
    ScenarioSet layout (target)
    ---------------------------
    theta_records    : [{part_id(str), scenario_index(int), theta_sampled(float)}]
    critical_records : [{part_id(str), supplier_id(str), day(str),
                         scenario_index(int), n_deliveries(int),
                         critical_count(int), workload_contribution_min(float)}]
    part_stats       : [{part_id, supplier_id, day, n_deliveries,
                         mean/std/p5/p50/p95_critical, mean_workload_min}]
    supplier_stats   : [{supplier_id, day,
                         mean/std/p5/p50/p95_workload_min}]
 
    Mapping rules
    -------------
    - part_id   : int (TP3) → str (mới)
    - supplier_id: int (TP3) → str (mới) — resolved via input_data part→supplier map
    - day       : delivery_day string ("Mon"…"Fri") kept as-is
    - critical_count per (part, day, omega):
          sum of is_critical observations matching that day in delivery_performance_by_part
    - n_deliveries per (part, day):
          taken from input_data["parts"][pid]["deliveries"][day]
          fallback: count of all observations on that day across scenarios (mode)
    - workload_contribution_min : critical_count × handling_time
 
    Parameters
    ----------
    tp3_scenarios   : list[dict] from create_scenarios_for_current_period()
    input_data      : dict from world_to_input_data()  (needed for n_deliveries)
    handling_time   : float  minutes per critical call-off
    alpha0, beta0   : Beta prior hyperparameters (for config field only)
    rng_seed        : int (for config field only)
 
    Returns
    -------
    dict matching ScenarioSet.model_dump() — ready to pass to ScenarioSet.model_validate()
    or to inspect directly.
    """
    if not tp3_scenarios:
        raise ValueError("tp3_scenarios is empty.")
 
    
 
    rep_days: List[str] = input_data.get("rep_days", ["Mon", "Tue", "Wed", "Thu", "Fri"])
    parts_dict: Dict[str, Any] = input_data.get("parts", {})
 
    # Build part → supplier mapping from input_data (str keys)
    part_to_supplier: Dict[str, str] = {}
    for sid, sup in input_data.get("suppliers", {}).items():
        for pid in sup.get("parts", []):
            part_to_supplier[str(pid)] = str(sid)
 
    n_scenarios = len(tp3_scenarios)
    source_period_id   = tp3_scenarios[0]["source_period_id"]
    forecast_period_id = tp3_scenarios[0]["forecast_period_id"]
 
    theta_records:    List[Dict] = []
    critical_records: List[Dict] = []
 
    for omega, scenario in enumerate(tp3_scenarios):
        perf_by_part = scenario.get("delivery_performance_by_part", {})
 
        for part_id_raw, perf in perf_by_part.items():
            pid_str = str(part_id_raw)
            sid_str = part_to_supplier.get(pid_str, "unknown")
 
            # ── theta record ───────────────────────────────────────────────
            theta_records.append({
                "part_id":        pid_str,
                "scenario_index": omega,
                "posterior_alpha": round(float(perf.get("posterior_alpha", -999)), 6),
                "posterior_beta":  round(float(perf.get("posterior_beta", -999)), 6),
                "theta_sampled":  round(float(perf.get("sampled_theta", -999)), 6),
                "true_theta": round(float(perf.get("true_theta", -999)), 6),
            })
 
            # ── critical count per day ─────────────────────────────────────
            # Count is_critical observations grouped by delivery_day
            day_criticals: Dict[str, int] = {d: 0 for d in rep_days}
            day_totals:    Dict[str, int] = {d: 0 for d in rep_days}
 
            for obs in perf.get("delivery_observations", []):
                day = obs.get("delivery_day")
                if day in day_criticals:
                    day_totals[day]    += 1
                    if obs.get("is_critical"):
                        day_criticals[day] += 1
 
            for tau in rep_days:
                # Not using input data
                # n_deliveries: prefer input_data, fall back to observed count
                # n_deliveries = (
                #     parts_dict.get(pid_str, {})
                #     .get("deliveries", {})
                #     .get(tau, day_totals[tau])
                # )
                n_deliveries = day_totals[tau]
 
                critical_count      = day_criticals[tau]
                workload_contribution = handling_time * critical_count
 
                critical_records.append({
                    "part_id":                    pid_str,
                    "supplier_id":                sid_str,
                    "day":                        tau,
                    "scenario_index":             omega,
                    "n_deliveries":               int(n_deliveries),
                    "critical_count":             critical_count,
                    "workload_contribution_min":  workload_contribution,
                })
 
    # ── Aggregate part_stats ───────────────────────────────────────────────
    bucket_part:  Dict[tuple, List[int]]   = {}
    meta_part:    Dict[tuple, tuple]       = {}
    for r in critical_records:
        key = (r["part_id"], r["day"])
        bucket_part.setdefault(key, []).append(r["critical_count"])
        meta_part[key] = (r["supplier_id"], r["n_deliveries"])
 
    part_stats: List[Dict] = []
    for (pid, tau) in sorted(bucket_part):
        arr = np.array(bucket_part[(pid, tau)], dtype=float)
        sid, n_del = meta_part[(pid, tau)]
        part_stats.append({
            "part_id":         pid,
            "supplier_id":     sid,
            "day":             tau,
            "n_deliveries":    n_del,
            "mean_critical":   round(float(np.mean(arr)),          4),
            "std_critical":    round(float(np.std(arr)),           4),
            "p5_critical":     round(float(np.percentile(arr,  5)), 4),
            "p50_critical":    round(float(np.percentile(arr, 50)), 4),
            "p95_critical":    round(float(np.percentile(arr, 95)), 4),
            "mean_workload_min": round(float(np.mean(arr)) * handling_time, 4),
        })
 
    # ── Aggregate supplier_stats ───────────────────────────────────────────
    # Re-aggregate properly:
    workload_sup_bucket: Dict[tuple, Dict[int, float]] = {}
    for r in critical_records:
        key = (r["supplier_id"], r["day"])
        omega_map = workload_sup_bucket.setdefault(key, {})
        omega_map[r["scenario_index"]] = (
            omega_map.get(r["scenario_index"], 0.0) + r["workload_contribution_min"]
        )
 
    supplier_stats: List[Dict] = []
    for (sid, tau) in sorted(workload_sup_bucket):
        arr = np.array(list(workload_sup_bucket[(sid, tau)].values()), dtype=float)
        supplier_stats.append({
            "supplier_id":         sid,
            "day":                 tau,
            "mean_workload_min":   round(float(np.mean(arr)),          4),
            "std_workload_min":    round(float(np.std(arr)),           4),
            "p5_workload_min":     round(float(np.percentile(arr,  5)), 4),
            "p50_workload_min":    round(float(np.percentile(arr, 50)), 4),
            "p95_workload_min":    round(float(np.percentile(arr, 95)), 4),
        })
 
    # ── Infer n_parts from suppliers ─────────────────────────────────
    n_parts = len(set(r["part_id"] for r in theta_records))
 
    return {
        "scenario_set_id": str(uuid4()),
        "data_id":         input_data.get("meta", {}).get("data_id", "tp3_world"),
        "config": {
            "n_scenarios":   n_scenarios,
            "handling_time": handling_time,
            "alpha0":        alpha0,
            "beta0":         beta0,
            "rng_seed":      rng_seed,
        },
        "created_at":       datetime.now(timezone.utc).isoformat(),
        "n_parts":          n_parts,
        "n_rep_days":       len(rep_days),
        "theta_records":    theta_records,
        "critical_records": critical_records,
        "part_stats":       part_stats,
        "supplier_stats":   supplier_stats,
        # Traceability
        "_source":          "tp3_scenarios_to_scenario_set",
        "_source_period_id":   source_period_id,
        "_forecast_period_id": forecast_period_id,
    }


def tp3_scenarios_chunk_to_scenario_set(
    tp3_scenarios,
    input_data,
    handling_time=30.0,
    scenario_index_offset=0,
):
    if not tp3_scenarios:
        raise ValueError("tp3_scenarios is empty.")

    rep_days = input_data.get(
        "rep_days",
        ["Mon", "Tue", "Wed", "Thu", "Fri"]
    )

    # Build part -> supplier mapping
    part_to_supplier = {}

    for sid, sup in input_data.get("suppliers", {}).items():
        for pid in sup.get("parts", []):
            part_to_supplier[str(pid)] = str(sid)

    n_scenarios = len(tp3_scenarios)

    source_period_id = tp3_scenarios[0]["source_period_id"]
    forecast_period_id = tp3_scenarios[0]["forecast_period_id"]

    theta_records = []
    critical_records = []

    # ============================================================
    # PROCESS ONLY THIS CHUNK
    # ============================================================

    for local_omega, scenario in enumerate(tp3_scenarios):

        omega = scenario_index_offset + local_omega

        perf_by_part = scenario.get(
            "delivery_performance_by_part",
            {}
        )

        for part_id_raw, perf in perf_by_part.items():

            pid_str = str(part_id_raw)

            sid_str = part_to_supplier.get(
                pid_str,
                "unknown"
            )

            # ----------------------------------------------------
            # theta
            # ----------------------------------------------------

            theta_records.append({
                "part_id": pid_str,
                "scenario_index": omega,
                "posterior_alpha": round(
                    float(perf.get("posterior_alpha", -999)),
                    6,
                ),
                "posterior_beta": round(
                    float(perf.get("posterior_beta", -999)),
                    6,
                ),
                "theta_sampled": round(
                    float(perf.get("sampled_theta", -999)),
                    6,
                ),
                "true_theta": round(
                    float(perf.get("true_theta", -999)),
                    6,
                ),
            })

            # ----------------------------------------------------
            # critical count
            # ----------------------------------------------------

            day_criticals = {
                d: 0
                for d in rep_days
            }

            day_totals = {
                d: 0
                for d in rep_days
            }

            for obs in perf.get(
                "delivery_observations",
                []
            ):
                day = obs.get("delivery_day")

                if day in day_criticals:

                    day_totals[day] += 1

                    if obs.get("is_critical"):
                        day_criticals[day] += 1

            for tau in rep_days:

                n_deliveries = day_totals[tau]

                critical_count = day_criticals[tau]

                workload_contribution = (
                    handling_time * critical_count
                )

                critical_records.append({
                    "part_id": pid_str,
                    "supplier_id": sid_str,
                    "day": tau,
                    "scenario_index": omega,
                    "n_deliveries": int(n_deliveries),
                    "critical_count": critical_count,
                    "workload_contribution_min":
                        workload_contribution,
                })

    return {
        "theta_records": theta_records,
        "critical_records": critical_records,
    }


 
def scenario_set_to_tp3_scenarios(
    scenario_set: Dict[str, Any],
    world: "World",
    period_t: int,
    rep_days: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Map một ``ScenarioSet`` dict sang layout ``create_scenarios_for_current_period()``
    của TP3 (list[dict], scenario-first).
 
    Dùng khi muốn đưa ScenarioSet sinh từ codebase mới vào pipeline TP3 cũ
    (ví dụ: feed vào MIP model của TP3 đang dùng workload_by_supplier).
 
    Reconstruction rules
    --------------------
    theta_records + critical_records → delivery_performance_by_part + workload_by_part
    workload_by_part (sum over parts per supplier) → workload_by_supplier
 
    Các fields phụ thuộc World được rebuild từ:
        - world.get_period(period_t)  → supplier_ids, dispatcher_ids
        - world.settings              → forecast_period_start_date, forecast_period_dates
 
    delivery_observations list:
        Rebuilt từ critical_records: mỗi record (part, day, omega) sinh ra
        n_deliveries entries trên day đó, với critical_count entries is_critical=True
        và phần còn lại is_critical=False.  observation_date dùng ngày đầu tiên
        của day trong forecast period.
 
    Parameters
    ----------
    scenario_set  : dict (from tp3_scenarios_to_scenario_set() hoặc ScenarioSet.model_dump())
    world         : World object (để lấy structure period)
    period_t      : int  source period id
    rep_days      : list[str]  default ["Mon"…"Fri"]
 
    Returns
    -------
    list[dict]  — cùng schema với create_scenarios_for_current_period()
    """
    if rep_days is None:
        rep_days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
 
    settings = world.settings
    period_t1 = period_t + 1
 
    p_t = world.get_period(period_t)
    if p_t is None:
        raise ValueError(f"period_t={period_t} not found in world.")
 
    # Forecast period dates (date objects, keyed by day-name for lookup)
    forecast_period_start = settings.simulation_start_date + timedelta(
        weeks=period_t1 * settings.weeks_per_period
    )
    forecast_period_dates: List[date] = []
    day_to_first_date:     Dict[str, date] = {}
    for week_idx in range(settings.weeks_per_period):
        week_start = forecast_period_start + timedelta(weeks=week_idx)
        for day in rep_days:
            d = week_start + timedelta(days=WEEKDAY_TO_OFFSET[day])
            forecast_period_dates.append(d)
            if day not in day_to_first_date:
                day_to_first_date[day] = d
 
    supplier_ids    = [s.supplier_id for s in p_t.supplier]
    dispatcher_ids  = [d.dispatcher_id for d in p_t.dispatcher]
 
    # Index theta_records by (part_id, omega)
    theta_index: Dict[tuple, float] = {
        (r["part_id"], r["scenario_index"]): r["theta_sampled"]
        for r in scenario_set.get("theta_records", [])
    }
 
    # Index critical_records by (part_id, day, omega)
    crit_index: Dict[tuple, Dict] = {
        (r["part_id"], r["day"], r["scenario_index"]): r
        for r in scenario_set.get("critical_records", [])
    }
 
    # Collect posterior alpha/beta per part (from first scenario's theta only —
    # we re-derive them from the Beta sufficient statistics across scenarios)
    # Approximate: mean=alpha/(alpha+beta), var=alpha*beta/((alpha+beta)^2*(alpha+beta+1))
    # We store (1,1) as fallback — sufficient for workload models that don't re-use posteriors.
    posterior_by_part: Dict[str, tuple] = {}
    theta_by_part: Dict[str, List[float]] = {}
    for r in scenario_set.get("theta_records", []):
        theta_by_part.setdefault(r["part_id"], []).append(r["theta_sampled"])
    for pid, thetas in theta_by_part.items():
        arr = np.array(thetas)
        m, v = float(np.mean(arr)), float(np.var(arr))
        if v > 0:
            kappa = m * (1 - m) / v - 1
            alpha = max(0.01, m * kappa)
            beta  = max(0.01, (1 - m) * kappa)
        else:
            alpha, beta = 1.0, 1.0
        posterior_by_part[pid] = (round(alpha, 4), round(beta, 4))
 
    n_scenarios    = scenario_set.get("config", {}).get("n_scenarios", max(
        (r["scenario_index"] for r in scenario_set.get("theta_records", [{"scenario_index": 0}])),
        default=0
    ) + 1)
    handling_time  = scenario_set.get("config", {}).get("handling_time", 30.0)
    part_ids_all   = sorted(theta_by_part.keys())
 
    scenarios_out: List[Dict[str, Any]] = []
 
    for omega in range(n_scenarios):
        delivery_performance_by_part: Dict[Any, Dict] = {}
        workload_by_part:             Dict[Any, Dict] = {}
 
        for pid_str in part_ids_all:
            pid_int_or_str: Any = pid_str
            # Try to restore int key to match TP3 convention
            try:
                pid_int_or_str = int(pid_str)
            except (ValueError, TypeError):
                pass
 
            theta   = theta_index.get((pid_str, omega), 0.0)
            alpha_p, beta_p = posterior_by_part.get(pid_str, (1.0, 1.0))
 
            # Rebuild delivery_observations from critical_records
            delivery_observations: List[Dict] = []
            n_possible = 0
            n_actual   = 0
            n_critical = 0
            n_noncrit  = 0
 
            daily_workload: Dict[date, float] = {d: 0.0 for d in forecast_period_dates}
 
            for tau in rep_days:
                rec = crit_index.get((pid_str, tau, omega))
                if rec is None:
                    continue
 
                n_del = rec["n_deliveries"]
                n_crit_day = rec["critical_count"]
                n_noncrit_day = n_del - n_crit_day
                obs_date = day_to_first_date.get(tau, forecast_period_start)
 
                n_possible += n_del
                n_actual   += n_del
                n_critical += n_crit_day
                n_noncrit  += n_noncrit_day
 
                for _ in range(n_crit_day):
                    delivery_observations.append({
                        "period_id":        period_t1,
                        "delivery_day":     tau,
                        "observation_date": obs_date,
                        "is_critical":      True,
                    })
                for _ in range(n_noncrit_day):
                    delivery_observations.append({
                        "period_id":        period_t1,
                        "delivery_day":     tau,
                        "observation_date": obs_date,
                        "is_critical":      False,
                    })
 
                if obs_date in daily_workload:
                    daily_workload[obs_date] += handling_time * n_crit_day
 
            total_workload = sum(daily_workload.values())
 
            delivery_performance_by_part[pid_int_or_str] = {
                "scenario_id":                   omega,
                "part_id":                       pid_int_or_str,
                "source_period_id":              period_t,
                "forecast_period_id":            period_t1,
                "forecast_period_start_date":    forecast_period_start,
                "forecast_period_dates":         list(forecast_period_dates),
                "posterior_alpha":               alpha_p,
                "posterior_beta":                beta_p,
                "sampled_theta":                 theta,
                "possible_delivery_calloffs":    [],   # not reconstructed
                "n_possible_delivery_calloffs":  n_possible,
                "n_actual_delivery_calloffs":    n_actual,
                "n_critical_delivery_calloffs":  n_critical,
                "n_non_critical_delivery_calloffs": n_noncrit,
                "delivery_observations":         delivery_observations,
            }
 
            workload_by_part[pid_int_or_str] = {
                "scenario_id":               omega,
                "part_id":                   pid_int_or_str,
                "source_period_id":          period_t,
                "forecast_period_id":        period_t1,
                "forecast_period_start_date": forecast_period_start,
                "forecast_period_dates":     list(forecast_period_dates),
                "posterior_alpha":           alpha_p,
                "posterior_beta":            beta_p,
                "sampled_theta":             theta,
                "critical_case_duration_minutes": handling_time,
                "possible_delivery_calloffs": [],
                "delivery_observations":     delivery_observations,
                "daily_workload_minutes":    daily_workload,
                "total_workload_minutes":    total_workload,
                "n_possible_delivery_calloffs": n_possible,
                "n_actual_delivery_calloffs":   n_actual,
                "n_critical_delivery_calloffs": n_critical,
                "n_non_critical_delivery_calloffs": n_noncrit,
            }
 
        # Supplier-level aggregation
        workload_by_supplier: Dict[Any, Dict] = {}
        for sup in p_t.supplier:
            sup_daily: Dict[date, float] = {d: 0.0 for d in forecast_period_dates}
            sup_total = 0.0
            sup_part_ids = []
 
            for part in sup.part:
                pid_key: Any = part.part_id
                entry = workload_by_part.get(pid_key)
                if entry is None:
                    pid_key = str(part.part_id)
                    entry = workload_by_part.get(pid_key)
                if entry is None:
                    continue
 
                sup_part_ids.append(pid_key)
                for day_date, wl in entry["daily_workload_minutes"].items():
                    sup_daily[day_date] = sup_daily.get(day_date, 0.0) + float(wl)
                sup_total += float(entry["total_workload_minutes"])
 
            workload_by_supplier[sup.supplier_id] = {
                "scenario_id":               omega,
                "supplier_id":               sup.supplier_id,
                "source_period_id":          period_t,
                "forecast_period_id":        period_t1,
                "forecast_period_start_date": forecast_period_start,
                "forecast_period_dates":     list(forecast_period_dates),
                "daily_workload_minutes":    sup_daily,
                "total_workload_minutes":    sup_total,
                "part_ids":                  sup_part_ids,
            }
 
        scenarios_out.append({
            "scenario_id":               omega,
            "source_period_id":          period_t,
            "forecast_period_id":        period_t1,
            "forecast_period_start_date": forecast_period_start,
            "forecast_period_dates":     list(forecast_period_dates),
            "days":                      list(rep_days),
            "supplier_ids":              list(supplier_ids),
            "dispatcher_ids":            list(dispatcher_ids),
            "delivery_performance_by_part": delivery_performance_by_part,
            "workload_by_part":          workload_by_part,
            "workload_by_supplier":      workload_by_supplier,
        })
 
    return scenarios_out

# ---------------------------------------------------------------------------
# Save / Load TP3 scenarios  (list[dict] from create_scenarios_for_current_period)
# ---------------------------------------------------------------------------
 
def _json_default(obj: Any) -> Any:
    """
    JSON serialiser for types not handled by the stdlib encoder.
 
    Handles:
        date, datetime  → ISO string  "YYYY-MM-DD" / "YYYY-MM-DDTHH:MM:SS"
        numpy scalars   → Python int / float
        numpy ndarray   → list
    """
    if isinstance(obj, (date, )):
        return obj.isoformat()
    try:
        from datetime import datetime as _dt
        if isinstance(obj, _dt):
            return obj.isoformat()
    except ImportError:
        pass
    try:
        import numpy as _np
        if isinstance(obj, _np.integer):
            return int(obj)
        if isinstance(obj, _np.floating):
            return float(obj)
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
 
 
def _json_object_hook(d: dict) -> dict:
    """
    Restore date strings back to ``datetime.date`` objects during JSON load.
 
    Keys that are date ISO strings are NOT converted — only *values* that
    match the date pattern ``YYYY-MM-DD`` are restored, because the TP3
    scenario dicts use ``datetime.date`` objects as *dict keys* in
    ``daily_workload_minutes``.
 
    Dict keys that look like dates are restored separately in
    ``_restore_date_keys()``.
    """
    return d

def _restore_date_keys(obj: Any) -> Any:
    """
    Recursively walk the decoded JSON object and convert any string key that
    looks like a date (``YYYY-MM-DD``) back to ``datetime.date``.
 
    Also converts string values that look like dates inside known date-value
    fields (``forecast_period_start_date``, ``observation_date``).
    """
    import re
    _DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    _DATE_VALUE_KEYS = {
        "forecast_period_start_date", "observation_date",
    }
 
    if isinstance(obj, dict):
        new: dict = {}
        for k, v in obj.items():
            # Restore date-string keys (used in daily_workload_minutes)
            new_k = date.fromisoformat(k) if isinstance(k, str) and _DATE_RE.match(k) else k
            # Restore date-string values for known fields
            if isinstance(v, str) and _DATE_RE.match(v) and k in _DATE_VALUE_KEYS:
                new[new_k] = date.fromisoformat(v)
            else:
                new[new_k] = _restore_date_keys(v)
        return new
    if isinstance(obj, list):
        return [_restore_date_keys(item) for item in obj]
    # Bare string that looks like a date (forecast_period_dates list elements)
    if isinstance(obj, str) and len(obj) == 10:
        try:
            return date.fromisoformat(obj)
        except ValueError:
            pass
    return obj

def _prepare_for_json(obj: Any) -> Any:
    """
    Recursively convert all non-JSON-native types in the TP3 scenario structure.
 
    JSON forbids non-string dict keys, so ``datetime.date`` keys in
    ``daily_workload_minutes`` must be converted to ISO strings before encoding.
    Values that are ``datetime.date`` objects are also converted.
    numpy scalars / ndarrays are converted to native Python types.
    """
    if isinstance(obj, dict):
        new: dict = {}
        for k, v in obj.items():
            # Convert date keys to ISO strings
            new_k = k.isoformat() if isinstance(k, date) else k
            new[new_k] = _prepare_for_json(v)
        return new
    if isinstance(obj, list):
        return [_prepare_for_json(item) for item in obj]
    if isinstance(obj, date):
        return obj.isoformat()
    try:
        from datetime import datetime as _dt
        if isinstance(obj, _dt):
            return obj.isoformat()
    except ImportError:
        pass
    try:
        import numpy as _np
        if isinstance(obj, _np.integer):
            return int(obj)
        if isinstance(obj, _np.floating):
            return float(obj)
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    return obj
 
 
def save_tp3_scenarios(
    scenarios: List[Dict[str, Any]],
    path: str,
    indent: int = 2,
) -> str:
    """
    Serialise TP3 scenario list to a JSON file, handling all non-serialisable
    types that appear in ``create_scenarios_for_current_period()`` output:
 
        - ``datetime.date`` objects as dict *keys* (daily_workload_minutes)
          and as values (forecast_period_dates, observation_date, …)
        - numpy scalars / arrays (if present)
 
    Parameters
    ----------
    scenarios : list[dict]
        Output of ``create_scenarios_for_current_period()``.
    path : str
        Target file path.  ``.json`` appended if absent.
    indent : int
        JSON indentation (default 2).
 
    Returns
    -------
    str — absolute path of the written file.
    """
    import json as _json
 
    if not path.endswith(".json"):
        path = path + ".json"
 
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
 
    # Pre-process: convert all date keys and values to strings
    serialisable = _prepare_for_json(scenarios)
 
    with open(path, "w", encoding="utf-8") as f:
        _json.dump(serialisable, f, indent=indent, ensure_ascii=False)
 
    abs_path = os.path.abspath(path)
    size_kb  = os.path.getsize(abs_path) / 1024
    print(f"[world_io] Saved {len(scenarios)} scenarios → {abs_path}  ({size_kb:.1f} KB)")
    return abs_path
 
 
def load_tp3_scenarios(path: str) -> List[Dict[str, Any]]:
    """
    Load TP3 scenarios from a JSON file written by ``save_tp3_scenarios()``.
 
    All date strings are restored to ``datetime.date`` objects — including
    dict *keys* (``daily_workload_minutes``) and list elements
    (``forecast_period_dates``).
 
    Parameters
    ----------
    path : str
        Path to the ``.json`` file (with or without extension).
 
    Returns
    -------
    list[dict] — same structure as ``create_scenarios_for_current_period()`` output.
    """
    import json as _json
 
    if not path.endswith(".json"):
        path = path + ".json"
 
    with open(path, "r", encoding="utf-8") as f:
        raw = _json.load(f)
 
    restored = _restore_date_keys(raw)
 
    n = len(restored)
    print(f"[world_io] Loaded {n} scenarios ← {os.path.abspath(path)}")
    return restored


def tp3_average_scenario_to_scenario_set(
    average_scenario: Dict[str, Any],
    input_data: Dict[str, Any],
    handling_time: float = 30.0,
    alpha0: float = 1.0,
    beta0: float = 1.0,
    rng_seed: int = 0,
) -> Dict[str, Any]:
    """
    Biến thể của tp3_scenarios_to_scenario_set dành riêng cho scenario
    trung bình (deterministic, critical_weight liên tục thay vì is_critical
    boolean). Luôn tạo ra ScenarioSet với đúng 1 scenario (scenario_index=0).
    """
    rep_days: List[str] = input_data.get("rep_days", ["Mon", "Tue", "Wed", "Thu", "Fri"])

    part_to_supplier: Dict[str, str] = {}
    for sid, sup in input_data.get("suppliers", {}).items():
        for pid in sup.get("parts", []):
            part_to_supplier[str(pid)] = str(sid)

    theta_records:    List[Dict] = []
    critical_records: List[Dict] = []

    perf_by_part = average_scenario.get("delivery_performance_by_part", {})

    for part_id_raw, perf in perf_by_part.items():
        pid_str = str(part_id_raw)
        sid_str = part_to_supplier.get(pid_str, "unknown")

        theta_records.append({
            "part_id":         pid_str,
            "scenario_index":  0,
            "posterior_alpha": round(float(perf["posterior_alpha"]), 6),
            "posterior_beta":  round(float(perf["posterior_beta"]), 6),
            "theta_sampled":   round(float(perf["sampled_theta"]), 6),  # = theta_mean
        })

        # Tổng critical_weight theo từng ngày (kỳ vọng liên tục, không phải đếm)
        day_criticals: Dict[str, float] = {d: 0.0 for d in rep_days}
        day_totals:    Dict[str, float] = {d: 0.0 for d in rep_days}

        for obs in perf.get("delivery_observations", []):
            day = obs.get("delivery_day")
            if day in day_criticals:
                day_totals[day]    += obs.get("delivery_weight", 1.0)
                day_criticals[day] += obs.get("critical_weight", 0.0)

        for tau in rep_days:
            critical_count = day_criticals[tau]          # float kỳ vọng
            workload_contribution = handling_time * critical_count

            critical_records.append({
                "part_id":                   pid_str,
                "supplier_id":               sid_str,
                "day":                       tau,
                "scenario_index":            0,
                "n_deliveries":              day_totals[tau],   # float kỳ vọng
                "critical_count":            critical_count,     # float, KHÔNG int()
                "workload_contribution_min": workload_contribution,
            })

    # part_stats / supplier_stats: với 1 scenario duy nhất, mean=std=0,
    # percentile đều = giá trị đó — giữ nguyên format để code downstream không vỡ.
    part_stats: List[Dict] = []
    bucket: Dict[tuple, tuple] = {}
    for r in critical_records:
        bucket[(r["part_id"], r["day"])] = (r["supplier_id"], r["n_deliveries"], r["critical_count"])
    for (pid, tau), (sid, n_del, cc) in sorted(bucket.items()):
        part_stats.append({
            "part_id": pid, "supplier_id": sid, "day": tau, "n_deliveries": n_del,
            "mean_critical": round(cc, 4), "std_critical": 0.0,
            "p5_critical": round(cc, 4), "p50_critical": round(cc, 4), "p95_critical": round(cc, 4),
            "mean_workload_min": round(cc * handling_time, 4),
        })

    sup_bucket: Dict[tuple, float] = {}
    for r in critical_records:
        key = (r["supplier_id"], r["day"])
        sup_bucket[key] = sup_bucket.get(key, 0.0) + r["workload_contribution_min"]
    supplier_stats: List[Dict] = []
    for (sid, tau), wl in sorted(sup_bucket.items()):
        supplier_stats.append({
            "supplier_id": sid, "day": tau,
            "mean_workload_min": round(wl, 4), "std_workload_min": 0.0,
            "p5_workload_min": round(wl, 4), "p50_workload_min": round(wl, 4), "p95_workload_min": round(wl, 4),
        })

    return {
        "scenario_set_id": str(uuid4()),
        "data_id": input_data.get("meta", {}).get("data_id", "tp3_world_average"),
        "config": {
            "n_scenarios": 1, "handling_time": handling_time,
            "alpha0": alpha0, "beta0": beta0, "rng_seed": rng_seed,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_parts": len(theta_records),
        "n_rep_days": len(rep_days),
        "theta_records": theta_records,
        "critical_records": critical_records,
        "part_stats": part_stats,
        "supplier_stats": supplier_stats,
        "_source": "tp3_average_scenario_to_scenario_set",
        "_source_period_id":   average_scenario.get("source_period_id"),
        "_forecast_period_id": average_scenario.get("forecast_period_id"),
    }