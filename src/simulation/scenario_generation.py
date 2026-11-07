from __future__ import annotations

from datetime import timedelta, date
import random
import numpy as np
from .simulation_entities import World, WEEKDAY_TO_OFFSET


def _build_forecast_period_dates(
    forecast_period_start_date: date,
    weeks_per_period: int,
) -> list[date]:
    """
    Build all weekday dates of the forecast period in chronological order.
    """
    forecast_period_dates: list[date] = []

    for week_idx in range(weeks_per_period):
        week_start = forecast_period_start_date + timedelta(weeks=week_idx)
        for day in ["Mon", "Tue", "Wed", "Thu", "Fri"]:
            forecast_period_dates.append(
                week_start + timedelta(days=WEEKDAY_TO_OFFSET[day])
            )

    return forecast_period_dates


def _build_possible_delivery_dates_from_part_settings(
    part,
    forecast_period_id: int,
    forecast_period_start_date: date,
    weeks_per_period: int,
) -> list[dict]:
    """
    Build possible delivery call-offs for the next period from the part settings.
    """
    possible_delivery_calloffs: list[dict] = []

    for week_idx in range(weeks_per_period):
        week_start = forecast_period_start_date + timedelta(weeks=week_idx)

        for day in part.delivery_day_pattern:
            observation_date = week_start + timedelta(days=WEEKDAY_TO_OFFSET[day])

            possible_delivery_calloffs.append(
                {
                    "period_id": forecast_period_id,
                    "delivery_day": day,
                    "observation_date": observation_date,
                }
            )

    return possible_delivery_calloffs


def create_delivery_performance_scenarios_for_part(
    world: World,
    part_id: int,
    source_period_id: int | None = None,
    n_scenarios: int = 10,
    random_seed: int | None = None,
    next_period_delivery_calloffs: list[dict] | None = None,
) -> list[dict]:
    """
    Create posterior-based delivery performance scenarios for one part
    for the next period.

    Returns one list entry per scenario.

    Note
    ----
    That constraint is applied at the world-level in
    create_scenarios_for_current_period, where the full pool of
    actual call-offs across all parts is visible.
    """
    if n_scenarios <= 0:
        raise ValueError("n_scenarios must be > 0.")

    if source_period_id is None:
        source_period_id = world.current_period

    if source_period_id < 0:
        raise ValueError("source_period_id must be >= 0.")

    forecast_period_id = source_period_id + 1
    if forecast_period_id >= world.settings.n_simulation_periods:
        raise ValueError("No forecast period available.")

    part = world.get_part_in_period(period_id=source_period_id, part_id=part_id)
    if part is None:
        raise ValueError(f"Part {part_id} not found in period {source_period_id}.")

    source_state = part.get_period_state(source_period_id)
    if source_state is None:
        raise ValueError(
            f"No PartPeriodState found for part {part_id} in period {source_period_id}."
        )

    if not hasattr(source_state, "posterior_alpha") or not hasattr(
        source_state, "posterior_beta"
    ):
        raise ValueError(
            "PartPeriodState does not contain posterior_alpha/posterior_beta."
        )

    posterior_alpha = float(source_state.posterior_alpha)
    posterior_beta = float(source_state.posterior_beta)

    forecast_period_start_date = world.settings.simulation_start_date + timedelta(
        weeks=forecast_period_id * world.settings.weeks_per_period
    )
    forecast_period_dates = _build_forecast_period_dates(
        forecast_period_start_date=forecast_period_start_date,
        weeks_per_period=world.settings.weeks_per_period,
    )

    if next_period_delivery_calloffs is None:
        possible_delivery_calloffs = _build_possible_delivery_dates_from_part_settings(
            part=part,
            forecast_period_id=forecast_period_id,
            forecast_period_start_date=forecast_period_start_date,
            weeks_per_period=world.settings.weeks_per_period,
        )
    else:
        possible_delivery_calloffs = []
        for entry in next_period_delivery_calloffs:
            possible_delivery_calloffs.append(
                {
                    "period_id": int(entry["period_id"]),
                    "delivery_day": entry["delivery_day"],
                    "observation_date": entry["observation_date"],
                }
            )

    rng = random.Random(random_seed)
    scenarios: list[dict] = []

    for scenario_id in range(n_scenarios):
        sampled_theta = rng.betavariate(posterior_alpha, posterior_beta)

        delivery_observations: list[dict] = []
        n_possible_delivery_calloffs = len(possible_delivery_calloffs)
        n_actual_delivery_calloffs = 0
        n_critical_delivery_calloffs = 0
        n_non_critical_delivery_calloffs = 0

        for calloff in possible_delivery_calloffs:
            delivery_happens = rng.random() < part.delivery_call_off_rate
            if not delivery_happens:
                continue

            n_actual_delivery_calloffs += 1

            is_critical = rng.random() < sampled_theta
            if is_critical:
                n_critical_delivery_calloffs += 1
            else:
                n_non_critical_delivery_calloffs += 1

            delivery_observations.append(
                {
                    "period_id": calloff["period_id"],
                    "delivery_day": calloff["delivery_day"],
                    "observation_date": calloff["observation_date"],
                    "is_critical": is_critical,
                }
            )

        scenarios.append(
            {
                "scenario_id": scenario_id,
                "part_id": part_id,
                "source_period_id": source_period_id,
                "forecast_period_id": forecast_period_id,
                "forecast_period_start_date": forecast_period_start_date,
                "forecast_period_dates": forecast_period_dates,
                "posterior_alpha": posterior_alpha,
                "posterior_beta": posterior_beta,
                "sampled_theta": sampled_theta,
                "possible_delivery_calloffs": list(possible_delivery_calloffs),
                "n_possible_delivery_calloffs": n_possible_delivery_calloffs,
                "n_actual_delivery_calloffs": n_actual_delivery_calloffs,
                "n_critical_delivery_calloffs": n_critical_delivery_calloffs,
                "n_non_critical_delivery_calloffs": n_non_critical_delivery_calloffs,
                "delivery_observations": delivery_observations,
            }
        )

    return scenarios

def create_workload_scenarios_from_delivery_performance_scenarios(
    delivery_performance_scenarios: list[dict],
    critical_case_duration_minutes: float,
) -> list[dict]:
    """
    Map delivery performance scenarios to workload scenarios in minutes.

    Returns one list entry per scenario.
    """
    if critical_case_duration_minutes < 0.0:
        raise ValueError("critical_case_duration_minutes must be >= 0.")

    workload_scenarios: list[dict] = []

    for scenario in delivery_performance_scenarios:
        forecast_period_dates = list(scenario["forecast_period_dates"])

        daily_workload_minutes = {day_date: 0.0 for day_date in forecast_period_dates}

        for obs in scenario["delivery_observations"]:
            if obs["is_critical"]:
                daily_workload_minutes[
                    obs["observation_date"]
                ] += critical_case_duration_minutes

        total_workload_minutes = sum(daily_workload_minutes.values())

        workload_scenarios.append(
            {
                "scenario_id": scenario["scenario_id"],
                "part_id": scenario["part_id"],
                "source_period_id": scenario["source_period_id"],
                "forecast_period_id": scenario["forecast_period_id"],
                "forecast_period_start_date": scenario["forecast_period_start_date"],
                "forecast_period_dates": forecast_period_dates,
                "posterior_alpha": scenario["posterior_alpha"],
                "posterior_beta": scenario["posterior_beta"],
                "sampled_theta": scenario["sampled_theta"],
                "critical_case_duration_minutes": critical_case_duration_minutes,
                "possible_delivery_calloffs": list(
                    scenario["possible_delivery_calloffs"]
                ),
                "delivery_observations": list(scenario["delivery_observations"]),
                "daily_workload_minutes": daily_workload_minutes,
                "total_workload_minutes": total_workload_minutes,
                "n_possible_delivery_calloffs": scenario[
                    "n_possible_delivery_calloffs"
                ],
                "n_actual_delivery_calloffs": scenario["n_actual_delivery_calloffs"],
                "n_critical_delivery_calloffs": scenario[
                    "n_critical_delivery_calloffs"
                ],
                "n_non_critical_delivery_calloffs": scenario[
                    "n_non_critical_delivery_calloffs"
                ],
            }
        )

    return workload_scenarios


def create_scenarios_for_current_period(
    world: World,
    n_scenarios: int = 10,
    critical_case_duration_minutes: float = 30.0,
    source_period_id: int | None = None,
    random_seed: int | None = None,
    next_period_delivery_calloffs_by_part: dict[int, list[dict]] | None = None,
) -> list[dict]:
    """
    Create scenario data as a list of scenarios.

    Parameters
    ----------
    world : World
    n_scenarios : int
    critical_case_duration_minutes : float
    source_period_id : int | None
    random_seed : int | None
    next_period_delivery_calloffs_by_part : dict[int, list[dict]] | None

    Output structure
    ----------------
    scenarios_data[scenario_id] is one full scenario dict containing:
        - delivery_performance_by_part
        - workload_by_part
        - workload_by_supplier
    """
    if source_period_id is None:
        source_period_id = world.current_period

    if source_period_id < 0:
        raise ValueError("source_period_id must be >= 0.")

    period = world.get_period(source_period_id)
    if period is None:
        raise ValueError(f"Period {source_period_id} not found.")

    forecast_period_id = source_period_id + 1
    if forecast_period_id >= world.settings.n_simulation_periods:
        raise ValueError("No forecast period available.")

    forecast_period_start_date = world.settings.simulation_start_date + timedelta(
        weeks=forecast_period_id * world.settings.weeks_per_period
    )
    forecast_period_dates = _build_forecast_period_dates(
        forecast_period_start_date=forecast_period_start_date,
        weeks_per_period=world.settings.weeks_per_period,
    )

    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    supplier_ids = [supplier.supplier_id for supplier in period.supplier]
    dispatcher_ids = [dispatcher.dispatcher_id for dispatcher in period.dispatcher]

    if next_period_delivery_calloffs_by_part is None:
        next_period_delivery_calloffs_by_part = {}

    # -----------------------------------------------------------------------
    # Step 1 — generate per-part scenario lists (standard stochastic sampling)
    # -----------------------------------------------------------------------
    delivery_performance_scenarios_by_part: dict[int, list[dict]] = {}
    workload_scenarios_by_part: dict[int, list[dict]] = {}

    for supplier in period.supplier:
        for part in supplier.part:
            part_calloffs = next_period_delivery_calloffs_by_part.get(part.part_id)

            delivery_performance_scenarios = (
                create_delivery_performance_scenarios_for_part(
                    world=world,
                    part_id=part.part_id,
                    source_period_id=source_period_id,
                    n_scenarios=n_scenarios,
                    random_seed=random_seed,
                    next_period_delivery_calloffs=part_calloffs,
                )
            )

            delivery_performance_scenarios_by_part[part.part_id] = (
                delivery_performance_scenarios
            )

    # -----------------------------------------------------------------------
    # Step 2 — optionally re-assign critical flags to enforce a fixed total,
    #          then derive workload scenarios from the (possibly patched) data.
    # -----------------------------------------------------------------------
    for supplier in period.supplier:
        for part in supplier.part:
            workload_scenarios_by_part[part.part_id] = []  # filled below per scenario

    # Build scenario-first view so we can apply the cross-part constraint
    scenarios_data: list[dict] = []

    for scenario_id in range(n_scenarios):
        # Assemble delivery_performance_by_part for this scenario
        delivery_performance_by_part: dict[int, dict] = {}
        for supplier in period.supplier:
            for part in supplier.part:
                delivery_performance_by_part[part.part_id] = (
                    delivery_performance_scenarios_by_part[part.part_id][scenario_id]
                )

        # Derive workload scenarios from (possibly patched) delivery performance
        workload_by_part: dict[int, dict] = {}
        for supplier in period.supplier:
            for part in supplier.part:
                part_dp_scenario = delivery_performance_by_part[part.part_id]

                part_workload_scenarios = (
                    create_workload_scenarios_from_delivery_performance_scenarios(
                        delivery_performance_scenarios=[part_dp_scenario],
                        critical_case_duration_minutes=critical_case_duration_minutes,
                    )
                )
                workload_by_part[part.part_id] = part_workload_scenarios[0]

        # Supplier-level aggregation
        workload_by_supplier: dict[int, dict] = {}
        for supplier in period.supplier:
            supplier_daily_workload_minutes = {
                day_date: 0.0 for day_date in forecast_period_dates
            }
            supplier_total_workload_minutes = 0.0
            supplier_part_ids: list[int] = []

            for part in supplier.part:
                part_workload_scenario = workload_by_part[part.part_id]
                supplier_part_ids.append(part.part_id)

                for day_date, workload in part_workload_scenario[
                    "daily_workload_minutes"
                ].items():
                    supplier_daily_workload_minutes[day_date] += float(workload)

                supplier_total_workload_minutes += float(
                    part_workload_scenario["total_workload_minutes"]
                )

            workload_by_supplier[supplier.supplier_id] = {
                "scenario_id": scenario_id,
                "supplier_id": supplier.supplier_id,
                "source_period_id": source_period_id,
                "forecast_period_id": forecast_period_id,
                "forecast_period_start_date": forecast_period_start_date,
                "forecast_period_dates": list(forecast_period_dates),
                "daily_workload_minutes": supplier_daily_workload_minutes,
                "total_workload_minutes": supplier_total_workload_minutes,
                "part_ids": supplier_part_ids,
            }

        scenarios_data.append(
            {
                "scenario_id": scenario_id,
                "source_period_id": source_period_id,
                "forecast_period_id": forecast_period_id,
                "forecast_period_start_date": forecast_period_start_date,
                "forecast_period_dates": list(forecast_period_dates),
                "days": list(days),
                "supplier_ids": list(supplier_ids),
                "dispatcher_ids": list(dispatcher_ids),
                "delivery_performance_by_part": delivery_performance_by_part,
                "workload_by_part": workload_by_part,
                "workload_by_supplier": workload_by_supplier,
            }
        )

    return scenarios_data


def _beta_mean(alpha: float, beta: float) -> float:
    """Mean of a Beta(alpha, beta) distribution."""
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be > 0.")
    return alpha / (alpha + beta)


def create_average_delivery_performance_scenario_for_part(
    world: World,
    part_id: int,
    source_period_id: int | None = None,
    next_period_delivery_calloffs: list[dict] | None = None,
) -> dict:
    """
    Create ONE deterministic 'average' delivery performance scenario for a
    part, to be used as a warm-start / mean-value scenario for the model.

    No randomness is used:
    - sampled_theta is replaced by theta_mean = alpha / (alpha + beta)
      (the mean of the posterior Beta distribution).
    - Each possible delivery call-off is not simulated as happening/not
      happening (Bernoulli draw). Instead, every possible call-off carries
      an expected 'critical_weight' = delivery_call_off_rate * theta_mean,
      representing E[1{delivery happens} * 1{critical}].

    n_actual_delivery_calloffs / n_critical_delivery_calloffs /
    n_non_critical_delivery_calloffs are therefore expected values
    (floats), not sampled counts.
    """
    if source_period_id is None:
        source_period_id = world.current_period

    if source_period_id < 0:
        raise ValueError("source_period_id must be >= 0.")

    forecast_period_id = source_period_id + 1
    if forecast_period_id >= world.settings.n_simulation_periods:
        raise ValueError("No forecast period available.")

    part = world.get_part_in_period(period_id=source_period_id, part_id=part_id)
    if part is None:
        raise ValueError(f"Part {part_id} not found in period {source_period_id}.")

    source_state = part.get_period_state(source_period_id)
    if source_state is None:
        raise ValueError(
            f"No PartPeriodState found for part {part_id} in period {source_period_id}."
        )

    if not hasattr(source_state, "posterior_alpha") or not hasattr(
        source_state, "posterior_beta"
    ):
        raise ValueError(
            "PartPeriodState does not contain posterior_alpha/posterior_beta."
        )

    posterior_alpha = float(source_state.posterior_alpha)
    posterior_beta = float(source_state.posterior_beta)
    theta_mean = _beta_mean(posterior_alpha, posterior_beta)

    forecast_period_start_date = world.settings.simulation_start_date + timedelta(
        weeks=forecast_period_id * world.settings.weeks_per_period
    )
    forecast_period_dates = _build_forecast_period_dates(
        forecast_period_start_date=forecast_period_start_date,
        weeks_per_period=world.settings.weeks_per_period,
    )

    if next_period_delivery_calloffs is None:
        possible_delivery_calloffs = _build_possible_delivery_dates_from_part_settings(
            part=part,
            forecast_period_id=forecast_period_id,
            forecast_period_start_date=forecast_period_start_date,
            weeks_per_period=world.settings.weeks_per_period,
        )
    else:
        possible_delivery_calloffs = []
        for entry in next_period_delivery_calloffs:
            possible_delivery_calloffs.append(
                {
                    "period_id": int(entry["period_id"]),
                    "delivery_day": entry["delivery_day"],
                    "observation_date": entry["observation_date"],
                }
            )

    n_possible_delivery_calloffs = len(possible_delivery_calloffs)

    # Expected weight that a given call-off both happens AND is critical.
    critical_weight_per_calloff = part.delivery_call_off_rate * theta_mean
    # Expected weight that a given call-off happens (critical or not).
    delivery_weight_per_calloff = part.delivery_call_off_rate

    delivery_observations: list[dict] = []
    for calloff in possible_delivery_calloffs:
        delivery_observations.append(
            {
                "period_id": calloff["period_id"],
                "delivery_day": calloff["delivery_day"],
                "observation_date": calloff["observation_date"],
                # Expected value in [0, 1] replacing the boolean is_critical.
                "critical_weight": critical_weight_per_calloff,
                "delivery_weight": delivery_weight_per_calloff,
            }
        )

    n_actual_delivery_calloffs = n_possible_delivery_calloffs * delivery_weight_per_calloff
    n_critical_delivery_calloffs = n_possible_delivery_calloffs * critical_weight_per_calloff
    n_non_critical_delivery_calloffs = (
        n_actual_delivery_calloffs - n_critical_delivery_calloffs
    )

    return {
        "scenario_id": "average",
        "part_id": part_id,
        "source_period_id": source_period_id,
        "forecast_period_id": forecast_period_id,
        "forecast_period_start_date": forecast_period_start_date,
        "forecast_period_dates": forecast_period_dates,
        "posterior_alpha": posterior_alpha,
        "posterior_beta": posterior_beta,
        "sampled_theta": theta_mean,  # mean instead of a random draw
        "possible_delivery_calloffs": list(possible_delivery_calloffs),
        "n_possible_delivery_calloffs": n_possible_delivery_calloffs,
        "n_actual_delivery_calloffs": n_actual_delivery_calloffs,
        "n_critical_delivery_calloffs": n_critical_delivery_calloffs,
        "n_non_critical_delivery_calloffs": n_non_critical_delivery_calloffs,
        "delivery_observations": delivery_observations,
    }


def create_average_workload_scenario_from_delivery_performance_scenario(
    delivery_performance_scenario: dict,
    critical_case_duration_minutes: float,
) -> dict:
    """
    Map ONE average delivery performance scenario to a workload scenario.
    Deterministic counterpart of
    create_workload_scenarios_from_delivery_performance_scenarios.
    """
    if critical_case_duration_minutes < 0.0:
        raise ValueError("critical_case_duration_minutes must be >= 0.")

    scenario = delivery_performance_scenario
    forecast_period_dates = list(scenario["forecast_period_dates"])

    daily_workload_minutes = {day_date: 0.0 for day_date in forecast_period_dates}

    for obs in scenario["delivery_observations"]:
        daily_workload_minutes[obs["observation_date"]] += (
            obs["critical_weight"] * critical_case_duration_minutes
        )

    total_workload_minutes = sum(daily_workload_minutes.values())

    return {
        "scenario_id": scenario["scenario_id"],
        "part_id": scenario["part_id"],
        "source_period_id": scenario["source_period_id"],
        "forecast_period_id": scenario["forecast_period_id"],
        "forecast_period_start_date": scenario["forecast_period_start_date"],
        "forecast_period_dates": forecast_period_dates,
        "posterior_alpha": scenario["posterior_alpha"],
        "posterior_beta": scenario["posterior_beta"],
        "sampled_theta": scenario["sampled_theta"],
        "critical_case_duration_minutes": critical_case_duration_minutes,
        "possible_delivery_calloffs": list(scenario["possible_delivery_calloffs"]),
        "delivery_observations": list(scenario["delivery_observations"]),
        "daily_workload_minutes": daily_workload_minutes,
        "total_workload_minutes": total_workload_minutes,
        "n_possible_delivery_calloffs": scenario["n_possible_delivery_calloffs"],
        "n_actual_delivery_calloffs": scenario["n_actual_delivery_calloffs"],
        "n_critical_delivery_calloffs": scenario["n_critical_delivery_calloffs"],
        "n_non_critical_delivery_calloffs": scenario["n_non_critical_delivery_calloffs"],
    }


def create_average_scenario_for_current_period(
    world: World,
    critical_case_duration_minutes: float = 30.0,
    source_period_id: int | None = None,
    next_period_delivery_calloffs_by_part: dict[int, list[dict]] | None = None,
) -> dict:
    """
    Create ONE fully deterministic 'average' scenario (mean of the Beta
    posterior, no sampling anywhere) for warm-starting the optimization
    model. Same output shape as one entry of
    create_scenarios_for_current_period, minus fixed_total_critical_calloffs
    (not needed: the mean scenario is already internally consistent by
    construction — nothing to rebalance).

    Output structure
    ----------------
    {
        "scenario_id": "average",
        "source_period_id": ...,
        "forecast_period_id": ...,
        "forecast_period_start_date": ...,
        "forecast_period_dates": [...],
        "days": [...],
        "supplier_ids": [...],
        "dispatcher_ids": [...],
        "delivery_performance_by_part": {part_id: {...}},
        "workload_by_part": {part_id: {...}},
        "workload_by_supplier": {supplier_id: {...}},
    }
    """
    if source_period_id is None:
        source_period_id = world.current_period

    if source_period_id < 0:
        raise ValueError("source_period_id must be >= 0.")

    period = world.get_period(source_period_id)
    if period is None:
        raise ValueError(f"Period {source_period_id} not found.")

    forecast_period_id = source_period_id + 1
    if forecast_period_id >= world.settings.n_simulation_periods:
        raise ValueError("No forecast period available.")

    forecast_period_start_date = world.settings.simulation_start_date + timedelta(
        weeks=forecast_period_id * world.settings.weeks_per_period
    )
    forecast_period_dates = _build_forecast_period_dates(
        forecast_period_start_date=forecast_period_start_date,
        weeks_per_period=world.settings.weeks_per_period,
    )

    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    supplier_ids = [supplier.supplier_id for supplier in period.supplier]
    dispatcher_ids = [dispatcher.dispatcher_id for dispatcher in period.dispatcher]

    if next_period_delivery_calloffs_by_part is None:
        next_period_delivery_calloffs_by_part = {}

    delivery_performance_by_part: dict[int, dict] = {}
    workload_by_part: dict[int, dict] = {}

    for supplier in period.supplier:
        for part in supplier.part:
            part_calloffs = next_period_delivery_calloffs_by_part.get(part.part_id)

            dp_scenario = create_average_delivery_performance_scenario_for_part(
                world=world,
                part_id=part.part_id,
                source_period_id=source_period_id,
                next_period_delivery_calloffs=part_calloffs,
            )
            delivery_performance_by_part[part.part_id] = dp_scenario

            workload_by_part[part.part_id] = (
                create_average_workload_scenario_from_delivery_performance_scenario(
                    delivery_performance_scenario=dp_scenario,
                    critical_case_duration_minutes=critical_case_duration_minutes,
                )
            )

    # Supplier-level aggregation
    workload_by_supplier: dict[int, dict] = {}
    for supplier in period.supplier:
        supplier_daily_workload_minutes = {
            day_date: 0.0 for day_date in forecast_period_dates
        }
        supplier_total_workload_minutes = 0.0
        supplier_part_ids: list[int] = []

        for part in supplier.part:
            part_workload_scenario = workload_by_part[part.part_id]
            supplier_part_ids.append(part.part_id)

            for day_date, workload in part_workload_scenario[
                "daily_workload_minutes"
            ].items():
                supplier_daily_workload_minutes[day_date] += float(workload)

            supplier_total_workload_minutes += float(
                part_workload_scenario["total_workload_minutes"]
            )

        workload_by_supplier[supplier.supplier_id] = {
            "scenario_id": "average",
            "supplier_id": supplier.supplier_id,
            "source_period_id": source_period_id,
            "forecast_period_id": forecast_period_id,
            "forecast_period_start_date": forecast_period_start_date,
            "forecast_period_dates": list(forecast_period_dates),
            "daily_workload_minutes": supplier_daily_workload_minutes,
            "total_workload_minutes": supplier_total_workload_minutes,
            "part_ids": supplier_part_ids,
        }

    return {
        "scenario_id": "average",
        "source_period_id": source_period_id,
        "forecast_period_id": forecast_period_id,
        "forecast_period_start_date": forecast_period_start_date,
        "forecast_period_dates": list(forecast_period_dates),
        "days": list(days),
        "supplier_ids": list(supplier_ids),
        "dispatcher_ids": list(dispatcher_ids),
        "delivery_performance_by_part": delivery_performance_by_part,
        "workload_by_part": workload_by_part,
        "workload_by_supplier": workload_by_supplier,
    }

# Ground truth generation functions for testing and validation

def create_delivery_performance_groundtruth_for_part(
    world: World,
    part_id: int,
    source_period_id: int | None = None,
    n_scenarios: int = 10_000,
    random_seed: int | None = None,
    next_period_delivery_calloffs: list[dict] | None = None,
) -> list[dict]:
    """
    Create groundtruth delivery performance realizations for one part
    for the next period, using the true underlying theta
    (part.base_true_part_theta) instead of a posterior-sampled theta.

    Structural counterpart of create_delivery_performance_scenarios_for_part.
    Meant for a large n_scenarios (default 10_000) used as an out-of-sample
    evaluation set for a first-stage decision obtained from a smaller SAA
    solve (e.g. create_delivery_performance_scenarios_for_part with
    n_scenarios=500) — hence the numpy-vectorized sampling below instead
    of a per-scenario random.Random loop.

    Returns one list entry per scenario. Each entry carries "true_theta"
    instead of "posterior_alpha" / "posterior_beta" / "sampled_theta",
    since the true theta is fixed, not itself a random draw — only the
    call-off / criticality realization is stochastic.
    """
    if n_scenarios <= 0:
        raise ValueError("n_scenarios must be > 0.")

    if source_period_id is None:
        source_period_id = world.current_period

    if source_period_id < 0:
        raise ValueError("source_period_id must be >= 0.")

    forecast_period_id = source_period_id + 1
    if forecast_period_id >= world.settings.n_simulation_periods:
        raise ValueError("No forecast period available.")

    part = world.get_part_in_period(period_id=source_period_id, part_id=part_id)
    if part is None:
        raise ValueError(f"Part {part_id} not found in period {source_period_id}.")

    if not hasattr(part, "base_true_part_theta"):
        raise ValueError("Part does not contain base_true_part_theta.")

    true_theta = float(part.base_true_part_theta)

    forecast_period_start_date = world.settings.simulation_start_date + timedelta(
        weeks=forecast_period_id * world.settings.weeks_per_period
    )
    forecast_period_dates = _build_forecast_period_dates(
        forecast_period_start_date=forecast_period_start_date,
        weeks_per_period=world.settings.weeks_per_period,
    )

    if next_period_delivery_calloffs is None:
        possible_delivery_calloffs = _build_possible_delivery_dates_from_part_settings(
            part=part,
            forecast_period_id=forecast_period_id,
            forecast_period_start_date=forecast_period_start_date,
            weeks_per_period=world.settings.weeks_per_period,
        )
    else:
        possible_delivery_calloffs = []
        for entry in next_period_delivery_calloffs:
            possible_delivery_calloffs.append(
                {
                    "period_id": int(entry["period_id"]),
                    "delivery_day": entry["delivery_day"],
                    "observation_date": entry["observation_date"],
                }
            )

    n_possible_delivery_calloffs = len(possible_delivery_calloffs)

    rng = np.random.default_rng(random_seed)
    # shape (n_scenarios, n_possible_delivery_calloffs) — one Bernoulli
    # draw per (scenario, calloff), vectorized instead of a Python loop.
    delivery_happens = (
        rng.random((n_scenarios, n_possible_delivery_calloffs))
        < part.delivery_call_off_rate
    )
    is_critical = (
        rng.random((n_scenarios, n_possible_delivery_calloffs)) < true_theta
    )

    scenarios: list[dict] = []

    for scenario_id in range(n_scenarios):
        happens_mask = delivery_happens[scenario_id]
        critical_mask = is_critical[scenario_id] & happens_mask

        delivery_observations = [
            {
                "period_id": calloff["period_id"],
                "delivery_day": calloff["delivery_day"],
                "observation_date": calloff["observation_date"],
                "is_critical": bool(critical_mask[i]),
            }
            for i, calloff in enumerate(possible_delivery_calloffs)
            if happens_mask[i]
        ]

        n_actual_delivery_calloffs = int(happens_mask.sum())
        n_critical_delivery_calloffs = int(critical_mask.sum())
        n_non_critical_delivery_calloffs = (
            n_actual_delivery_calloffs - n_critical_delivery_calloffs
        )

        scenarios.append(
            {
                "scenario_id": scenario_id,
                "part_id": part_id,
                "source_period_id": source_period_id,
                "forecast_period_id": forecast_period_id,
                "forecast_period_start_date": forecast_period_start_date,
                "forecast_period_dates": forecast_period_dates,
                "true_theta": true_theta,
                "possible_delivery_calloffs": list(possible_delivery_calloffs),
                "n_possible_delivery_calloffs": n_possible_delivery_calloffs,
                "n_actual_delivery_calloffs": n_actual_delivery_calloffs,
                "n_critical_delivery_calloffs": n_critical_delivery_calloffs,
                "n_non_critical_delivery_calloffs": n_non_critical_delivery_calloffs,
                "delivery_observations": delivery_observations,
            }
        )

    return scenarios


def create_workload_groundtruth_from_delivery_performance_groundtruth(
    delivery_performance_groundtruth: list[dict],
    critical_case_duration_minutes: float,
) -> list[dict]:
    """
    Map groundtruth delivery performance realizations to workload
    realizations in minutes.

    Structural counterpart of
    create_workload_scenarios_from_delivery_performance_scenarios, for
    entries produced by create_delivery_performance_groundtruth_for_part
    (which carry "true_theta" instead of posterior_alpha/posterior_beta/
    sampled_theta).

    Returns one list entry per scenario.
    """
    if critical_case_duration_minutes < 0.0:
        raise ValueError("critical_case_duration_minutes must be >= 0.")

    workload_groundtruth: list[dict] = []

    for scenario in delivery_performance_groundtruth:
        forecast_period_dates = list(scenario["forecast_period_dates"])

        daily_workload_minutes = {day_date: 0.0 for day_date in forecast_period_dates}

        for obs in scenario["delivery_observations"]:
            if obs["is_critical"]:
                daily_workload_minutes[
                    obs["observation_date"]
                ] += critical_case_duration_minutes

        total_workload_minutes = sum(daily_workload_minutes.values())

        workload_groundtruth.append(
            {
                "scenario_id": scenario["scenario_id"],
                "part_id": scenario["part_id"],
                "source_period_id": scenario["source_period_id"],
                "forecast_period_id": scenario["forecast_period_id"],
                "forecast_period_start_date": scenario["forecast_period_start_date"],
                "forecast_period_dates": forecast_period_dates,
                "true_theta": scenario["true_theta"],
                "critical_case_duration_minutes": critical_case_duration_minutes,
                "possible_delivery_calloffs": list(
                    scenario["possible_delivery_calloffs"]
                ),
                "delivery_observations": list(scenario["delivery_observations"]),
                "daily_workload_minutes": daily_workload_minutes,
                "total_workload_minutes": total_workload_minutes,
                "n_possible_delivery_calloffs": scenario[
                    "n_possible_delivery_calloffs"
                ],
                "n_actual_delivery_calloffs": scenario["n_actual_delivery_calloffs"],
                "n_critical_delivery_calloffs": scenario[
                    "n_critical_delivery_calloffs"
                ],
                "n_non_critical_delivery_calloffs": scenario[
                    "n_non_critical_delivery_calloffs"
                ],
            }
        )

    return workload_groundtruth

def create_groundtruth_scenarios_for_current_period(
    world: World,
    n_scenarios: int = 10_000,
    critical_case_duration_minutes: float = 30.0,
    source_period_id: int | None = None,
    random_seed: int | None = None,
    next_period_delivery_calloffs_by_part: dict[int, list[dict]] | None = None,
) -> list[dict]:
    """
    World-level groundtruth counterpart of create_scenarios_for_current_period.

    Uses base_true_part_theta (not the posterior) for every part. Meant for
    a large n_scenarios (default 10_000) used as an out-of-sample evaluation
    set — e.g. to evaluate a first-stage decision obtained from a smaller
    SAA solve (create_scenarios_for_current_period with n_scenarios=500,
    posterior-based) against the true underlying process, as in a
    two-stage stochastic program.

    Output structure
    ----------------
    Same shape as create_scenarios_for_current_period's output, one dict
    per scenario:
        - delivery_performance_by_part
        - workload_by_part
        - workload_by_supplier
    """
    if source_period_id is None:
        source_period_id = world.current_period

    if source_period_id < 0:
        raise ValueError("source_period_id must be >= 0.")

    period = world.get_period(source_period_id)
    if period is None:
        raise ValueError(f"Period {source_period_id} not found.")

    forecast_period_id = source_period_id + 1
    if forecast_period_id >= world.settings.n_simulation_periods:
        raise ValueError("No forecast period available.")

    forecast_period_start_date = world.settings.simulation_start_date + timedelta(
        weeks=forecast_period_id * world.settings.weeks_per_period
    )
    forecast_period_dates = _build_forecast_period_dates(
        forecast_period_start_date=forecast_period_start_date,
        weeks_per_period=world.settings.weeks_per_period,
    )

    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    supplier_ids = [supplier.supplier_id for supplier in period.supplier]
    dispatcher_ids = [dispatcher.dispatcher_id for dispatcher in period.dispatcher]

    if next_period_delivery_calloffs_by_part is None:
        next_period_delivery_calloffs_by_part = {}

    # -----------------------------------------------------------------------
    # Step 1 — generate per-part groundtruth lists (true-theta, vectorized)
    # -----------------------------------------------------------------------
    delivery_performance_groundtruth_by_part: dict[int, list[dict]] = {}

    for supplier in period.supplier:
        for part in supplier.part:
            part_calloffs = next_period_delivery_calloffs_by_part.get(part.part_id)

            delivery_performance_groundtruth = (
                create_delivery_performance_groundtruth_for_part(
                    world=world,
                    part_id=part.part_id,
                    source_period_id=source_period_id,
                    n_scenarios=n_scenarios,
                    random_seed=random_seed,
                    next_period_delivery_calloffs=part_calloffs,
                )
            )

            delivery_performance_groundtruth_by_part[part.part_id] = (
                delivery_performance_groundtruth
            )

    # -----------------------------------------------------------------------
    # Step 2 — assemble scenario-first view and derive workload realizations
    # -----------------------------------------------------------------------
    groundtruth_data: list[dict] = []

    for scenario_id in range(n_scenarios):
        delivery_performance_by_part: dict[int, dict] = {}
        for supplier in period.supplier:
            for part in supplier.part:
                delivery_performance_by_part[part.part_id] = (
                    delivery_performance_groundtruth_by_part[part.part_id][
                        scenario_id
                    ]
                )

        workload_by_part: dict[int, dict] = {}
        for supplier in period.supplier:
            for part in supplier.part:
                part_dp_scenario = delivery_performance_by_part[part.part_id]

                part_workload_groundtruth = (
                    create_workload_groundtruth_from_delivery_performance_groundtruth(
                        delivery_performance_groundtruth=[part_dp_scenario],
                        critical_case_duration_minutes=critical_case_duration_minutes,
                    )
                )
                workload_by_part[part.part_id] = part_workload_groundtruth[0]

        # Supplier-level aggregation
        workload_by_supplier: dict[int, dict] = {}
        for supplier in period.supplier:
            supplier_daily_workload_minutes = {
                day_date: 0.0 for day_date in forecast_period_dates
            }
            supplier_total_workload_minutes = 0.0
            supplier_part_ids: list[int] = []

            for part in supplier.part:
                part_workload_scenario = workload_by_part[part.part_id]
                supplier_part_ids.append(part.part_id)

                for day_date, workload in part_workload_scenario[
                    "daily_workload_minutes"
                ].items():
                    supplier_daily_workload_minutes[day_date] += float(workload)

                supplier_total_workload_minutes += float(
                    part_workload_scenario["total_workload_minutes"]
                )

            workload_by_supplier[supplier.supplier_id] = {
                "scenario_id": scenario_id,
                "supplier_id": supplier.supplier_id,
                "source_period_id": source_period_id,
                "forecast_period_id": forecast_period_id,
                "forecast_period_start_date": forecast_period_start_date,
                "forecast_period_dates": list(forecast_period_dates),
                "daily_workload_minutes": supplier_daily_workload_minutes,
                "total_workload_minutes": supplier_total_workload_minutes,
                "part_ids": supplier_part_ids,
            }

        groundtruth_data.append(
            {
                "scenario_id": scenario_id,
                "source_period_id": source_period_id,
                "forecast_period_id": forecast_period_id,
                "forecast_period_start_date": forecast_period_start_date,
                "forecast_period_dates": list(forecast_period_dates),
                "days": list(days),
                "supplier_ids": list(supplier_ids),
                "dispatcher_ids": list(dispatcher_ids),
                "delivery_performance_by_part": delivery_performance_by_part,
                "workload_by_part": workload_by_part,
                "workload_by_supplier": workload_by_supplier,
            }
        )

    return groundtruth_data