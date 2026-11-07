import numpy as np
import pandas as pd

def evaluate_assignment_v1(
    assignment,
    scenario_set,
    target_workload,
    use_underload_penalty=True,
):
    """
    Evaluate assignment over all scenarios.

    Parameters
    ----------
    assignment:
        {part_id: dispatcher_id}

    scenario_set:
        ScenarioSet

    target_workload:
        {
            dispatcher_id: {
                day: target_minutes
            }
        }

    use_underload_penalty:
        True  -> |workload - target|
        False -> max(workload - target, 0)
    """

    M = scenario_set.get_part_workload_matrix()
    n_scenarios = scenario_set.config.n_scenarios

    dispatchers = sorted(set(assignment.values()))

    # lấy danh sách day
    days = sorted({
        tau
        for (pid, tau, omega) in M.keys()
        if pid in assignment
    })

    scenario_results = []

    for omega in range(n_scenarios):

        deviation = 0.0

        for tau in days:

            for d in dispatchers:

                workload = sum(
                    M.get((pid, tau, omega), 0.0)
                    for pid, assigned_d in assignment.items()
                    if assigned_d == d
                )

                target = target_workload[d][tau]

                if use_underload_penalty:
                    dev = abs(workload - target)
                else:
                    dev = max(workload - target, 0.0)

                deviation += dev

        scenario_results.append({
            "scenario": omega,
            "deviation": deviation,
        })

    df = pd.DataFrame(scenario_results)

    return {
        "scenario_deviation": df,
        "mean_deviation": df["deviation"].mean(),
        "median_deviation": df["deviation"].median(),
        "p95_deviation": df["deviation"].quantile(0.95),
        "max_deviation": df["deviation"].max(),
    }

def evaluate_assignment(
    assignment,
    scenario_set,
    dispatchers_active,
    rep_days,
    use_underload_penalty=True,
    use_scenario_based_target=False,
):
    """
    Evaluate a fixed assignment over all scenarios.

    Parameters
    ----------
    assignment : dict
        {part_id: dispatcher_id}

    scenario_set : ScenarioSet
        ScenarioSet containing workload:
        (part_id, day, scenario) -> workload

    dispatchers_active : dict
        {
            dispatcher_id: dispatcher_object
        }

        dispatcher_object.target_workload[day]
        dispatcher_object.fte_factor

    rep_days : list[str]
        Representative days.

    use_underload_penalty : bool
        True:
            deviation = abs(workload - target)

        False:
            deviation = max(workload - target, 0)

    use_scenario_based_target : bool
        If True, target is calculated from scenario workload
        according to dispatcher FTE share.

    Returns
    -------
    dict
        {
            "detail": DataFrame,
            "scenario_summary": DataFrame,
            "mean_deviation": float,
            "median_deviation": float,
            "p95_deviation": float,
            "cvar95_deviation": float,
            "overload_probability": float,
            "maximum_overload": float,
            "maximum_overload_pct": float,
        }
    """

    # =========================================================
    # Scenario workload
    # =========================================================
    M = scenario_set.get_part_workload_matrix()

    n_scenarios = scenario_set.config.n_scenarios

    dispatchers = sorted(set(assignment.values()))

    # =========================================================
    # Precompute parts assigned to each dispatcher
    # =========================================================
    dispatcher_parts = {
        d: []
        for d in dispatchers
    }

    for part_id, dispatcher_id in assignment.items():
        dispatcher_parts.setdefault(dispatcher_id, []).append(part_id)

    # =========================================================
    # FTE share
    # =========================================================
    total_fte = sum(
        dispatchers_active[d].fte_factor
        for d in dispatchers
    )

    # =========================================================
    # Detailed results
    # =========================================================
    detail_rows = []

    for omega in range(n_scenarios):

        # -----------------------------------------------------
        # Scenario total workload by day
        # -----------------------------------------------------
        scenario_total_workload = {}

        for tau in rep_days:
            scenario_total_workload[tau] = sum(
                M.get((p, tau, omega), 0.0)
                for p in assignment
            )

        for tau in rep_days:

            for d in dispatchers:

                # -------------------------------------------------
                # Dispatcher workload
                # -------------------------------------------------
                workload = sum(
                    M.get((p, tau, omega), 0.0)
                    for p in dispatcher_parts[d]
                )

                # -------------------------------------------------
                # Target
                # -------------------------------------------------
                if use_scenario_based_target:

                    target = (
                        scenario_total_workload[tau]
                        * dispatchers_active[d].fte_factor
                        / total_fte
                    )

                else:

                    target = (
                        dispatchers_active[d]
                        .target_workload[tau]
                    )

                # -------------------------------------------------
                # Deviation
                # -------------------------------------------------
                if use_underload_penalty:
                    deviation = abs(workload - target)
                else:
                    deviation = max(
                        workload - target,
                        0.0,
                    )

                # -------------------------------------------------
                # Overload
                # -------------------------------------------------
                overload = max(
                    workload - target,
                    0.0,
                )

                overload_pct = (
                    overload / target * 100.0
                    if target > 0
                    else 0.0
                )

                is_overloaded = overload > 0

                detail_rows.append({
                    "scenario": omega,
                    "dispatcher": d,
                    "day": tau,
                    "workload": workload,
                    "target": target,
                    "deviation": deviation,
                    "overload": overload,
                    "overload_pct": overload_pct,
                    "is_overloaded": is_overloaded,
                })

    # =========================================================
    # Detail DataFrame
    # =========================================================
    detail = pd.DataFrame(detail_rows)

    # =========================================================
    # Scenario-level summary
    # =========================================================
    scenario_summary = (
        detail
        .groupby("scenario")
        .agg(
            total_deviation=("deviation", "sum"),
            total_overload=("overload", "sum"),
            max_overload=("overload", "max"),
            max_overload_pct=("overload_pct", "max"),
            n_overloaded=("is_overloaded", "sum"),
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # Scenario is overloaded if ANY dispatcher/day is overloaded
    # ---------------------------------------------------------
    scenario_summary["scenario_overloaded"] = (
        scenario_summary["n_overloaded"] > 0
    )

    # =========================================================
    # Deviation statistics
    # =========================================================
    deviations = (
        scenario_summary["total_deviation"]
        .to_numpy()
    )

    mean_deviation = float(
        np.mean(deviations)
    )

    median_deviation = float(
        np.median(deviations)
    )

    p95_deviation = float(
        np.percentile(deviations, 95)
    )

    # =========================================================
    # CVaR 95%
    # =========================================================
    var95 = np.percentile(
        deviations,
        95,
    )

    tail = deviations[
        deviations >= var95
    ]

    cvar95_deviation = float(
        np.mean(tail)
    )

    # =========================================================
    # Overload probability
    #
    # Probability that at least one dispatcher/day
    # is overloaded in a scenario.
    # =========================================================
    overload_probability = float(
        scenario_summary["scenario_overloaded"].mean()
    )

    # =========================================================
    # Maximum overload
    # =========================================================
    maximum_overload = float(
        detail["overload"].max()
    )

    maximum_overload_pct = float(
        detail["overload_pct"].max()
    )

    # =========================================================
    # Return
    # =========================================================
    return {
        "detail": detail,
        "scenario_summary": scenario_summary,

        "mean_deviation": mean_deviation,
        "median_deviation": median_deviation,
        "p95_deviation": p95_deviation,
        "cvar95_deviation": cvar95_deviation,

        "overload_probability": overload_probability,
        "maximum_overload": maximum_overload,
        "maximum_overload_pct": maximum_overload_pct,
    }

import numpy as np


def improvement_ci95(
    baseline,
    method,
    metric_func=np.mean,
    n_bootstrap=10_000,
    confidence=0.95,
    seed=42,
):
    """
    Bootstrap CI95% for improvement of `method`
    relative to `baseline`.

    Improvement:
        (baseline_metric - method_metric)
        / baseline_metric * 100

    Assumes lower metric value is better.

    Parameters
    ----------
    baseline : array-like
        Scenario-level baseline results.

    method : array-like
        Scenario-level method results.

    metric_func : callable
        Statistic to evaluate:
            np.mean
            np.median
            lambda x: np.percentile(x, 95)
            cvar95
            np.max
            ...

    Returns
    -------
    tuple
        (ci_lower, ci_upper)
    """

    baseline = np.asarray(baseline, dtype=float)
    method = np.asarray(method, dtype=float)

    # Check empty input
    if baseline.size == 0 or method.size == 0:
        return -999, -999

    if baseline.shape != method.shape:
        raise ValueError(
            "baseline and method must have the same shape."
        )

    if len(baseline) == 0:
        raise ValueError(
            "Input arrays cannot be empty."
        )

    rng = np.random.default_rng(seed)

    n = len(baseline)

    bootstrap_improvements = np.empty(n_bootstrap)

    for i in range(n_bootstrap):

        # Paired bootstrap:
        # same scenario indices for baseline and method
        indices = rng.integers(
            0,
            n,
            size=n,
        )

        baseline_sample = baseline[indices]
        method_sample = method[indices]

        baseline_metric = metric_func(
            baseline_sample
        )

        method_metric = metric_func(
            method_sample
        )

        if baseline_metric == 0:
            bootstrap_improvements[i] = np.nan
        else:
            bootstrap_improvements[i] = (
                (baseline_metric - method_metric)
                / baseline_metric
                * 100
            )

    bootstrap_improvements = (
        bootstrap_improvements[
            np.isfinite(bootstrap_improvements)
        ]
    )

    alpha = 1 - confidence

    ci_lower = np.percentile(
        bootstrap_improvements,
        alpha / 2 * 100,
    )

    ci_upper = np.percentile(
        bootstrap_improvements,
        (1 - alpha / 2) * 100,
    )

    return ci_lower, ci_upper

def format_ci95(ci):
    lower, upper = ci
    return [round(float(lower),3), round(float(upper),3)]

def cvar95(x):
    var95 = np.percentile(x, 95)
    tail = x[x >= var95]

    if len(tail) == 0:
        return var95

    return tail.mean()