from __future__ import annotations

from typing import Optional, Callable
from datetime import date, timedelta
from .bayesian_stats import *
import random


from .simulation_entities import (
    WEEKDAYS,
    WEEKDAY_TO_OFFSET,
    DeliveryObservation,
    PartPeriodState,
    Part,
    PartTrueThetaTrajectory,
    Supplier,
    Dispatcher,
    Period,
    World,
    WorldSettings,
)


def make_int_drawer(low: int, high: int) -> Callable[[random.Random], int]:
    if low > high:
        raise ValueError("low must be <= high.")

    def draw(rng: random.Random) -> int:
        return rng.randint(low, high)

    return draw


def make_constant_float_drawer(value: float) -> Callable[[random.Random], float]:
    def draw(rng: random.Random) -> float:
        return value

    return draw


def default_supplier_part_count_draw(rng: random.Random) -> int:
    return rng.randint(3, 8)


def default_delivery_day_pattern_draw(rng: random.Random) -> list[str]:
    n_days = rng.randint(1, len(WEEKDAYS))
    sampled_days = rng.sample(WEEKDAYS, k=n_days)
    weekday_order = {day: idx for idx, day in enumerate(WEEKDAYS)}
    sampled_days.sort(key=lambda day: weekday_order[day])
    return sampled_days


def default_base_shortage_rate_draw(rng: random.Random) -> float:
    return rng.uniform(0.02, 0.20)


def default_delivery_call_off_rate_draw(rng: random.Random) -> float:
    return 1.0


def default_regime_change_target_draw(rng: random.Random) -> float:
    return rng.uniform(0.05, 0.95)


def default_temporary_shock_value_draw(rng: random.Random) -> float:
    return rng.uniform(0.05, 0.95)


def default_linear_trend_target_draw(rng: random.Random) -> float:
    return rng.uniform(0.0, 1.0)


def _clip_probability(x: float, eps: float = 1e-3) -> float:
    return max(eps, min(1.0 - eps, x))


def _compute_period_start_date(
    simulation_start_date: date,
    period_id: int,
    weeks_per_period: int,
) -> date:
    return simulation_start_date + timedelta(weeks=period_id * weeks_per_period)


def _sample_dirichlet_weights(
    n: int,
    alpha: float,
    rng: random.Random,
) -> list[float]:
    if n <= 0:
        return []
    if alpha <= 0.0:
        raise ValueError("alpha must be positive.")

    draws = [rng.gammavariate(alpha, 1.0) for _ in range(n)]
    total = sum(draws)

    if total <= 0.0:
        return [1.0 / n for _ in range(n)]

    return [x / total for x in draws]


def _compute_supplier_concentration_weights(
    n_suppliers: int,
    rho: float,
    rng: random.Random,
    target_alpha: float = 0.15,
) -> list[float]:
    """
    Compute supplier concentration weights with the following behavior:
    - rho = 0.0: equal weights across suppliers
    - rho = 1.0: strongly concentrated target distribution
    - 0 < rho < 1: linear interpolation between both

    The weights always sum to 1.
    """
    if n_suppliers <= 0:
        return []

    rho = max(0.0, min(1.0, rho))

    equal_weights = [1.0 / n_suppliers for _ in range(n_suppliers)]
    concentrated_weights = _sample_dirichlet_weights(
        n=n_suppliers,
        alpha=target_alpha,
        rng=rng,
    )

    mixed_weights = [
        (1.0 - rho) * w_equal + rho * w_conc
        for w_equal, w_conc in zip(equal_weights, concentrated_weights)
    ]

    total = sum(mixed_weights)
    if total <= 0.0:
        return [1.0 / n_suppliers for _ in range(n_suppliers)]

    return [w / total for w in mixed_weights]


def _compute_supplier_expected_calloff_exposure(
    supplier: Supplier,
    weeks_per_period: int,
) -> float:
    """
    Compute expected delivery call-off exposure of a supplier in one period.

    Exposure is the expected number of actual delivery call-offs in the period,
    based on:
    - number of potential delivery dates
    - delivery_call_off_rate of each part
    """
    exposure = 0.0

    for part in supplier.part:
        n_possible_calloffs = weeks_per_period * len(part.delivery_day_pattern)
        exposure += n_possible_calloffs * float(part.delivery_call_off_rate)

    return exposure


def _compute_supplier_theta_by_concentration(
    suppliers: list[Supplier],
    weeks_per_period: int,
    theta_global: float,
    rho: float,
    rng: random.Random,
    target_alpha: float = 0.15,
    eps: float = 1e-6,
) -> dict[int, float]:
    """
    Compute supplier-specific theta values such that:
    - the global expected number of critical events stays approximately constant
    - rho controls only the concentration across suppliers

    Construction:
    -------------
    Let E_i be the expected call-off exposure of supplier i.
    Let w_i(rho) be concentration weights summing to 1.
    Let total_expected_critical = theta_global * sum_i E_i.

    Then supplier i receives:
        expected_critical_i = w_i * total_expected_critical
        theta_i = expected_critical_i / E_i
    """
    if not (0.0 <= theta_global <= 1.0):
        raise ValueError("theta_global must be in [0, 1].")

    if len(suppliers) == 0:
        return {}

    weights = _compute_supplier_concentration_weights(
        n_suppliers=len(suppliers),
        rho=rho,
        rng=rng,
        target_alpha=target_alpha,
    )

    exposures: list[float] = []
    for supplier in suppliers:
        exposure = _compute_supplier_expected_calloff_exposure(
            supplier=supplier,
            weeks_per_period=weeks_per_period,
        )
        exposures.append(max(exposure, eps))

    total_exposure = sum(exposures)
    total_expected_critical = theta_global * total_exposure

    supplier_theta: dict[int, float] = {}

    for supplier, weight, exposure in zip(suppliers, weights, exposures):
        expected_critical_supplier = weight * total_expected_critical
        theta_supplier = expected_critical_supplier / exposure
        supplier_theta[supplier.supplier_id] = _clip_probability(theta_supplier)

    return supplier_theta


def _compute_supplier_risk_factors(
    n_suppliers: int,
    rho: float,
    rng: random.Random,
    max_supplier_risk_spread: float = 0.12,
) -> list[float]:
    """
    Legacy helper retained for backward compatibility.

    This function is no longer used to control concentration in the new setup.
    It returns zero-centered factors, but the new concentration logic is handled
    via supplier-specific theta construction based on exposure-normalized weights.
    """
    if n_suppliers <= 0:
        return []

    return [0.0 for _ in range(n_suppliers)]


# ---------------------------------------------------------------------------
# Hierarchical Beta-Bernoulli theta sampling (Section 1 of the document)
# ---------------------------------------------------------------------------

def _sample_supplier_theta_hierarchical(
    m_global: float,
    kappa_s: float,
    rng: random.Random,
) -> float:
    """
    Draw a supplier-level risk parameter from the upper-level Beta prior.

    Model
    -----
        θs ~ Beta(m_global * kappa_s,  (1 - m_global) * kappa_s)

    The mean of θs equals m_global and the variance is governed by kappa_s:
        Var(θs) = m_global * (1 - m_global) / (kappa_s + 1)

    Small kappa_s  →  high cross-supplier heterogeneity.
    Large kappa_s  →  homogeneous supplier population.

    Parameters
    ----------
    m_global:
        Average delivery disruption probability across the supplier population.
    kappa_s:
        Concentration (precision) parameter for the supplier-level Beta.
    rng:
        Random generator.

    Returns
    -------
    float in (0, 1)  — the drawn θs, clipped to avoid degeneracy.
    """
    if not (0.0 < m_global < 1.0):
        raise ValueError("m_global must be in (0, 1).")
    if kappa_s <= 0.0:
        raise ValueError("kappa_s must be > 0.")

    alpha = m_global * kappa_s
    beta  = (1.0 - m_global) * kappa_s
    return _clip_probability(rng.betavariate(alpha, beta))


def _sample_part_theta_hierarchical(
    supplier_theta: float,
    kappa_p: float,
    rng: random.Random,
) -> float:
    """
    Draw a part-level risk parameter from the lower-level Beta prior,
    conditional on the supplier-level risk θs.

    Model
    -----
        θps | θs ~ Beta(θs * kappa_p,  (1 - θs) * kappa_p)

    The conditional mean equals the supplier risk:
        E[θps | θs] = θs

    and the conditional variance is governed by kappa_p:
        Var(θps | θs) = θs * (1 - θs) / (kappa_p + 1)

    Small kappa_p  →  large within-supplier part heterogeneity.
    Large kappa_p  →  parts are nearly identical within their supplier.

    Parameters
    ----------
    supplier_theta:
        The supplier-level risk parameter θs (mean of the part-level prior).
    kappa_p:
        Concentration (precision) parameter for the part-level Beta.
    rng:
        Random generator.

    Returns
    -------
    float in (0, 1)  — the drawn θps, clipped to avoid degeneracy.
    """
    if not (0.0 < supplier_theta < 1.0):
        raise ValueError("supplier_theta must be in (0, 1).")
    if kappa_p <= 0.0:
        raise ValueError("kappa_p must be > 0.")

    alpha = supplier_theta * kappa_p
    beta  = (1.0 - supplier_theta) * kappa_p
    return _clip_probability(rng.betavariate(alpha, beta))


def _assign_hierarchical_theta_to_suppliers_and_parts(
    world: World,
    suppliers: list[Supplier],
) -> None:
    """
    Draw and assign hierarchical theta values using nested RNG structure.

    Seed hierarchy:
        ks_seed  → ks_rng  → draws θs for all suppliers
        kp_seed  → kp_rng  → draws θps for all parts (conditional on θs)

    This ensures:
        - Same ks_seed always produces same supplier-level thetas
        - Same (ks_seed, kp_seed) pair always produces same part-level thetas
        - kp seeds are independent across ks seeds
    """
    settings = world.settings

    if not (0.0 < settings.m_global < 1.0):
        raise ValueError("settings.m_global must be in (0, 1).")
    if settings.kappa_s <= 0.0:
        raise ValueError("settings.kappa_s must be > 0.")
    if settings.kappa_p <= 0.0:
        raise ValueError("settings.kappa_p must be > 0.")

    for supplier in suppliers:
        # ── Step 1: supplier-level draw ──────────────────────────────────────
        theta_s = _sample_supplier_theta_hierarchical(
            m_global=settings.m_global,
            kappa_s=settings.kappa_s,
            rng=world.ks_rng,
        )
        supplier.supplier_theta       = theta_s
        supplier.supplier_risk_factor = theta_s - settings.m_global

        # ── Step 2: part-level draw conditional on θs ────────────────────────
        for part in supplier.part:
            theta_ps = _sample_part_theta_hierarchical(
                supplier_theta=theta_s,
                kappa_p=settings.kappa_p,
                rng=world.kp_rng,
            )
            part.base_true_part_theta = theta_ps

            trajectory = _generate_true_theta_trajectory_for_part(
                part=part,
                settings=settings,
                rng=world.kp_rng,
            )
            world.set_true_theta_trajectory(trajectory)


def _create_heterogeneity_regime(
    kappa_s: float,
    kappa_p: float,
    m_global: float = 0.10,
) -> dict:
    """
    Return a parameter dict that describes one heterogeneity regime.

    Use this as a convenience builder when running the 9-regime grid
    experiment from Section 1.5 (kappa_s × kappa_p combinations).

    Recommended grids
    -----------------
        kappa_s  in {10, 30, 100}   (high / medium / low supplier heterogeneity)
        kappa_p  in {5,  20, 100}   (high / medium / low within-supplier heterogeneity)

    Example
    -------
        for ks in [10, 30, 100]:
            for kp in [5, 20, 100]:
                regime = _create_heterogeneity_regime(ks, kp)
                # plug into WorldSettings(**regime, ...)

    Parameters
    ----------
    kappa_s:
        Concentration across suppliers.
    kappa_p:
        Concentration within a supplier across parts.
    m_global:
        Average disruption probability in the population.

    Returns
    -------
    dict with keys: use_hierarchical_theta, m_global, kappa_s, kappa_p.
    """
    if kappa_s <= 0.0:
        raise ValueError("kappa_s must be > 0.")
    if kappa_p <= 0.0:
        raise ValueError("kappa_p must be > 0.")
    if not (0.0 < m_global < 1.0):
        raise ValueError("m_global must be in (0, 1).")

    return {
        "use_hierarchical_theta": True,
        "m_global": m_global,
        "kappa_s":  kappa_s,
        "kappa_p":  kappa_p,
    }


def _clone_dispatchers_from_previous_period(
    previous_period: Period,
) -> list[Dispatcher]:
    cloned_dispatchers: list[Dispatcher] = []

    for dispatcher in previous_period.dispatcher:
        cloned_dispatchers.append(
            Dispatcher(
                dispatcher_id=dispatcher.dispatcher_id,
                employment_type=dispatcher.employment_type,
                target_workload_per_day=dispatcher.target_workload_per_day,
                assigned_supplier_ids=[],
            )
        )

    return cloned_dispatchers


def _clone_suppliers_from_previous_period(previous_period: Period) -> list[Supplier]:
    cloned_suppliers: list[Supplier] = []

    for supplier in previous_period.supplier:
        cloned_parts: list[Part] = []

        for part in supplier.part:
            cloned_parts.append(
                Part(
                    part_id=part.part_id,
                    start_period=part.start_period,
                    end_period=part.end_period,
                    delivery_day_pattern=list(part.delivery_day_pattern),
                    delivery_call_off_rate=part.delivery_call_off_rate,
                    base_true_part_theta=part.base_true_part_theta,
                    is_true_theta_stationary=part.is_true_theta_stationary,
                    delivery_observations=list(part.delivery_observations),
                    period_state=list(part.period_state),
                )
            )

        cloned_suppliers.append(
            Supplier(
                supplier_id=supplier.supplier_id,
                part=cloned_parts,
                supplier_risk_factor=supplier.supplier_risk_factor,
            )
        )

    return cloned_suppliers


def _validate_and_fill_default_drawers(settings: WorldSettings) -> None:
    if settings.n_simulation_periods <= 0:
        raise ValueError("n_simulation_periods must be positive.")

    if settings.weeks_per_period <= 0:
        raise ValueError("weeks_per_period must be positive.")

    if (
        settings.current_period < -1
        or settings.current_period >= settings.n_simulation_periods
    ):
        raise ValueError(
            "current_period must satisfy -1 <= current_period < n_simulation_periods."
        )

    if not (0.0 <= settings.non_stationary_theta_share <= 1.0):
        raise ValueError("non_stationary_theta_share must be in [0, 1].")

    if settings.supplier_part_count_sampler is None:
        settings.supplier_part_count_sampler = default_supplier_part_count_draw

    if settings.delivery_day_pattern_sampler is None:
        settings.delivery_day_pattern_sampler = default_delivery_day_pattern_draw

    if settings.base_shortage_rate_sampler is None:
        settings.base_shortage_rate_sampler = default_base_shortage_rate_draw

    if settings.delivery_call_off_rate_sampler is None:
        settings.delivery_call_off_rate_sampler = default_delivery_call_off_rate_draw

    if settings.regime_change_target_sampler is None:
        settings.regime_change_target_sampler = default_regime_change_target_draw

    if settings.temporary_shock_value_sampler is None:
        settings.temporary_shock_value_sampler = default_temporary_shock_value_draw

    if settings.linear_trend_target_sampler is None:
        settings.linear_trend_target_sampler = default_linear_trend_target_draw

    allowed_modes = {"regime_change", "temporary_shock", "linear_trend"}
    for mode in settings.allowed_non_stationary_theta_modes:
        if mode not in allowed_modes:
            raise ValueError(
                "allowed_non_stationary_theta_modes may only contain "
                "'regime_change', 'temporary_shock', 'linear_trend'."
            )

    if settings.world_config is not None:
        _validate_world_config(settings.world_config, settings)


def _validate_world_config(config: dict, settings: WorldSettings) -> None:
    """
    Validate the structure and consistency of a world_config dict.

    Checks performed
    ----------------
    - Top-level keys must be a subset of {"suppliers", "parts", "dispatchers"}.
    - "suppliers" section: each entry must have a positive integer "n_parts".
      The number of supplier entries must equal settings.n_init_suppliers.
    - "parts" section: each entry must have a valid "delivery_day_pattern"
      (non-empty list of weekday strings).  Optional keys
      "delivery_call_off_rate" and "is_true_theta_stationary" are type-checked.
      The total number of parts must equal the sum of n_parts across all
      supplier entries (when both sections are present).
    - "dispatchers" section: each entry must have "employment_type" in
      {"fulltime", "parttime"}.  The number of entries must equal
      settings.n_dispatchers.
    """
    allowed_top_keys = {"suppliers", "parts", "dispatchers"}
    unknown = set(config.keys()) - allowed_top_keys
    if unknown:
        raise ValueError(
            f"world_config contains unknown top-level keys: {unknown}. "
            f"Allowed: {allowed_top_keys}."
        )

    # ── "suppliers" section ──────────────────────────────────────────────────
    supplier_cfg = config.get("suppliers", {})
    if supplier_cfg:
        if len(supplier_cfg) != settings.n_init_suppliers:
            raise ValueError(
                f"world_config['suppliers'] has {len(supplier_cfg)} entries but "
                f"settings.n_init_suppliers = {settings.n_init_suppliers}."
            )
        for sid, scfg in supplier_cfg.items():
            if "n_parts" not in scfg:
                raise ValueError(
                    f"world_config['suppliers'][{sid}] is missing required key 'n_parts'."
                )
            if not isinstance(scfg["n_parts"], int) or scfg["n_parts"] < 1:
                raise ValueError(
                    f"world_config['suppliers'][{sid}]['n_parts'] must be a positive int."
                )

    # ── "dispatchers" section ────────────────────────────────────────────────
    dispatcher_cfg = config.get("dispatchers", {})
    if dispatcher_cfg:
        if len(dispatcher_cfg) != settings.n_dispatchers:
            raise ValueError(
                f"world_config['dispatchers'] has {len(dispatcher_cfg)} entries but "
                f"settings.n_dispatchers = {settings.n_dispatchers}."
            )
        valid_employment_types = {"fulltime", "parttime"}
        for did, dcfg in dispatcher_cfg.items():
            if "employment_type" not in dcfg:
                raise ValueError(
                    f"world_config['dispatchers'][{did}] is missing required key "
                    f"'employment_type'."
                )
            if dcfg["employment_type"] not in valid_employment_types:
                raise ValueError(
                    f"world_config['dispatchers'][{did}]['employment_type'] must be "
                    f"'fulltime' or 'parttime', got '{dcfg['employment_type']}'."
                )

    # ── "parts" section ──────────────────────────────────────────────────────
    part_cfg = config.get("parts", {})
    if part_cfg:
        valid_weekdays = {"Mon", "Tue", "Wed", "Thu", "Fri"}
        for pid, pcfg in part_cfg.items():
            if "delivery_day_pattern" not in pcfg:
                raise ValueError(
                    f"world_config['parts'][{pid}] is missing required key "
                    f"'delivery_day_pattern'."
                )
            pattern = pcfg["delivery_day_pattern"]
            if not isinstance(pattern, list) or len(pattern) == 0:
                raise ValueError(
                    f"world_config['parts'][{pid}]['delivery_day_pattern'] must be "
                    f"a non-empty list."
                )
            invalid_days = set(pattern) - valid_weekdays
            if invalid_days:
                raise ValueError(
                    f"world_config['parts'][{pid}]['delivery_day_pattern'] contains "
                    f"invalid weekday(s): {invalid_days}."
                )
            if "delivery_call_off_rate" in pcfg:
                rate = pcfg["delivery_call_off_rate"]
                if not isinstance(rate, (int, float)) or not (0.0 <= rate <= 1.0):
                    raise ValueError(
                        f"world_config['parts'][{pid}]['delivery_call_off_rate'] must "
                        f"be a float in [0, 1]."
                    )
            if "is_true_theta_stationary" in pcfg:
                if not isinstance(pcfg["is_true_theta_stationary"], bool):
                    raise ValueError(
                        f"world_config['parts'][{pid}]['is_true_theta_stationary'] "
                        f"must be a bool."
                    )

        # Cross-check total part count against supplier section if both present
        if supplier_cfg:
            expected_n_parts = sum(
                scfg["n_parts"] for scfg in supplier_cfg.values()
            )
            if len(part_cfg) != expected_n_parts:
                raise ValueError(
                    f"world_config['parts'] has {len(part_cfg)} entries but the sum "
                    f"of n_parts across all suppliers is {expected_n_parts}. "
                    f"These must match."
                )


def _draw_supplier_part_count(settings: WorldSettings, rng: random.Random) -> int:
    return max(1, int(settings.supplier_part_count_sampler(rng)))


def _draw_delivery_day_pattern(
    settings: WorldSettings, rng: random.Random
) -> list[str]:
    pattern = settings.delivery_day_pattern_sampler(rng)
    if len(pattern) == 0:
        raise ValueError("delivery_day_pattern_sampler returned an empty pattern.")
    return list(pattern)


def _draw_base_true_theta(
    settings: WorldSettings,
    rng: random.Random,
    supplier_risk_factor: float,
) -> float:
    base_theta = settings.base_shortage_rate_sampler(rng)
    return _clip_probability(base_theta + supplier_risk_factor)


def _draw_delivery_call_off_rate(settings: WorldSettings, rng: random.Random) -> float:
    rate = settings.delivery_call_off_rate_sampler(rng)
    if not (0.0 <= rate <= 1.0):
        raise ValueError("delivery_call_off_rate must be in [0, 1].")
    return rate


def _target_workload_from_employment_type(
    employment_type: str,
    settings: WorldSettings,
) -> float:
    if employment_type == "fulltime":
        return settings.target_workload_dispatcher_fulltime_per_day
    if employment_type == "parttime":
        return settings.target_workload_dispatcher_parttime_per_day
    raise ValueError("employment_type must be either 'fulltime' or 'parttime'.")


def _make_stationarity_flag(settings: WorldSettings, rng: random.Random) -> bool:
    return rng.random() >= settings.non_stationary_theta_share


def _generate_regime_change_trajectory(
    part: Part,
    settings: WorldSettings,
    rng: random.Random,
) -> PartTrueThetaTrajectory:
    min_period = max(part.start_period + 1, settings.regime_change_min_period)
    max_period = (
        settings.regime_change_max_period
        if settings.regime_change_max_period is not None
        else part.end_period
    )
    max_period = min(max_period, part.end_period)

    if min_period > max_period:
        theta_by_period = {
            period_id: part.base_true_part_theta
            for period_id in range(part.start_period, part.end_period + 1)
        }
        return PartTrueThetaTrajectory(
            part_id=part.part_id,
            theta_by_period=theta_by_period,
        )

    change_period = rng.randint(min_period, max_period)
    target_theta = _clip_probability(settings.regime_change_target_sampler(rng))

    theta_by_period: dict[int, float] = {}
    for period_id in range(part.start_period, part.end_period + 1):
        if period_id < change_period:
            theta_by_period[period_id] = part.base_true_part_theta
        else:
            theta_by_period[period_id] = target_theta

    return PartTrueThetaTrajectory(
        part_id=part.part_id,
        theta_by_period=theta_by_period,
    )


def _generate_temporary_shock_trajectory(
    part: Part,
    settings: WorldSettings,
    rng: random.Random,
) -> PartTrueThetaTrajectory:
    min_start = max(part.start_period, settings.temporary_shock_min_start_period)
    max_start = (
        settings.temporary_shock_max_start_period
        if settings.temporary_shock_max_start_period is not None
        else part.end_period
    )
    max_start = min(max_start, part.end_period)

    if min_start > max_start:
        theta_by_period = {
            period_id: part.base_true_part_theta
            for period_id in range(part.start_period, part.end_period + 1)
        }
        return PartTrueThetaTrajectory(
            part_id=part.part_id,
            theta_by_period=theta_by_period,
        )

    shock_start = rng.randint(min_start, max_start)
    duration = rng.randint(
        settings.temporary_shock_min_duration_periods,
        settings.temporary_shock_max_duration_periods,
    )
    shock_end = min(part.end_period, shock_start + duration - 1)
    shock_theta = _clip_probability(settings.temporary_shock_value_sampler(rng))

    theta_by_period: dict[int, float] = {}
    for period_id in range(part.start_period, part.end_period + 1):
        if shock_start <= period_id <= shock_end:
            theta_by_period[period_id] = shock_theta
        else:
            theta_by_period[period_id] = part.base_true_part_theta

    return PartTrueThetaTrajectory(
        part_id=part.part_id,
        theta_by_period=theta_by_period,
    )


def _generate_linear_trend_trajectory(
    part: Part,
    settings: WorldSettings,
    rng: random.Random,
) -> PartTrueThetaTrajectory:
    final_period = (
        settings.linear_trend_final_period
        if settings.linear_trend_final_period is not None
        else part.end_period
    )
    final_period = min(final_period, part.end_period)

    target_theta = _clip_probability(settings.linear_trend_target_sampler(rng))

    if final_period <= part.start_period:
        theta_by_period = {
            period_id: target_theta
            for period_id in range(part.start_period, part.end_period + 1)
        }
        return PartTrueThetaTrajectory(
            part_id=part.part_id,
            theta_by_period=theta_by_period,
        )

    theta_by_period: dict[int, float] = {}
    for period_id in range(part.start_period, part.end_period + 1):
        if period_id <= part.start_period:
            theta_by_period[period_id] = part.base_true_part_theta
        elif period_id >= final_period:
            theta_by_period[period_id] = target_theta
        else:
            progress = (period_id - part.start_period) / (
                final_period - part.start_period
            )
            theta = part.base_true_part_theta + progress * (
                target_theta - part.base_true_part_theta
            )
            theta_by_period[period_id] = _clip_probability(theta)

    return PartTrueThetaTrajectory(
        part_id=part.part_id,
        theta_by_period=theta_by_period,
    )


def _generate_true_theta_trajectory_for_part(
    part: Part,
    settings: WorldSettings,
    rng: random.Random,
) -> PartTrueThetaTrajectory:
    if part.is_true_theta_stationary:
        theta_by_period = {
            period_id: part.base_true_part_theta
            for period_id in range(part.start_period, part.end_period + 1)
        }
        return PartTrueThetaTrajectory(
            part_id=part.part_id,
            theta_by_period=theta_by_period,
        )

    if len(settings.allowed_non_stationary_theta_modes) == 0:
        theta_by_period = {
            period_id: part.base_true_part_theta
            for period_id in range(part.start_period, part.end_period + 1)
        }
        return PartTrueThetaTrajectory(
            part_id=part.part_id,
            theta_by_period=theta_by_period,
        )

    mode = rng.choice(list(settings.allowed_non_stationary_theta_modes))

    if mode == "regime_change":
        return _generate_regime_change_trajectory(part, settings, rng)
    if mode == "temporary_shock":
        return _generate_temporary_shock_trajectory(part, settings, rng)
    if mode == "linear_trend":
        return _generate_linear_trend_trajectory(part, settings, rng)

    raise ValueError(f"Unknown non-stationary theta mode: {mode}")


def _make_new_part(
    world: World,
    supplier_risk_factor: float,
    current_period_id: int,
    delivery_day_pattern: Optional[list[str]] = None,
    delivery_call_off_rate: Optional[float] = None,
    base_true_part_theta: Optional[float] = None,
    is_true_theta_stationary: Optional[bool] = None,
    start_period: Optional[int] = None,
    end_period: Optional[int] = None,
) -> Part:
    rng = world.rng
    settings = world.settings

    effective_start_period = current_period_id if start_period is None else start_period
    effective_end_period = (
        settings.n_simulation_periods - 1 if end_period is None else end_period
    )

    effective_delivery_day_pattern = (
        _draw_delivery_day_pattern(settings, rng)
        if delivery_day_pattern is None
        else list(delivery_day_pattern)
    )

    effective_delivery_call_off_rate = (
        _draw_delivery_call_off_rate(settings, rng)
        if delivery_call_off_rate is None
        else delivery_call_off_rate
    )

    effective_base_true_theta = (
        _draw_base_true_theta(settings, rng, supplier_risk_factor)
        if base_true_part_theta is None
        else _clip_probability(base_true_part_theta)
    )

    effective_is_stationary = (
        _make_stationarity_flag(settings, rng)
        if is_true_theta_stationary is None
        else is_true_theta_stationary
    )

    part = Part(
        part_id=world.next_part_id,
        start_period=effective_start_period,
        end_period=effective_end_period,
        delivery_day_pattern=effective_delivery_day_pattern,
        delivery_call_off_rate=effective_delivery_call_off_rate,
        base_true_part_theta=effective_base_true_theta,
        is_true_theta_stationary=effective_is_stationary,
    )
    world.next_part_id += 1

    trajectory = _generate_true_theta_trajectory_for_part(
        part=part,
        settings=settings,
        rng=rng,
    )
    world.set_true_theta_trajectory(trajectory)

    return part


def _make_new_supplier(
    world: World,
    current_period_id: int,
    supplier_risk_factor: Optional[float] = None,
    n_parts: Optional[int] = None,
) -> Supplier:
    rng = world.rng
    settings = world.settings

    effective_supplier_risk_factor = supplier_risk_factor
    if effective_supplier_risk_factor is None:
        effective_supplier_risk_factor = _compute_supplier_risk_factors(
            n_suppliers=1,
            rho=settings.shortage_concentration_rho,
            rng=rng,
        )[0]

    supplier = Supplier(
        supplier_id=world.next_supplier_id,
        supplier_risk_factor=effective_supplier_risk_factor,
    )
    world.next_supplier_id += 1

    n_parts_effective = (
        _draw_supplier_part_count(settings, rng)
        if n_parts is None
        else max(1, int(n_parts))
    )

    for _ in range(n_parts_effective):
        supplier.add_part(
            _make_new_part(
                world=world,
                supplier_risk_factor=supplier.supplier_risk_factor,
                current_period_id=current_period_id,
            )
        )

    return supplier


def _assign_supplier_level_theta_to_parts(
    world: World,
    suppliers: list[Supplier],
    supplier_theta_by_id: dict[int, float],
) -> None:
    """
    Assign the supplier-level theta to all parts of the supplier and refresh
    their trajectories accordingly.

    For stationary parts:
        base_true_part_theta becomes the supplier theta.

    For non-stationary parts:
        base_true_part_theta is also reset to the supplier theta, and a new
        trajectory is generated from that baseline.
    """
    for supplier in suppliers:
        supplier_theta = supplier_theta_by_id[supplier.supplier_id]

        for part in supplier.part:
            part.base_true_part_theta = _clip_probability(supplier_theta)

            trajectory = _generate_true_theta_trajectory_for_part(
                part=part,
                settings=world.settings,
                rng=world.rng,
            )
            world.set_true_theta_trajectory(trajectory)


def _generate_missing_observations_for_period(
    world: World,
    period_obj: Period,
    weeks_per_period: int,
    rng: random.Random,
) -> None:
    for supplier in period_obj.supplier:
        for part in supplier.part:
            _generate_observations_for_part_in_period(
                world=world,
                part=part,
                period_id=period_obj.period_id,
                period_start_date=period_obj.start_date,
                weeks_per_period=weeks_per_period,
                rng=rng,
            )


def _create_initial_period(world: World) -> Period:
    settings = world.settings
    rng = world.rng
    config = settings.world_config  # may be None

    initial_period = Period(
        period_id=0,
        start_date=_compute_period_start_date(
            settings.simulation_start_date,
            0,
            settings.weeks_per_period,
        ),
    )

    # ── Dispatchers ──────────────────────────────────────────────────────────
    dispatcher_cfg = (config or {}).get("dispatchers", {})

    if dispatcher_cfg:
        # Use config order (sorted by dispatcher_id key) to assign IDs
        # deterministically.
        for _, dcfg in sorted(dispatcher_cfg.items()):
            employment_type = dcfg["employment_type"]
            target_workload = _target_workload_from_employment_type(
                employment_type, settings
            )
            initial_period.add_dispatcher(
                Dispatcher(
                    dispatcher_id=world.next_dispatcher_id,
                    employment_type=employment_type,
                    target_workload_per_day=target_workload,
                )
            )
            world.next_dispatcher_id += 1
    else:
        # Legacy random path
        for _ in range(settings.n_dispatchers):
            employment_type = "fulltime" if rng.random() < 0.7 else "parttime"
            target_workload = _target_workload_from_employment_type(
                employment_type, settings
            )
            initial_period.add_dispatcher(
                Dispatcher(
                    dispatcher_id=world.next_dispatcher_id,
                    employment_type=employment_type,
                    target_workload_per_day=target_workload,
                )
            )
            world.next_dispatcher_id += 1

    # ── Suppliers and parts ───────────────────────────────────────────────────
    # Build a flat iterator of part-level config entries in creation order so
    # that _make_new_part can consume them one by one as part_ids are allocated.
    supplier_cfg = (config or {}).get("suppliers", {})
    part_cfg = (config or {}).get("parts", {})

    # Pre-build an ordered list of (n_parts_for_supplier) from config.
    # If supplier_cfg is absent we fall back to the random drawer per supplier.
    if supplier_cfg:
        ordered_supplier_n_parts = [
            scfg["n_parts"]
            for _, scfg in sorted(supplier_cfg.items())
        ]
    else:
        ordered_supplier_n_parts = [None] * settings.n_init_suppliers

    # Pre-build an ordered list of part configs in creation order.
    # Part IDs are allocated sequentially starting from world.next_part_id,
    # so we sort part_cfg by key to match that order.
    if part_cfg:
        ordered_part_cfgs = [
            pcfg for _, pcfg in sorted(part_cfg.items())
        ]
    else:
        ordered_part_cfgs = []

    part_cfg_cursor = 0  # index into ordered_part_cfgs

    suppliers_tmp: list[Supplier] = []

    for supplier_idx in range(settings.n_init_suppliers):
        supplier = Supplier(
            supplier_id=world.next_supplier_id,
            supplier_risk_factor=0.0,
        )
        world.next_supplier_id += 1

        n_parts = ordered_supplier_n_parts[supplier_idx]
        if n_parts is None:
            n_parts = _draw_supplier_part_count(settings, rng)
        n_parts = max(1, int(n_parts))

        for _ in range(n_parts):
            # Pull the next part-level config entry (if any)
            if part_cfg_cursor < len(ordered_part_cfgs):
                pcfg = ordered_part_cfgs[part_cfg_cursor]
                part_cfg_cursor += 1
            else:
                pcfg = {}

            supplier.add_part(
                _make_new_part(
                    world=world,
                    supplier_risk_factor=0.0,
                    current_period_id=0,
                    start_period=0,
                    end_period=settings.n_simulation_periods - 1,
                    # Fixed overrides from config (None → random draw inside _make_new_part)
                    delivery_day_pattern=pcfg.get("delivery_day_pattern"),
                    delivery_call_off_rate=pcfg.get("delivery_call_off_rate"),
                    is_true_theta_stationary=pcfg.get("is_true_theta_stationary"),
                )
            )

        suppliers_tmp.append(supplier)

    # ── Theta assignment: legacy path or hierarchical path ───────────────────
    if settings.use_hierarchical_theta:
        _assign_hierarchical_theta_to_suppliers_and_parts(
            world=world,
            suppliers=suppliers_tmp,
        )

        for supplier in suppliers_tmp:
            initial_period.add_supplier(supplier)

    else:
        # Legacy concentration-weight heuristic (unchanged from original code).
        theta_global = float(default_base_shortage_rate_draw(rng))
        supplier_theta_by_id = _compute_supplier_theta_by_concentration(
            suppliers=suppliers_tmp,
            weeks_per_period=settings.weeks_per_period,
            theta_global=theta_global,
            rho=settings.shortage_concentration_rho,
            rng=rng,
            target_alpha=0.15,
        )

        _assign_supplier_level_theta_to_parts(
            world=world,
            suppliers=suppliers_tmp,
            supplier_theta_by_id=supplier_theta_by_id,
        )

        for supplier in suppliers_tmp:
            supplier.supplier_risk_factor = (
                supplier_theta_by_id[supplier.supplier_id] - theta_global
            )
            initial_period.add_supplier(supplier)

    initial_period.clear_assignments()
    return initial_period


def _create_next_period_from_previous(world: World, next_period_id: int) -> Period:
    previous_period = world.get_period(next_period_id - 1)
    if previous_period is None:
        raise ValueError(f"Previous period {next_period_id - 1} not found.")

    next_period = Period(
        period_id=next_period_id,
        start_date=_compute_period_start_date(
            world.settings.simulation_start_date,
            next_period_id,
            world.settings.weeks_per_period,
        ),
    )

    next_period.dispatcher = _clone_dispatchers_from_previous_period(previous_period)
    next_period.supplier = _clone_suppliers_from_previous_period(previous_period)
    next_period.clear_assignments()

    return next_period


def initialize_world(
    settings: WorldSettings,
    random_seed: Optional[int] = None,
    ks_seed: Optional[int] = None,
    kp_seed: Optional[int] = None,
) -> World:
    """
    Create the world structure without generating any delivery observations.

    Behavior
    --------
    - Creates the initial supplier and dispatcher structure.
    - Creates all parts.
    - Creates and stores the true theta trajectories for all parts.
    - Does not generate any delivery call-offs.
    - Does not perform any assignment.
    - Sets world.current_period = -1, meaning no period has been realized yet.
    """
    _validate_and_fill_default_drawers(settings)

    rng = random.Random(random_seed)
    ks_rng = random.Random(ks_seed)
    kp_rng = random.Random(kp_seed)

    world = World(
        settings=settings,
        current_period=-1,
        rng=rng,
        next_supplier_id=1,
        next_dispatcher_id=1,
        next_part_id=1,
        ks_rng=ks_rng,
        kp_rng=kp_rng,
    )

    initial_period = _create_initial_period(world)
    world.add_period(initial_period)

    settings.current_period = -1
    return world


def _generate_observations_for_part_in_period(
    world: World,
    part: Part,
    period_id: int,
    period_start_date: date,
    weeks_per_period: int,
    rng: random.Random,
) -> None:
    if not part.is_active_in_period(period_id):
        return

    if part.get_period_state(period_id) is not None:
        return

    true_theta_in_period = world.get_true_theta_in_period(part, period_id)

    n_possible_delivery_calloffs = 0
    n_actual_delivery_calloffs = 0
    n_critical_delivery_calloffs = 0

    for week_idx in range(weeks_per_period):
        week_start = period_start_date + timedelta(weeks=week_idx)

        for day in part.delivery_day_pattern:
            n_possible_delivery_calloffs += 1

            delivery_calloff_happens = rng.random() < part.delivery_call_off_rate
            if not delivery_calloff_happens:
                continue

            n_actual_delivery_calloffs += 1
            observation_date = week_start + timedelta(days=WEEKDAY_TO_OFFSET[day])

            is_critical = rng.random() < true_theta_in_period
            if is_critical:
                n_critical_delivery_calloffs += 1

            part.add_delivery_observation(
                DeliveryObservation(
                    period_id=period_id,
                    delivery_day=day,
                    observation_date=observation_date,
                    is_critical=is_critical,
                )
            )

    n_non_critical_delivery_calloffs = (
        n_actual_delivery_calloffs - n_critical_delivery_calloffs
    )

    part.add_period_state(
        PartPeriodState(
            period_id=period_id,
            true_part_theta_in_period=true_theta_in_period,
            n_possible_delivery_calloffs=n_possible_delivery_calloffs,
            n_actual_delivery_calloffs=n_actual_delivery_calloffs,
            n_critical_delivery_calloffs=n_critical_delivery_calloffs,
            n_non_critical_delivery_calloffs=n_non_critical_delivery_calloffs,
        )
    )


def step_world_forward(
    world: World,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
) -> World:
    """
    Advance the simulation by exactly one period and generate realizations
    only for that period.

    Behavior
    --------
    - Increases current_period by 1.
    - Creates the next period structure by cloning the previous one if needed.
    - Generates delivery call-off realizations for the new current period.
    - Updates Bayesian statistics for the Part objects stored in the new period.
    - Does not perform any assignment.
    """
    if alpha_prior <= 0.0:
        raise ValueError("alpha_prior must be > 0.")
    if beta_prior <= 0.0:
        raise ValueError("beta_prior must be > 0.")

    settings = world.settings

    if world.current_period >= settings.n_simulation_periods - 1:
        raise ValueError("World is already at the final simulation period.")

    next_period_id = world.current_period + 1

    current_period_obj = world.get_period(next_period_id)
    if current_period_obj is None:
        current_period_obj = _create_next_period_from_previous(world, next_period_id)
        world.add_period(current_period_obj)

    _generate_missing_observations_for_period(
        world=world,
        period_obj=current_period_obj,
        weeks_per_period=settings.weeks_per_period,
        rng=world.rng,
    )

    update_bayesian_state_for_period_parts(
        period=current_period_obj,
        period_id=next_period_id,
        alpha_prior=alpha_prior,
        beta_prior=beta_prior,
    )

    current_period_obj.clear_assignments()
    world.current_period = next_period_id
    settings.current_period = next_period_id

    return world


def apply_supplier_assignment_to_period(
    world: World,
    assignment_dict: dict[int, int],
    period_id: int | None = None,
    clear_existing_assignment: bool = True,
) -> None:
    """
    Apply a supplier-to-dispatcher assignment to a world period.

    Parameters
    ----------
    world:
        Simulation world.
    assignment_dict:
        Mapping:
            assignment_dict[supplier_id] = dispatcher_id
    period_id:
        Target period to which the assignment should be written.
        If None, world.current_period is used.
    clear_existing_assignment:
        If True, replace the full existing assignment.
        If False, update the existing assignment only for the given suppliers.
    """
    if period_id is None:
        period_id = world.current_period

    if period_id < 0:
        raise ValueError("period_id must be >= 0.")

    period = world.get_period(period_id)
    if period is None:
        raise ValueError(f"Period {period_id} not found.")

    valid_supplier_ids = {supplier.supplier_id for supplier in period.supplier}
    valid_dispatcher_ids = {
        dispatcher.dispatcher_id for dispatcher in period.dispatcher
    }

    for supplier_id, dispatcher_id in assignment_dict.items():
        if supplier_id not in valid_supplier_ids:
            raise ValueError(
                f"Supplier {supplier_id} does not exist in period {period_id}."
            )
        if dispatcher_id not in valid_dispatcher_ids:
            raise ValueError(
                f"Dispatcher {dispatcher_id} does not exist in period {period_id}."
            )

    if clear_existing_assignment:
        period.assignment = dict(assignment_dict)
    else:
        if not hasattr(period, "assignment") or period.assignment is None:
            period.assignment = {}

        for supplier_id, dispatcher_id in assignment_dict.items():
            period.assignment[supplier_id] = dispatcher_id


# ---------------------------------------------------------------------------
# Heterogeneity experiment grid (Section 1.5 of the document)
# ---------------------------------------------------------------------------

def build_heterogeneity_experiment_grid(
    kappa_s_values: list[float] | None = None,
    kappa_p_values: list[float] | None = None,
    m_global: float = 0.10,
) -> list[dict]:
    """
    Build the full grid of heterogeneity regimes for the 3×3 experiment
    described in Section 1.5 of the hierarchical modeling document.

    Default grid (Section 1.5 recommendation)
    ------------------------------------------
        kappa_s ∈ {10, 30, 100}  — high / medium / low supplier heterogeneity
        kappa_p ∈ {5,  20, 100}  — high / medium / low within-supplier heterogeneity

    This yields 9 regimes. Each entry in the returned list is a dict ready to
    be splatted into ``WorldSettings`` (or inspected for logging):

        {
            "use_hierarchical_theta": True,
            "m_global": <float>,
            "kappa_s":  <float>,
            "kappa_p":  <float>,
            "regime_label": "ks10_kp5",
        }

    Usage example
    -------------
        from simulation_functions import build_heterogeneity_experiment_grid
        from simulation_entities import WorldSettings
        from datetime import date

        for regime in build_heterogeneity_experiment_grid():
            settings = WorldSettings(
                n_simulation_periods=12,
                current_period=-1,
                simulation_start_date=date(2024, 1, 1),
                weeks_per_period=4,
                n_init_suppliers=10,
                n_dispatchers=3,
                **{k: v for k, v in regime.items() if k != "regime_label"},
            )
            world = initialize_world(settings, random_seed=42)
            # ... run experiment for this regime ...

    Parameters
    ----------
    kappa_s_values:
        Supplier-level concentration values to sweep.
        Defaults to [10, 30, 100] (Section 1.5).
    kappa_p_values:
        Part-level concentration values to sweep.
        Defaults to [5, 20, 100] (Section 1.5).
    m_global:
        Global average disruption probability. Default 0.10.

    Returns
    -------
    list of dicts, one per (kappa_s, kappa_p) combination.
    """
    if kappa_s_values is None:
        kappa_s_values = [10.0, 30.0, 100.0]
    if kappa_p_values is None:
        kappa_p_values = [5.0, 20.0, 100.0]

    if not (0.0 < m_global < 1.0):
        raise ValueError("m_global must be in (0, 1).")

    regimes: list[dict] = []
    for ks in kappa_s_values:
        for kp in kappa_p_values:
            regime = _create_heterogeneity_regime(
                kappa_s=ks,
                kappa_p=kp,
                m_global=m_global,
            )
            regime["regime_label"] = f"ks{int(ks)}_kp{int(kp)}"
            regimes.append(regime)

    return regimes