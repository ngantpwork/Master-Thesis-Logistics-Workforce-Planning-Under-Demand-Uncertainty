from __future__ import annotations

from .simulation_entities import Part, Period, World


def compute_bayesian_stats_for_part(
    part: Part,
    up_to_period_id: int | None = None,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
) -> dict[str, float | int | None]:
    """
    Compute cumulative Bayesian Beta-Bernoulli statistics for one part.

    Parameters
    ----------
    part:
        Part whose realized delivery observations are analyzed.
    up_to_period_id:
        If given, only observations with obs.period_id <= up_to_period_id are used.
        If None, all realized observations of the part are used.
    alpha_prior:
        Prior alpha parameter. Default is 1.0.
    beta_prior:
        Prior beta parameter. Default is 1.0.

    Returns
    -------
    dict with keys:
        part_id
        n_observations
        n_critical
        n_non_critical
        observed_theta
        posterior_alpha
        posterior_beta
        posterior_mean
        posterior_variance
    """
    if alpha_prior <= 0.0:
        raise ValueError("alpha_prior must be > 0.")
    if beta_prior <= 0.0:
        raise ValueError("beta_prior must be > 0.")

    if up_to_period_id is None:
        observations = list(part.delivery_observations)
    else:
        observations = [
            obs
            for obs in part.delivery_observations
            if obs.period_id <= up_to_period_id
        ]

    n_critical = sum(1 for obs in observations if obs.is_critical)
    n_non_critical = sum(1 for obs in observations if not obs.is_critical)
    n_observations = n_critical + n_non_critical

    observed_theta = None
    if n_observations > 0:
        observed_theta = n_critical / n_observations

    posterior_alpha = alpha_prior + n_critical
    posterior_beta = beta_prior + n_non_critical

    posterior_mass = posterior_alpha + posterior_beta
    posterior_mean = posterior_alpha / posterior_mass
    posterior_variance = (
        posterior_alpha * posterior_beta / (posterior_mass**2 * (posterior_mass + 1.0))
    )

    return {
        "part_id": part.part_id,
        "n_observations": n_observations,
        "n_critical": n_critical,
        "n_non_critical": n_non_critical,
        "observed_theta": observed_theta,
        "posterior_alpha": posterior_alpha,
        "posterior_beta": posterior_beta,
        "posterior_mean": posterior_mean,
        "posterior_variance": posterior_variance,
    }


def compute_bayesian_stats_for_part_in_period(
    part: Part,
    period_id: int,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
) -> dict[str, float | int | None]:
    """
    Compute Bayesian statistics for one part up to and including one period,
    and additionally report the period-specific observed theta.

    Parameters
    ----------
    part:
        Part whose realized delivery observations are analyzed.
    period_id:
        Use all observations with obs.period_id <= period_id.
    alpha_prior:
        Prior alpha parameter. Default is 1.0.
    beta_prior:
        Prior beta parameter. Default is 1.0.

    Returns
    -------
    dict with keys:
        part_id
        period_id
        n_observations
        n_critical
        n_non_critical
        observed_theta
        observed_theta_in_period
        posterior_alpha
        posterior_beta
        posterior_mean
        posterior_variance
    """
    if period_id < 0:
        raise ValueError("period_id must be >= 0.")
    if alpha_prior <= 0.0:
        raise ValueError("alpha_prior must be > 0.")
    if beta_prior <= 0.0:
        raise ValueError("beta_prior must be > 0.")

    cumulative_stats = compute_bayesian_stats_for_part(
        part=part,
        up_to_period_id=period_id,
        alpha_prior=alpha_prior,
        beta_prior=beta_prior,
    )

    period_observations = part.get_observations_in_period(period_id)
    n_period_observations = len(period_observations)

    observed_theta_in_period = None
    if n_period_observations > 0:
        n_period_critical = sum(1 for obs in period_observations if obs.is_critical)
        observed_theta_in_period = n_period_critical / n_period_observations

    return {
        "part_id": part.part_id,
        "period_id": period_id,
        "n_observations": cumulative_stats["n_observations"],
        "n_critical": cumulative_stats["n_critical"],
        "n_non_critical": cumulative_stats["n_non_critical"],
        "observed_theta": cumulative_stats["observed_theta"],
        "observed_theta_in_period": observed_theta_in_period,
        "posterior_alpha": cumulative_stats["posterior_alpha"],
        "posterior_beta": cumulative_stats["posterior_beta"],
        "posterior_mean": cumulative_stats["posterior_mean"],
        "posterior_variance": cumulative_stats["posterior_variance"],
    }


def update_part_period_bayesian_state(
    part: Part,
    period_id: int,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
) -> None:
    """
    Compute Bayesian statistics up to and including period_id and write them
    into the corresponding PartPeriodState.

    This function assumes that PartPeriodState contains the fields:
        observed_theta_in_period
        posterior_alpha
        posterior_beta
        posterior_mean
        posterior_variance
    """
    if alpha_prior <= 0.0:
        raise ValueError("alpha_prior must be > 0.")
    if beta_prior <= 0.0:
        raise ValueError("beta_prior must be > 0.")

    state = part.get_period_state(period_id)
    if state is None:
        raise ValueError(
            f"Part {part.part_id} has no PartPeriodState for period {period_id}."
        )

    stats = compute_bayesian_stats_for_part_in_period(
        part=part,
        period_id=period_id,
        alpha_prior=alpha_prior,
        beta_prior=beta_prior,
    )

    state.observed_theta_in_period = stats["observed_theta_in_period"]
    state.posterior_alpha = float(stats["posterior_alpha"])
    state.posterior_beta = float(stats["posterior_beta"])
    state.posterior_mean = float(stats["posterior_mean"])
    state.posterior_variance = float(stats["posterior_variance"])


def update_bayesian_state_for_period_parts(
    period: Period,
    period_id: int,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
) -> None:
    """
    Update Bayesian state only for the part objects stored in one specific period.

    This is the correct updater for cloned period-based worlds, because it writes
    the Bayesian state into exactly the Part instances that will later be inspected
    in that same period.

    Parameters
    ----------
    period:
        Period whose Part objects should be updated.
    period_id:
        Bayesian stats are computed cumulatively up to and including this period_id.
    alpha_prior:
        Prior alpha parameter. Default is 1.0.
    beta_prior:
        Prior beta parameter. Default is 1.0.
    """
    if alpha_prior <= 0.0:
        raise ValueError("alpha_prior must be > 0.")
    if beta_prior <= 0.0:
        raise ValueError("beta_prior must be > 0.")
    if period_id < 0:
        raise ValueError("period_id must be >= 0.")

    for supplier in period.supplier:
        for part in supplier.part:
            update_part_period_bayesian_state(
                part=part,
                period_id=period_id,
                alpha_prior=alpha_prior,
                beta_prior=beta_prior,
            )


def update_bayesian_state_for_all_parts(
    world: World,
    up_to_period_id: int | None = None,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
) -> None:
    """
    Update stored Bayesian state fields for all unique parts in the world.

    Notes
    -----
    This function is mainly useful for full recomputation or consistency checks.
    In a cloned period-based world, the preferred online update during simulation is
    `update_bayesian_state_for_period_parts(...)`.
    """
    if alpha_prior <= 0.0:
        raise ValueError("alpha_prior must be > 0.")
    if beta_prior <= 0.0:
        raise ValueError("beta_prior must be > 0.")

    unique_parts: dict[int, Part] = {}

    for period in world.period:
        for supplier in period.supplier:
            for part in supplier.part:
                unique_parts[part.part_id] = part

    for part in unique_parts.values():
        for state in part.period_state:
            if up_to_period_id is not None and state.period_id > up_to_period_id:
                continue

            update_part_period_bayesian_state(
                part=part,
                period_id=state.period_id,
                alpha_prior=alpha_prior,
                beta_prior=beta_prior,
            )


def compute_bayesian_stats_for_all_parts(
    world: World,
    up_to_period_id: int | None = None,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
) -> dict[int, dict[str, float | int | None]]:
    """
    Compute cumulative Bayesian Beta-Bernoulli statistics for all parts in the world.

    Parameters
    ----------
    world:
        Simulation world.
    up_to_period_id:
        If given, only observations with obs.period_id <= up_to_period_id are used.
        If None, all realized observations are used.
    alpha_prior:
        Prior alpha parameter. Default is 1.0.
    beta_prior:
        Prior beta parameter. Default is 1.0.

    Returns
    -------
    dict:
        result[part_id] = stats_dict
    """
    if alpha_prior <= 0.0:
        raise ValueError("alpha_prior must be > 0.")
    if beta_prior <= 0.0:
        raise ValueError("beta_prior must be > 0.")

    result: dict[int, dict[str, float | int | None]] = {}
    unique_parts: dict[int, Part] = {}

    for period in world.period:
        for supplier in period.supplier:
            for part in supplier.part:
                unique_parts[part.part_id] = part

    for part_id, part in unique_parts.items():
        result[part_id] = compute_bayesian_stats_for_part(
            part=part,
            up_to_period_id=up_to_period_id,
            alpha_prior=alpha_prior,
            beta_prior=beta_prior,
        )

    return result
