from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable
from datetime import date
import random


WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]


WEEKDAY_TO_OFFSET = {
    "Mon": 0,
    "Tue": 1,
    "Wed": 2,
    "Thu": 3,
    "Fri": 4,
}


@dataclass
class DeliveryObservation:
    """
    One realized delivery call-off observation for one part on one day within one period.
    """

    period_id: int
    delivery_day: str
    observation_date: date
    is_critical: bool


@dataclass
class PartPeriodState:
    """
    Period-specific realized state of a part.

    true_part_theta_in_period always stores the actual true theta that applied
    in this period, regardless of whether the part is stationary or non-stationary.
    """

    period_id: int
    true_part_theta_in_period: float

    n_possible_delivery_calloffs: int = 0
    n_actual_delivery_calloffs: int = 0
    n_critical_delivery_calloffs: int = 0
    n_non_critical_delivery_calloffs: int = 0

    observed_theta_in_period: Optional[float] = None

    posterior_alpha: float = 1.0
    posterior_beta: float = 1.0
    posterior_mean: float = 0.5
    posterior_variance: float = 1.0 / 12.0


@dataclass
class Part:
    """
    A part belongs to one supplier.

    base_true_part_theta:
        Baseline theta of the part. For stationary parts this is the theta in every period.
        For non-stationary parts this is only the baseline/reference level.

    is_true_theta_stationary:
        True if the part has stationary true theta, False otherwise.

    The actual non-stationary theta path is intentionally stored outside the Part object.
    """

    part_id: int
    start_period: int
    end_period: int
    delivery_day_pattern: list[str] = field(default_factory=list)
    delivery_call_off_rate: float = 1.0
    base_true_part_theta: float = 0.0
    is_true_theta_stationary: bool = True
    delivery_observations: list[DeliveryObservation] = field(default_factory=list)
    period_state: list[PartPeriodState] = field(default_factory=list)

    def is_active_in_period(self, period_id: int) -> bool:
        return self.start_period <= period_id <= self.end_period

    def get_period_state(self, period_id: int) -> Optional[PartPeriodState]:
        for state in self.period_state:
            if state.period_id == period_id:
                return state
        return None

    def add_delivery_observation(self, observation: DeliveryObservation) -> None:
        self.delivery_observations.append(observation)

    def add_period_state(self, state: PartPeriodState) -> None:
        self.period_state.append(state)

    def get_observations_in_period(self, period_id: int) -> list[DeliveryObservation]:
        return [obs for obs in self.delivery_observations if obs.period_id == period_id]

    def get_cumulative_delivery_calloff_counts_until_period(
        self, period_id: int
    ) -> tuple[int, int]:
        """
        Return cumulative counts up to and including period_id.

        Returns:
            (n_critical_delivery_calloffs, n_non_critical_delivery_calloffs)
        """
        n_critical_delivery_calloffs = 0
        n_non_critical_delivery_calloffs = 0

        for obs in self.delivery_observations:
            if obs.period_id <= period_id:
                if obs.is_critical:
                    n_critical_delivery_calloffs += 1
                else:
                    n_non_critical_delivery_calloffs += 1

        return n_critical_delivery_calloffs, n_non_critical_delivery_calloffs


@dataclass
class PartTrueThetaTrajectory:
    """
    Stores the actual true theta trajectory of one part across periods.

    For stationary parts, theta_by_period may simply be empty and the base theta
    from Part can be used everywhere.

    For non-stationary parts, theta_by_period should contain the period-specific
    true theta values that override the base theta.
    """

    part_id: int
    theta_by_period: dict[int, float] = field(default_factory=dict)

    def get_theta_for_period(
        self, period_id: int, base_true_part_theta: float
    ) -> float:
        return self.theta_by_period.get(period_id, base_true_part_theta)


@dataclass
class Supplier:
    """
    A supplier owns multiple parts.

    supplier_theta:
        Supplier-level delivery risk parameter θs drawn from the hierarchical
        Beta prior.  Set only when use_hierarchical_theta=True; otherwise None.
        Parts use this value as the mean of their own Beta prior so that
        within-supplier heterogeneity is controlled by kappa_p.
    """

    supplier_id: int
    part: list[Part] = field(default_factory=list)
    supplier_risk_factor: float = 0.0
    supplier_theta: Optional[float] = None

    def add_part(self, part: Part) -> None:
        self.part.append(part)

    @property
    def n_parts(self) -> int:
        return len(self.part)


@dataclass
class Dispatcher:
    """
    A dispatcher can be assigned suppliers.

    target_workload_per_day:
        Target workload per working day in minutes.
        This value is used as the reference capacity in workload balancing
        and optimization models.
    """

    dispatcher_id: int
    employment_type: str
    target_workload_per_day: float
    assigned_supplier_ids: list[int] = field(default_factory=list)

    def assign_supplier(self, supplier_id: int) -> None:
        if supplier_id not in self.assigned_supplier_ids:
            self.assigned_supplier_ids.append(supplier_id)

    def clear_assigned_suppliers(self) -> None:
        self.assigned_supplier_ids = []


@dataclass
class Period:
    """
    One simulation period.
    A period can contain multiple weeks.

    assignment stores the supplier-to-dispatcher assignment of this period:
        assignment[supplier_id] = dispatcher_id
    """

    period_id: int
    start_date: date
    supplier: list[Supplier] = field(default_factory=list)
    dispatcher: list[Dispatcher] = field(default_factory=list)
    assignment: dict[int, int] = field(default_factory=dict)

    def add_supplier(self, supplier: Supplier) -> None:
        self.supplier.append(supplier)

    def add_dispatcher(self, dispatcher: Dispatcher) -> None:
        self.dispatcher.append(dispatcher)

    def set_assignment(self, supplier_id: int, dispatcher_id: int) -> None:
        self.assignment[supplier_id] = dispatcher_id

        for dispatcher in self.dispatcher:
            if dispatcher.dispatcher_id == dispatcher_id:
                dispatcher.assign_supplier(supplier_id)
                break

    def clear_assignments(self) -> None:
        self.assignment = {}
        for dispatcher in self.dispatcher:
            dispatcher.clear_assigned_suppliers()

    def get_assigned_dispatcher(self, supplier_id: int) -> Optional[int]:
        return self.assignment.get(supplier_id)

    def get_supplier(self, supplier_id: int) -> Optional[Supplier]:
        for supplier in self.supplier:
            if supplier.supplier_id == supplier_id:
                return supplier
        return None

    def get_dispatcher(self, dispatcher_id: int) -> Optional[Dispatcher]:
        for dispatcher in self.dispatcher:
            if dispatcher.dispatcher_id == dispatcher_id:
                return dispatcher
        return None

    def get_part(self, part_id: int) -> Optional[Part]:
        for supplier in self.supplier:
            for part in supplier.part:
                if part.part_id == part_id:
                    return part
        return None

    def get_supplier_of_part(self, part_id: int) -> Optional[Supplier]:
        for supplier in self.supplier:
            for part in supplier.part:
                if part.part_id == part_id:
                    return supplier
        return None


@dataclass
class WorldSettings:
    """
    Global settings and generation hooks for the simulation.

    non_stationary_theta_share:
        Share of parts for which a non-stationary true theta trajectory should be created.

    allowed_non_stationary_theta_modes:
        Which modes may be used when generating non-stationary trajectories.
        The actual generated trajectory itself should be stored in PartTrueThetaTrajectory.
    """

    n_simulation_periods: int
    current_period: int
    simulation_start_date: date
    weeks_per_period: int

    n_init_suppliers: int
    n_dispatchers: int
    n_new_suppliers_per_period: int = 0
    n_new_dispatchers_per_period: int = 0

    target_workload_dispatcher_fulltime_per_day: float = 300
    target_workload_dispatcher_parttime_per_day: float = 180

    shortage_concentration_rho: float = 0.0
    freeze_structure_across_horizon: bool = False

    supplier_part_count_sampler: Optional[Callable[[random.Random], int]] = None
    delivery_day_pattern_sampler: Optional[Callable[[random.Random], list[str]]] = None
    base_shortage_rate_sampler: Optional[Callable[[random.Random], float]] = None
    delivery_call_off_rate_sampler: Optional[Callable[[random.Random], float]] = None

    non_stationary_theta_share: float = 0.0
    allowed_non_stationary_theta_modes: tuple[str, ...] = (
        "regime_change",
        "temporary_shock",
        "linear_trend",
    )

    # -----------------------------------------------------------------------
    # Hierarchical Beta-Bernoulli parameters (Section 1 — new)
    #
    # When use_hierarchical_theta=True, the world initialization switches from
    # the legacy concentration-weight heuristic to a proper two-level
    # hierarchical sampling procedure:
    #
    #   θs  ~ Beta(m_global * kappa_s,  (1 - m_global) * kappa_s)
    #   θps ~ Beta(θs * kappa_p,        (1 - θs)       * kappa_p)
    #
    # kappa_s: concentration across suppliers.
    #   Small  → high supplier heterogeneity  (e.g. kappa_s = 10)
    #   Large  → homogeneous supplier pool    (e.g. kappa_s = 100)
    #
    # kappa_p: concentration of parts within a supplier.
    #   Small  → parts differ a lot within same supplier  (e.g. kappa_p = 5)
    #   Large  → parts are nearly identical within supplier (e.g. kappa_p = 100)
    #
    # m_global: average disruption probability across the supplier population.
    #   Replaces/extends the role of base_shortage_rate_sampler for the
    #   hierarchical path. Must be in (0, 1).
    # -----------------------------------------------------------------------
    use_hierarchical_theta: bool = False
    m_global: float = 0.10
    kappa_s: float = 30.0
    kappa_p: float = 20.0

    # -----------------------------------------------------------------------
    # Fixed world configuration (for reproducible comparisons)
    #
    # When world_config is provided, the world initializer uses its values
    # instead of sampling from the random distributions above.  This pins
    # the structural skeleton (supplier part counts, part delivery patterns,
    # dispatcher employment types) so that experiment runs differ only in
    # the stochastic parameters being studied (e.g. kappa_s / kappa_p).
    #
    # Expected format
    # ---------------
    #   world_config = {
    #       "suppliers": {
    #           <supplier_id: int>: {
    #               "n_parts": <int>,          # required
    #           },
    #           ...
    #       },
    #       "parts": {
    #           <part_id: int>: {
    #               "delivery_day_pattern": <list[str]>,   # required
    #               # optional overrides:
    #               "delivery_call_off_rate": <float>,
    #               "is_true_theta_stationary": <bool>,
    #           },
    #           ...
    #       },
    #       "dispatchers": {
    #           <dispatcher_id: int>: {
    #               "employment_type": "fulltime" | "parttime",  # required
    #           },
    #           ...
    #       },
    #   }
    #
    # Part IDs in the "parts" sub-dict are assigned *in order of creation*
    # (supplier 1 → part 1, part 2, …; supplier 2 → part 3, …).  The keys
    # in world_config["parts"] must therefore match the part_ids that
    # initialize_world will allocate (starting from 1).
    # -----------------------------------------------------------------------
    world_config: Optional[dict] = None

    regime_change_min_period: int = 1
    regime_change_max_period: Optional[int] = None
    regime_change_target_sampler: Optional[Callable[[random.Random], float]] = None

    temporary_shock_min_start_period: int = 1
    temporary_shock_max_start_period: Optional[int] = None
    temporary_shock_min_duration_periods: int = 1
    temporary_shock_max_duration_periods: int = 1
    temporary_shock_value_sampler: Optional[Callable[[random.Random], float]] = None

    linear_trend_final_period: Optional[int] = None
    linear_trend_target_sampler: Optional[Callable[[random.Random], float]] = None


@dataclass
class World:
    """
    Full simulation world across all periods.

    true_theta_trajectory_by_part_id:
        Stores the period-specific true theta paths of parts.
        Stationary parts may simply have no entry here.
    """

    settings: WorldSettings
    current_period: int
    period: list[Period] = field(default_factory=list)

    rng: random.Random = field(default_factory=random.Random, repr=False)
    ks_rng: random.Random = field(default_factory=random.Random, repr=False)
    kp_rng: random.Random = field(default_factory=random.Random, repr=False)
    next_supplier_id: int = 1
    next_dispatcher_id: int = 1
    next_part_id: int = 1

    true_theta_trajectory_by_part_id: dict[int, PartTrueThetaTrajectory] = field(
        default_factory=dict
    )

    def add_period(self, period: Period) -> None:
        self.period.append(period)

    def get_current_period(self) -> Optional[Period]:
        for p in self.period:
            if p.period_id == self.current_period:
                return p
        return None

    def get_period(self, period_id: int) -> Optional[Period]:
        for p in self.period:
            if p.period_id == period_id:
                return p
        return None

    def get_part_in_period(self, period_id: int, part_id: int) -> Optional[Part]:
        period_obj = self.get_period(period_id)
        if period_obj is None:
            return None
        return period_obj.get_part(part_id)

    def get_part_in_current_period(self, part_id: int) -> Optional[Part]:
        current_period_obj = self.get_current_period()
        if current_period_obj is None:
            return None
        return current_period_obj.get_part(part_id)

    def get_supplier_in_period(
        self, period_id: int, supplier_id: int
    ) -> Optional[Supplier]:
        period_obj = self.get_period(period_id)
        if period_obj is None:
            return None
        return period_obj.get_supplier(supplier_id)

    def get_dispatcher_in_period(
        self, period_id: int, dispatcher_id: int
    ) -> Optional[Dispatcher]:
        period_obj = self.get_period(period_id)
        if period_obj is None:
            return None
        return period_obj.get_dispatcher(dispatcher_id)

    def set_true_theta_trajectory(self, trajectory: PartTrueThetaTrajectory) -> None:
        self.true_theta_trajectory_by_part_id[trajectory.part_id] = trajectory

    def get_true_theta_in_period(self, part: Part, period_id: int) -> float:
        """
        Return the true theta of a part in a given period.

        - stationary part: base_true_part_theta
        - non-stationary part: use stored trajectory if available, otherwise fallback to base_true_part_theta
        """
        if part.is_true_theta_stationary:
            return part.base_true_part_theta

        trajectory = self.true_theta_trajectory_by_part_id.get(part.part_id)
        if trajectory is None:
            return part.base_true_part_theta

        return trajectory.get_theta_for_period(
            period_id=period_id,
            base_true_part_theta=part.base_true_part_theta,
        )