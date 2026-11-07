from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from datetime import timedelta
from .simulation_entities import World, WEEKDAY_TO_OFFSET


def plot_part_delivery_calloffs_in_period(
    world: World,
    part_id: int,
    period_id: int,
    weeks_per_period: int | None = None,
) -> None:
    """
    Plot actual delivery call-offs of one part within one period.

    Parameters
    ----------
    world:
        Simulation world.
    part_id:
        ID of the part to plot.
    period_id:
        Period whose observations should be shown.
    weeks_per_period:
        Optional manual override. If None, world.settings.weeks_per_period is used.

    Notes
    -----
    - X-axis shows week and weekday.
    - Non-critical delivery call-offs are shown in green.
    - Critical delivery call-offs are shown in red.
    """
    if weeks_per_period is None:
        weeks_per_period = world.settings.weeks_per_period

    part = world.get_part_in_period(period_id=period_id, part_id=part_id)
    if part is None:
        raise ValueError(f"Part {part_id} not found in period {period_id}.")

    observations = part.get_observations_in_period(period_id)

    if len(observations) == 0:
        raise ValueError(
            f"No delivery observations found for part {part_id} in period {period_id}."
        )

    observations = sorted(observations, key=lambda obs: obs.observation_date)

    x_positions = list(range(len(observations)))
    colors = ["red" if obs.is_critical else "green" for obs in observations]
    labels: list[str] = []

    period_observation_dates = [obs.observation_date for obs in observations]
    period_start_date = min(period_observation_dates)

    for obs in observations:
        day_diff = (obs.observation_date - period_start_date).days
        week_idx = (day_diff // 7) + 1
        week_idx = min(week_idx, weeks_per_period)
        labels.append(f"W{week_idx}-{obs.delivery_day}")

    n_critical = sum(obs.is_critical for obs in observations)
    n_non_critical = len(observations) - n_critical

    plt.figure(figsize=(max(12, len(observations) * 0.35), 4))
    plt.scatter(x_positions, [1] * len(observations), c=colors, s=70)
    plt.yticks([])
    plt.xticks(x_positions, labels, rotation=90)
    plt.xlabel("Week and Weekday")
    plt.title(
        f"Part {part_id}, Period {period_id}, "
        f"Critical delivery call-offs: {n_critical}, "
        f"Non-critical delivery call-offs: {n_non_critical}"
    )
    plt.tight_layout()
    plt.show()


def plot_part_delivery_calloffs_in_period(
    world: World,
    part_id: int,
    period_id: int,
    weeks_per_period: int | None = None,
) -> None:
    """
    Plot actual delivery call-offs of one part within one period.
    """
    if weeks_per_period is None:
        weeks_per_period = world.settings.weeks_per_period

    part = world.get_part_in_period(period_id=period_id, part_id=part_id)
    if part is None:
        raise ValueError(f"Part {part_id} not found in period {period_id}.")

    observations = part.get_observations_in_period(period_id)

    if len(observations) == 0:
        raise ValueError(
            f"No delivery observations found for part {part_id} in period {period_id}."
        )

    observations = sorted(observations, key=lambda obs: obs.observation_date)

    x_positions = list(range(len(observations)))
    colors = ["red" if obs.is_critical else "green" for obs in observations]
    labels: list[str] = []

    period_observation_dates = [obs.observation_date for obs in observations]
    period_start_date = min(period_observation_dates)

    for obs in observations:
        day_diff = (obs.observation_date - period_start_date).days
        week_idx = (day_diff // 7) + 1
        week_idx = min(week_idx, weeks_per_period)
        labels.append(f"W{week_idx}-{obs.delivery_day}")

    n_critical = sum(obs.is_critical for obs in observations)
    n_non_critical = len(observations) - n_critical

    plt.figure(figsize=(max(12, len(observations) * 0.35), 4))
    plt.scatter(x_positions, [1] * len(observations), c=colors, s=70)
    plt.yticks([])
    plt.xticks(x_positions, labels, rotation=90)
    plt.xlabel("Week and Weekday")
    plt.title(
        f"Part {part_id}, Period {period_id}, "
        f"Critical delivery call-offs: {n_critical}, "
        f"Non-critical delivery call-offs: {n_non_critical}"
    )
    plt.tight_layout()
    plt.show()


def plot_supplier_delivery_calloffs_in_period(
    world: World,
    supplier_id: int,
    period_id: int,
    weeks_per_period: int | None = None,
    show_parts_without_observations: bool = True,
) -> None:
    """
    Plot actual delivery call-offs of all parts of one supplier within one period.

    Parameters
    ----------
    world:
        Simulation world.
    supplier_id:
        ID of the supplier to plot.
    period_id:
        Period whose observations should be shown.
    weeks_per_period:
        Optional manual override. If None, world.settings.weeks_per_period is used.
    show_parts_without_observations:
        If True, parts without observations are still shown as empty rows.
        If False, only parts with at least one observation are shown.

    Notes
    -----
    - Each horizontal row corresponds to one part.
    - Non-critical delivery call-offs are shown in green.
    - Critical delivery call-offs are shown in red.
    """
    if weeks_per_period is None:
        weeks_per_period = world.settings.weeks_per_period

    supplier = world.get_supplier_in_period(
        period_id=period_id, supplier_id=supplier_id
    )
    if supplier is None:
        raise ValueError(f"Supplier {supplier_id} not found in period {period_id}.")

    parts = sorted(supplier.part, key=lambda part: part.part_id)

    if not show_parts_without_observations:
        parts = [
            part
            for part in parts
            if len(part.get_observations_in_period(period_id)) > 0
        ]

    if len(parts) == 0:
        raise ValueError(
            f"No parts available for supplier {supplier_id} in period {period_id}."
        )

    plotted_any_observation = False
    x_positions: list[int] = []
    y_positions: list[int] = []
    colors: list[str] = []

    x_tick_labels = []
    for week_idx in range(1, weeks_per_period + 1):
        for day in ["Mon", "Tue", "Wed", "Thu", "Fri"]:
            x_tick_labels.append(f"W{week_idx}-{day}")

    x_label_to_pos = {label: idx for idx, label in enumerate(x_tick_labels)}

    y_tick_positions = []
    y_tick_labels = []

    n_total_critical = 0
    n_total_non_critical = 0

    for row_idx, part in enumerate(parts):
        y_value = row_idx + 1
        y_tick_positions.append(y_value)
        y_tick_labels.append(f"Part {part.part_id}")

        observations = sorted(
            part.get_observations_in_period(period_id),
            key=lambda obs: obs.observation_date,
        )

        for obs in observations:
            state = part.get_period_state(period_id)
            if state is None:
                continue

            if state.n_possible_delivery_calloffs <= 0:
                continue

            period_start_date = world.get_period(period_id).start_date
            day_diff = (obs.observation_date - period_start_date).days
            week_idx = (day_diff // 7) + 1
            week_idx = min(max(week_idx, 1), weeks_per_period)

            label = f"W{week_idx}-{obs.delivery_day}"
            x_value = x_label_to_pos[label]

            x_positions.append(x_value)
            y_positions.append(y_value)
            colors.append("red" if obs.is_critical else "green")

            if obs.is_critical:
                n_total_critical += 1
            else:
                n_total_non_critical += 1

            plotted_any_observation = True

    if not plotted_any_observation:
        raise ValueError(
            f"No delivery observations found for supplier {supplier_id} in period {period_id}."
        )

    figure_height = max(4, len(parts) * 0.45)

    plt.figure(figsize=(max(12, len(x_tick_labels) * 0.35), figure_height))
    plt.scatter(x_positions, y_positions, c=colors, s=70)

    plt.yticks(y_tick_positions, y_tick_labels)
    plt.xticks(list(range(len(x_tick_labels))), x_tick_labels, rotation=90)
    plt.xlabel("Week and Weekday")
    plt.ylabel("Parts")
    plt.title(
        f"Supplier {supplier_id}, Period {period_id}, "
        f"Critical delivery call-offs: {n_total_critical}, "
        f"Non-critical delivery call-offs: {n_total_non_critical}"
    )
    plt.tight_layout()
    plt.show()


def plot_part_true_theta_trajectory(
    world: World,
    part_id: int,
    start_period: int | None = None,
    end_period: int | None = None,
    show_markers: bool = True,
) -> None:
    """
    Plot the full true theta trajectory of one part across its lifetime.

    Parameters
    ----------
    world:
        Simulation world.
    part_id:
        ID of the part whose true theta trajectory should be plotted.
    start_period:
        Optional first period to include. If None, the part start period is used.
    end_period:
        Optional last period to include. If None, the part end period is used.
    show_markers:
        If True, show markers on the line.

    Notes
    -----
    This plot uses the stored full trajectory of the part, not only already
    simulated periods. Therefore it can show the complete theta path from
    part.start_period to part.end_period immediately after world initialization.
    """
    reference_part = None
    for period in world.period:
        candidate = period.get_part(part_id)
        if candidate is not None:
            reference_part = candidate
            break

    if reference_part is None:
        raise ValueError(f"Part {part_id} not found in the world.")

    if start_period is None:
        start_period = reference_part.start_period

    if end_period is None:
        end_period = reference_part.end_period

    if start_period > end_period:
        raise ValueError("start_period must be <= end_period.")

    trajectory = world.true_theta_trajectory_by_part_id.get(part_id)
    if trajectory is None:
        raise ValueError(f"No true theta trajectory found for part {part_id}.")

    period_ids = list(range(start_period, end_period + 1))
    theta_values = [
        trajectory.get_theta_for_period(
            period_id=period_id,
            base_true_part_theta=reference_part.base_true_part_theta,
        )
        for period_id in period_ids
    ]

    plt.figure(figsize=(8, 4))
    plt.plot(
        period_ids,
        theta_values,
        marker="o" if show_markers else None,
        linewidth=2,
    )
    plt.xlabel("Period")
    plt.ylabel("True theta")
    plt.title(
        f"True theta trajectory of part {part_id}, "
        f"stationary={reference_part.is_true_theta_stationary}"
    )
    plt.xticks(period_ids)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_sorted_critical_calloffs_per_supplier(
    world: World,
    period_id: int,
    normalize: bool = False,
    show_cumulative: bool = True,
) -> None:
    """
    Plot critical delivery call-offs per supplier, sorted descending.

    Parameters
    ----------
    world:
        Simulation world.
    period_id:
        Period to analyze.
    normalize:
        If True, show share instead of absolute counts.
    show_cumulative:
        If True, also plot cumulative curve (Lorenz-style insight).

    Interpretation
    --------------
    - Steep drop: strong concentration (few suppliers dominate)
    - Flat curve: more uniform distribution
    """
    period = world.get_period(period_id)
    if period is None:
        raise ValueError(f"Period {period_id} not found.")

    supplier_critical = []

    for supplier in period.supplier:
        n_critical = 0

        for part in supplier.part:
            state = part.get_period_state(period_id)
            if state is None:
                continue
            n_critical += state.n_critical_delivery_calloffs

        supplier_critical.append((supplier.supplier_id, n_critical))

    if len(supplier_critical) == 0:
        raise ValueError("No supplier data available.")

    # Sort descending by critical call-offs
    supplier_critical.sort(key=lambda x: x[1], reverse=True)

    supplier_ids = [x[0] for x in supplier_critical]
    critical_values = [x[1] for x in supplier_critical]

    total_critical = sum(critical_values)

    if normalize:
        if total_critical > 0:
            values = [v / total_critical for v in critical_values]
        else:
            values = [0.0 for _ in critical_values]
        ylabel = "Share of critical delivery call-offs"
    else:
        values = critical_values
        ylabel = "Critical delivery call-offs"

    x_positions = list(range(len(values)))

    plt.figure(figsize=(10, 5))
    plt.bar(x_positions, values)

    plt.xlabel("Suppliers (sorted by critical call-offs)")
    plt.ylabel(ylabel)
    plt.title(f"Critical delivery call-offs per supplier, period {period_id}")

    plt.xticks(
        x_positions,
        [f"S{sid}" for sid in supplier_ids],
        rotation=90,
    )

    if show_cumulative:
        cumulative = []
        running_sum = 0.0

        for v in values:
            running_sum += v
            cumulative.append(running_sum)

        if normalize:
            cumulative_y = cumulative
        else:
            if total_critical > 0:
                cumulative_y = [c / total_critical for c in cumulative]
            else:
                cumulative_y = [0.0 for _ in cumulative]

        ax2 = plt.gca().twinx()
        ax2.plot(x_positions, cumulative_y, linestyle="--")
        ax2.set_ylabel("Cumulative share")

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_number_of_parts_per_supplier(
    world: World,
    period_id: int,
) -> None:
    """
    Plot the number of parts per supplier in a given period.

    - x-axis: supplier_id
    - y-axis: number of parts
    - value labels shown above each bar
    - y-axis ticks are integers
    """
    period = world.get_period(period_id)
    if period is None:
        raise ValueError(f"Period {period_id} not found.")

    suppliers = sorted(period.supplier, key=lambda s: s.supplier_id)
    if len(suppliers) == 0:
        raise ValueError(f"No suppliers found in period {period_id}.")

    supplier_ids = [s.supplier_id for s in suppliers]
    n_parts = [s.n_parts for s in suppliers]

    x_positions = list(range(len(suppliers)))

    plt.figure(figsize=(10, 5))
    bars = plt.bar(x_positions, n_parts)

    # Labels über den Balken
    for x, value in zip(x_positions, n_parts):
        plt.text(
            x,
            value + 0.1,
            str(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xlabel("Supplier ID")
    plt.ylabel("Number of parts")
    plt.title(f"Number of parts per supplier, period {period_id}")

    plt.xticks(
        x_positions,
        [f"S{sid}" for sid in supplier_ids],
        rotation=90,
    )

    # Ganzzahlige y-Ticks erzwingen
    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_supplier_part_counts_colored_by_dispatcher(
    world: World,
    period_id: int,
    show_dispatcher_label: bool = True,
) -> None:
    """
    Plot the number of parts per supplier in a given period and color each supplier
    by the assigned dispatcher.

    Suppliers are sorted descending by number of parts, tie-break by supplier_id.
    """
    period = world.get_period(period_id)
    if period is None:
        raise ValueError(f"Period {period_id} not found.")

    if len(period.assignment) == 0:
        raise ValueError(
            f"No assignment found in period {period_id}. "
            "Run the assignment logic first."
        )

    supplier_data = []
    for supplier in period.supplier:
        dispatcher_id = period.get_assigned_dispatcher(supplier.supplier_id)
        supplier_data.append(
            {
                "supplier_id": supplier.supplier_id,
                "n_parts": supplier.n_parts,
                "dispatcher_id": dispatcher_id,
            }
        )

    supplier_data.sort(key=lambda row: (-row["n_parts"], row["supplier_id"]))

    supplier_ids = [row["supplier_id"] for row in supplier_data]
    n_parts = [row["n_parts"] for row in supplier_data]
    assigned_dispatchers = [row["dispatcher_id"] for row in supplier_data]

    dispatcher_ids = sorted(
        {
            dispatcher_id
            for dispatcher_id in assigned_dispatchers
            if dispatcher_id is not None
        }
    )

    if len(dispatcher_ids) == 0:
        raise ValueError(f"No assigned dispatchers found in period {period_id}.")

    cmap = plt.get_cmap("tab10")
    dispatcher_to_color = {
        dispatcher_id: cmap(idx % 10)
        for idx, dispatcher_id in enumerate(dispatcher_ids)
    }

    bar_colors = [
        dispatcher_to_color[dispatcher_id] if dispatcher_id is not None else "lightgray"
        for dispatcher_id in assigned_dispatchers
    ]

    x_positions = list(range(len(supplier_ids)))

    plt.figure(figsize=(11, 5))
    bars = plt.bar(x_positions, n_parts, color=bar_colors)

    for x, value, dispatcher_id in zip(x_positions, n_parts, assigned_dispatchers):
        plt.text(
            x,
            value + 0.08,
            str(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )

        if show_dispatcher_label and dispatcher_id is not None:
            plt.text(
                x,
                value / 2 if value > 0 else 0.1,
                f"D{dispatcher_id}",
                ha="center",
                va="center",
                fontsize=9,
                color="black",
            )

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=dispatcher_to_color[dispatcher_id])
        for dispatcher_id in dispatcher_ids
    ]
    legend_labels = [f"Dispatcher {dispatcher_id}" for dispatcher_id in dispatcher_ids]
    plt.legend(legend_handles, legend_labels, title="Assignment")

    plt.xlabel("Suppliers, sorted by number of parts")
    plt.ylabel("Number of parts")
    plt.title(
        f"Supplier part counts colored by assigned dispatcher, period {period_id}"
    )

    plt.xticks(
        x_positions,
        [f"S{supplier_id}" for supplier_id in supplier_ids],
        rotation=90,
    )

    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_dispatcher_delivery_calloffs_in_period(
    world: World,
    dispatcher_id: int,
    period_id: int,
    weeks_per_period: int | None = None,
    show_parts_without_observations: bool = True,
    group_labels_by_supplier: bool = True,
) -> None:
    """
    Plot actual delivery call-offs of all parts assigned to one dispatcher within one period.

    Parameters
    ----------
    world:
        Simulation world.
    dispatcher_id:
        ID of the dispatcher to plot.
    period_id:
        Period whose observations should be shown.
    weeks_per_period:
        Optional manual override. If None, world.settings.weeks_per_period is used.
    show_parts_without_observations:
        If True, parts without observations are still shown as empty rows.
        If False, only parts with at least one observation are shown.
    group_labels_by_supplier:
        If True, y-axis labels contain both supplier and part information.

    Notes
    -----
    - Each horizontal row corresponds to one part assigned indirectly via supplier assignment.
    - Non-critical delivery call-offs are shown in green.
    - Critical delivery call-offs are shown in red.
    """
    if weeks_per_period is None:
        weeks_per_period = world.settings.weeks_per_period

    period = world.get_period(period_id)
    if period is None:
        raise ValueError(f"Period {period_id} not found.")

    dispatcher = period.get_dispatcher(dispatcher_id)
    if dispatcher is None:
        raise ValueError(f"Dispatcher {dispatcher_id} not found in period {period_id}.")

    if len(period.assignment) == 0:
        raise ValueError(
            f"No assignment found in period {period_id}. "
            "Run the assignment logic first."
        )

    assigned_supplier_ids = sorted(
        supplier_id
        for supplier_id, assigned_dispatcher_id in period.assignment.items()
        if assigned_dispatcher_id == dispatcher_id
    )

    if len(assigned_supplier_ids) == 0:
        raise ValueError(
            f"No suppliers assigned to dispatcher {dispatcher_id} in period {period_id}."
        )

    part_rows: list[tuple[int, int, object]] = []
    for supplier_id in assigned_supplier_ids:
        supplier = period.get_supplier(supplier_id)
        if supplier is None:
            continue

        for part in sorted(supplier.part, key=lambda part: part.part_id):
            observations = part.get_observations_in_period(period_id)
            if show_parts_without_observations or len(observations) > 0:
                part_rows.append((supplier_id, part.part_id, part))

    if len(part_rows) == 0:
        raise ValueError(
            f"No parts available for dispatcher {dispatcher_id} in period {period_id}."
        )

    plotted_any_observation = False
    x_positions: list[int] = []
    y_positions: list[int] = []
    colors: list[str] = []

    x_tick_labels = []
    for week_idx in range(1, weeks_per_period + 1):
        for day in ["Mon", "Tue", "Wed", "Thu", "Fri"]:
            x_tick_labels.append(f"W{week_idx}-{day}")

    x_label_to_pos = {label: idx for idx, label in enumerate(x_tick_labels)}

    y_tick_positions = []
    y_tick_labels = []

    n_total_critical = 0
    n_total_non_critical = 0

    for row_idx, (supplier_id, part_id, part) in enumerate(part_rows):
        y_value = row_idx + 1
        y_tick_positions.append(y_value)

        if group_labels_by_supplier:
            y_tick_labels.append(f"S{supplier_id}-P{part_id}")
        else:
            y_tick_labels.append(f"Part {part_id}")

        observations = sorted(
            part.get_observations_in_period(period_id),
            key=lambda obs: obs.observation_date,
        )

        for obs in observations:
            state = part.get_period_state(period_id)
            if state is None:
                continue

            if state.n_possible_delivery_calloffs <= 0:
                continue

            period_start_date = period.start_date
            day_diff = (obs.observation_date - period_start_date).days
            week_idx = (day_diff // 7) + 1
            week_idx = min(max(week_idx, 1), weeks_per_period)

            label = f"W{week_idx}-{obs.delivery_day}"
            x_value = x_label_to_pos[label]

            x_positions.append(x_value)
            y_positions.append(y_value)
            colors.append("red" if obs.is_critical else "green")

            if obs.is_critical:
                n_total_critical += 1
            else:
                n_total_non_critical += 1

            plotted_any_observation = True

    if not plotted_any_observation:
        raise ValueError(
            f"No delivery observations found for dispatcher {dispatcher_id} in period {period_id}."
        )

    figure_height = max(4, len(part_rows) * 0.4)

    plt.figure(figsize=(max(12, len(x_tick_labels) * 0.35), figure_height))
    plt.scatter(x_positions, y_positions, c=colors, s=70)

    plt.yticks(y_tick_positions, y_tick_labels)
    plt.xticks(list(range(len(x_tick_labels))), x_tick_labels, rotation=90)
    plt.xlabel("Week and Weekday")
    plt.ylabel("Assigned parts")
    plt.title(
        f"Dispatcher {dispatcher_id}, Period {period_id}, "
        f"Critical delivery call-offs: {n_total_critical}, "
        f"Non-critical delivery call-offs: {n_total_non_critical}"
    )
    plt.tight_layout()
    plt.show()


def plot_dispatcher_daily_workload_in_period(
    world: World,
    dispatcher_id: int,
    period_id: int,
    critical_case_duration_minutes: float,
    target_workload_per_day_minutes: float,
) -> None:
    """
    Plot daily workload of one dispatcher within one period as a bar chart.

    Parameters
    ----------
    world:
        Simulation world.
    dispatcher_id:
        ID of the dispatcher to analyze.
    period_id:
        Period whose daily workload should be shown.
    critical_case_duration_minutes:
        Workload contribution in minutes of one critical case.
    target_workload_per_day_minutes:
        Target workload per day in minutes, shown as a dashed horizontal line.

    Notes
    -----
    - Only critical delivery call-offs generate workload.
    - Non-critical delivery call-offs generate zero workload.
    - Supplier-to-dispatcher assignment must already exist for the period.
    """
    if critical_case_duration_minutes < 0:
        raise ValueError("critical_case_duration_minutes must be >= 0.")

    if target_workload_per_day_minutes < 0:
        raise ValueError("target_workload_per_day_minutes must be >= 0.")

    period = world.get_period(period_id)
    if period is None:
        raise ValueError(f"Period {period_id} not found.")

    dispatcher = period.get_dispatcher(dispatcher_id)
    if dispatcher is None:
        raise ValueError(f"Dispatcher {dispatcher_id} not found in period {period_id}.")

    if len(period.assignment) == 0:
        raise ValueError(
            f"No assignment found in period {period_id}. "
            "Run the assignment logic first."
        )

    assigned_supplier_ids = sorted(
        supplier_id
        for supplier_id, assigned_dispatcher_id in period.assignment.items()
        if assigned_dispatcher_id == dispatcher_id
    )

    if len(assigned_supplier_ids) == 0:
        raise ValueError(
            f"No suppliers assigned to dispatcher {dispatcher_id} in period {period_id}."
        )

    all_dates_in_period = []
    for week_idx in range(world.settings.weeks_per_period):
        week_start = period.start_date + timedelta(weeks=week_idx)
        for day in ["Mon", "Tue", "Wed", "Thu", "Fri"]:
            all_dates_in_period.append(
                week_start + timedelta(days=WEEKDAY_TO_OFFSET[day])
            )

    workload_by_date = {day_date: 0.0 for day_date in all_dates_in_period}

    n_critical_cases = 0

    for supplier_id in assigned_supplier_ids:
        supplier = period.get_supplier(supplier_id)
        if supplier is None:
            continue

        for part in supplier.part:
            observations = part.get_observations_in_period(period_id)

            for obs in observations:
                if not obs.is_critical:
                    continue

                workload_by_date[obs.observation_date] += critical_case_duration_minutes
                n_critical_cases += 1

    x_labels = []
    workload_values = []

    for day_date in all_dates_in_period:
        day_diff = (day_date - period.start_date).days
        week_idx = (day_diff // 7) + 1
        weekday_label = day_date.strftime("%a")
        x_labels.append(f"W{week_idx}-{weekday_label}")
        workload_values.append(workload_by_date[day_date])

    x_positions = list(range(len(x_labels)))

    max_workload = max(workload_values) if len(workload_values) > 0 else 0.0
    upper_ref = max(max_workload, target_workload_per_day_minutes, 1.0)
    label_offset = max(1.0, 0.03 * upper_ref)
    top_padding = max(3.0, 0.12 * upper_ref)

    y_min = 0.0
    y_max = max_workload + label_offset + top_padding
    y_max = max(y_max, target_workload_per_day_minutes + top_padding)

    plt.figure(figsize=(max(12, len(x_labels) * 0.6), 5))
    plt.bar(x_positions, workload_values)
    plt.axhline(
        y=target_workload_per_day_minutes,
        linestyle="--",
        linewidth=2,
        label=f"Target workload = {target_workload_per_day_minutes:.1f} min/day",
    )

    for x, value in zip(x_positions, workload_values):
        plt.text(
            x,
            value + label_offset,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xticks(x_positions, x_labels, rotation=90)
    plt.ylim(y_min, y_max)
    plt.xlabel("Day in period")
    plt.ylabel("Workload in minutes")
    plt.title(
        f"Dispatcher {dispatcher_id}, Period {period_id}, "
        f"critical cases: {n_critical_cases}, "
        f"duration per critical case: {critical_case_duration_minutes:.1f} min"
    )
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_dispatcher_daily_workload_deviation_in_period(
    world: World,
    dispatcher_id: int,
    period_id: int,
    critical_case_duration_minutes: float,
    target_workload_per_day_minutes: float,
) -> None:
    """
    Plot the daily deviation from target workload for one dispatcher within one period.

    Parameters
    ----------
    world:
        Simulation world.
    dispatcher_id:
        ID of the dispatcher to analyze.
    period_id:
        Period whose daily workload deviation should be shown.
    critical_case_duration_minutes:
        Workload contribution in minutes of one critical case.
    target_workload_per_day_minutes:
        Target workload per workday in minutes.

    Notes
    -----
    - Only critical delivery call-offs generate workload.
    - Non-critical delivery call-offs generate zero workload.
    - Supplier-to-dispatcher assignment must already exist for the period.
    - Positive bar values mean overtime.
    - Negative bar values mean idle time relative to the target workload.
    """
    if critical_case_duration_minutes < 0:
        raise ValueError("critical_case_duration_minutes must be >= 0.")

    if target_workload_per_day_minutes < 0:
        raise ValueError("target_workload_per_day_minutes must be >= 0.")

    period = world.get_period(period_id)
    if period is None:
        raise ValueError(f"Period {period_id} not found.")

    dispatcher = period.get_dispatcher(dispatcher_id)
    if dispatcher is None:
        raise ValueError(f"Dispatcher {dispatcher_id} not found in period {period_id}.")

    if len(period.assignment) == 0:
        raise ValueError(
            f"No assignment found in period {period_id}. "
            "Run the assignment logic first."
        )

    assigned_supplier_ids = sorted(
        supplier_id
        for supplier_id, assigned_dispatcher_id in period.assignment.items()
        if assigned_dispatcher_id == dispatcher_id
    )

    if len(assigned_supplier_ids) == 0:
        raise ValueError(
            f"No suppliers assigned to dispatcher {dispatcher_id} in period {period_id}."
        )

    all_dates_in_period = []
    for week_idx in range(world.settings.weeks_per_period):
        week_start = period.start_date + timedelta(weeks=week_idx)
        for day in ["Mon", "Tue", "Wed", "Thu", "Fri"]:
            all_dates_in_period.append(
                week_start + timedelta(days=WEEKDAY_TO_OFFSET[day])
            )

    actual_workload_by_date = {day_date: 0.0 for day_date in all_dates_in_period}

    for supplier_id in assigned_supplier_ids:
        supplier = period.get_supplier(supplier_id)
        if supplier is None:
            continue

        for part in supplier.part:
            observations = part.get_observations_in_period(period_id)

            for obs in observations:
                if not obs.is_critical:
                    continue

                actual_workload_by_date[
                    obs.observation_date
                ] += critical_case_duration_minutes

    x_labels = []
    deviation_values = []

    for day_date in all_dates_in_period:
        day_diff = (day_date - period.start_date).days
        week_idx = (day_diff // 7) + 1
        weekday_label = day_date.strftime("%a")
        x_labels.append(f"W{week_idx}-{weekday_label}")

        actual_workload = actual_workload_by_date[day_date]
        deviation = actual_workload - target_workload_per_day_minutes
        deviation_values.append(deviation)

    x_positions = list(range(len(x_labels)))

    max_abs_deviation = (
        max(abs(v) for v in deviation_values) if len(deviation_values) > 0 else 0.0
    )
    ref = max(max_abs_deviation, 1.0)
    label_offset = max(1.0, 0.05 * ref)
    ylim_padding = max(3.0, 0.18 * ref)

    y_min = (
        min(deviation_values) - label_offset - ylim_padding
        if deviation_values
        else -1.0
    )
    y_max = (
        max(deviation_values) + label_offset + ylim_padding if deviation_values else 1.0
    )

    plt.figure(figsize=(max(12, len(x_labels) * 0.6), 5))
    plt.bar(x_positions, deviation_values, color="orange")
    plt.axhline(y=0.0, color="black", linewidth=1)

    for x, value in zip(x_positions, deviation_values):
        if value >= 0:
            plt.text(
                x,
                value + label_offset,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        else:
            plt.text(
                x,
                value - label_offset,
                f"{value:.1f}",
                ha="center",
                va="top",
                fontsize=9,
            )

    plt.xticks(x_positions, x_labels, rotation=90)
    plt.ylim(y_min, y_max)
    plt.xlabel("Day in period")
    plt.ylabel("Deviation from target workload in minutes")
    plt.title(
        f"Dispatcher {dispatcher_id}, Period {period_id}, "
        f"daily deviation from target workload "
        f"(target = {target_workload_per_day_minutes:.1f} min/day)"
    )
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_part_delivery_performance_scenario_calloffs(
    scenarios_data: list[dict],
    scenario_id: int,
    part_id: int,
    weeks_per_period: int,
) -> None:
    """
    Plot delivery call-offs of one part in one scenario.

    Parameters
    ----------
    scenarios_data:
        List returned by create_scenarios_for_current_period.
    scenario_id:
        Scenario index to inspect.
    part_id:
        Part to inspect.
    weeks_per_period:
        Number of weeks in the forecast period.
    """
    if weeks_per_period <= 0:
        raise ValueError("weeks_per_period must be > 0.")
    if scenario_id < 0 or scenario_id >= len(scenarios_data):
        raise ValueError(f"Scenario {scenario_id} not found.")

    scenario_data = scenarios_data[scenario_id]

    if part_id not in scenario_data["delivery_performance_by_part"]:
        raise ValueError(f"Part {part_id} not found in scenario {scenario_id}.")

    part_scenario = scenario_data["delivery_performance_by_part"][part_id]

    observations = sorted(
        part_scenario["delivery_observations"],
        key=lambda obs: obs["observation_date"],
    )

    x_positions = list(range(len(observations)))
    colors = ["red" if obs["is_critical"] else "green" for obs in observations]
    labels: list[str] = []

    forecast_period_start_date = part_scenario["forecast_period_start_date"]

    for obs in observations:
        day_diff = (obs["observation_date"] - forecast_period_start_date).days
        week_idx = (day_diff // 7) + 1
        week_idx = min(max(week_idx, 1), weeks_per_period)
        labels.append(f"W{week_idx}-{obs['delivery_day']}")

    plt.figure(figsize=(max(12, max(1, len(observations)) * 0.35), 4))

    if len(observations) > 0:
        plt.scatter(x_positions, [1] * len(observations), c=colors, s=70)
        plt.xticks(x_positions, labels, rotation=90)
    else:
        plt.xticks([])

    plt.yticks([])
    plt.xlabel("Week and Weekday")
    plt.title(
        f"Scenario {scenario_id}, Part {part_id}, "
        f"Forecast period {part_scenario['forecast_period_id']}, "
        f"sampled theta={part_scenario['sampled_theta']:.4f}, "
        f"Critical delivery call-offs: {part_scenario['n_critical_delivery_calloffs']}, "
        f"Non-critical delivery call-offs: {part_scenario['n_non_critical_delivery_calloffs']}"
    )
    plt.tight_layout()
    plt.show()


def plot_part_workload_scenario_daily_workload(
    scenarios_data: list[dict],
    scenario_id: int,
    part_id: int,
    target_workload_per_day: float | None = None,
) -> None:
    """
    Plot daily workload of one part in one scenario.

    Parameters
    ----------
    scenarios_data:
        List returned by create_scenarios_for_current_period.
    scenario_id:
        Scenario index to inspect.
    part_id:
        Part to inspect.
    target_workload_per_day:
        Optional dashed target line in minutes per day.
    """
    if scenario_id < 0 or scenario_id >= len(scenarios_data):
        raise ValueError(f"Scenario {scenario_id} not found.")
    if target_workload_per_day is not None and target_workload_per_day < 0.0:
        raise ValueError("target_workload_per_day must be >= 0.")

    scenario_data = scenarios_data[scenario_id]

    if part_id not in scenario_data["workload_by_part"]:
        raise ValueError(f"Part {part_id} not found in scenario {scenario_id}.")

    part_scenario = scenario_data["workload_by_part"][part_id]

    forecast_period_start_date = part_scenario["forecast_period_start_date"]
    forecast_period_dates = list(part_scenario["forecast_period_dates"])

    workload_by_date = {day_date: 0.0 for day_date in forecast_period_dates}
    for day_date, workload in part_scenario["daily_workload_minutes"].items():
        workload_by_date[day_date] = float(workload)

    x_labels = []
    workload_values = []

    for day_date in forecast_period_dates:
        day_diff = (day_date - forecast_period_start_date).days
        week_idx = (day_diff // 7) + 1
        weekday_label = day_date.strftime("%a")
        x_labels.append(f"W{week_idx}-{weekday_label}")
        workload_values.append(workload_by_date[day_date])

    x_positions = list(range(len(x_labels)))

    plt.figure(figsize=(max(12, len(x_labels) * 0.6), 5))
    plt.bar(x_positions, workload_values)

    if target_workload_per_day is not None:
        plt.axhline(
            y=target_workload_per_day,
            linestyle="--",
            linewidth=2,
            label=f"Target workload = {target_workload_per_day:.2f} min/day",
        )
        plt.legend()

    for x, value in zip(x_positions, workload_values):
        plt.text(
            x,
            value
            + max(
                1.0, 0.02 * max(1.0, max(workload_values) if workload_values else 1.0)
            ),
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xticks(x_positions, x_labels, rotation=90)
    plt.xlabel("Day in forecast period")
    plt.ylabel("Workload in minutes")
    plt.title(
        f"Scenario {scenario_id}, Part {part_id}, "
        f"Forecast period {part_scenario['forecast_period_id']}, "
        f"sampled theta={part_scenario['sampled_theta']:.4f}, "
        f"total workload={part_scenario['total_workload_minutes']:.2f} min"
    )
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_supplier_workload_scenario_daily_workload(
    scenarios_data: list[dict],
    scenario_id: int,
    supplier_id: int,
    target_workload_per_day: float | None = None,
) -> None:
    """
    Plot daily workload of one supplier in one scenario.

    Parameters
    ----------
    scenarios_data:
        List returned by create_scenarios_for_current_period.
    scenario_id:
        Scenario index to inspect.
    supplier_id:
        Supplier to inspect.
    target_workload_per_day:
        Optional dashed target line in minutes per day.
    """
    if scenario_id < 0 or scenario_id >= len(scenarios_data):
        raise ValueError(f"Scenario {scenario_id} not found.")
    if target_workload_per_day is not None and target_workload_per_day < 0.0:
        raise ValueError("target_workload_per_day must be >= 0.")

    scenario_data = scenarios_data[scenario_id]

    if supplier_id not in scenario_data["workload_by_supplier"]:
        raise ValueError(f"Supplier {supplier_id} not found in scenario {scenario_id}.")

    supplier_scenario = scenario_data["workload_by_supplier"][supplier_id]

    forecast_period_start_date = supplier_scenario["forecast_period_start_date"]
    forecast_period_dates = list(supplier_scenario["forecast_period_dates"])

    workload_by_date = {day_date: 0.0 for day_date in forecast_period_dates}
    for day_date, workload in supplier_scenario["daily_workload_minutes"].items():
        workload_by_date[day_date] = float(workload)

    x_labels = []
    workload_values = []

    for day_date in forecast_period_dates:
        day_diff = (day_date - forecast_period_start_date).days
        week_idx = (day_diff // 7) + 1
        weekday_label = day_date.strftime("%a")
        x_labels.append(f"W{week_idx}-{weekday_label}")
        workload_values.append(workload_by_date[day_date])

    x_positions = list(range(len(x_labels)))

    plt.figure(figsize=(max(12, len(x_labels) * 0.6), 5))
    plt.bar(x_positions, workload_values)

    if target_workload_per_day is not None:
        plt.axhline(
            y=target_workload_per_day,
            linestyle="--",
            linewidth=2,
            label=f"Target workload = {target_workload_per_day:.2f} min/day",
        )
        plt.legend()

    for x, value in zip(x_positions, workload_values):
        plt.text(
            x,
            value
            + max(
                1.0, 0.02 * max(1.0, max(workload_values) if workload_values else 1.0)
            ),
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xticks(x_positions, x_labels, rotation=90)
    plt.xlabel("Day in forecast period")
    plt.ylabel("Workload in minutes")
    plt.title(
        f"Scenario {scenario_id}, Supplier {supplier_id}, "
        f"Forecast period {supplier_scenario['forecast_period_id']}, "
        f"total workload={supplier_scenario['total_workload_minutes']:.2f} min"
    )
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
