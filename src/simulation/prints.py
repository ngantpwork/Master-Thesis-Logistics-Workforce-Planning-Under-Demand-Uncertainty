from .simulation_entities import Part


def print_part_period_history(
    part: Part,
    up_to_period_id: int,
    show_only_realized_periods: bool = True,
    compact: bool = False,
) -> None:
    if part is None:
        raise ValueError("part must not be None.")
    if up_to_period_id < 0:
        raise ValueError("up_to_period_id must be >= 0.")

    print(f"Part {part.part_id} history up to period {up_to_period_id}")
    print("-" * 100)

    total_critical = 0
    total_non_critical = 0
    total_actual = 0
    total_possible = 0

    for period_id in range(up_to_period_id + 1):
        state = part.get_period_state(period_id)

        if state is None:
            if not show_only_realized_periods:
                print(f"Period {period_id}: no data")
            continue

        total_critical += state.n_critical_delivery_calloffs
        total_non_critical += state.n_non_critical_delivery_calloffs
        total_actual += state.n_actual_delivery_calloffs
        total_possible += state.n_possible_delivery_calloffs

        if compact:
            print(
                f"Period {period_id} | "
                f"theta={state.true_part_theta_in_period:.3f}, "
                f"actual={state.n_actual_delivery_calloffs}, "
                f"crit={state.n_critical_delivery_calloffs}, "
                f"post_mean={state.posterior_mean:.4f}"
            )
        else:
            print(f"Period {period_id}")
            print(f"  True theta in period: {state.true_part_theta_in_period:.4f}")
            print(
                f"  Possible delivery call-offs: {state.n_possible_delivery_calloffs}"
            )
            print(f"  Actual delivery call-offs: {state.n_actual_delivery_calloffs}")
            print(
                f"  Critical delivery call-offs: {state.n_critical_delivery_calloffs}"
            )
            print(
                f"  Non-critical delivery call-offs: {state.n_non_critical_delivery_calloffs}"
            )
            print(f"  Observed theta in period: {state.observed_theta_in_period}")

            print()

            print(
                f"  Posterior: Beta(alpha={state.posterior_alpha:.4f}, "
                f"beta={state.posterior_beta:.4f})"
            )
            print(f"  Posterior mean: {state.posterior_mean:.6f}")
            print(f"  Posterior variance: {state.posterior_variance:.6f}")
            print()

    print("-" * 100)
    print(f"Cumulative possible call-offs: {total_possible}")
    print(f"Cumulative actual call-offs: {total_actual}")
    print(f"Cumulative critical call-offs: {total_critical}")
    print(f"Cumulative non-critical call-offs: {total_non_critical}")


def print_realized_assignment_evaluation(
    evaluation_result: dict,
) -> None:
    """
    Print a compact summary of the realized assignment evaluation.
    """
    print(f"Evaluation for period {evaluation_result['period_id']}")
    print("-" * 80)

    print(f"Total workload minutes: {evaluation_result['total_workload_minutes']:.2f}")
    print(f"Total overtime minutes: {evaluation_result['total_overtime_minutes']:.2f}")
    print(
        f"Total undertime minutes: {evaluation_result['total_undertime_minutes']:.2f}"
    )
    print(
        "Max daily absolute deviation minutes: "
        f"{evaluation_result['max_daily_abs_deviation_minutes']:.2f}"
    )
    print(f"Spike cost: {evaluation_result['spike_cost']:.2f}")

    if "n_reassignments" in evaluation_result:
        print(f"Number of reassignments: {evaluation_result['n_reassignments']}")
        print(f"Reassigned suppliers: {evaluation_result['reassigned_suppliers']}")

    if "assignment" in evaluation_result:
        print("\nAssignment (supplier -> dispatcher):")
        for supplier_id, dispatcher_id in sorted(
            evaluation_result["assignment"].items()
        ):
            print(f"  Supplier {supplier_id} -> Dispatcher {dispatcher_id}")

    if "dispatcher_day_deviation_minutes" in evaluation_result:
        print("\nDaily workload deviation (minutes):")
        for (dispatcher_id, day_date), deviation in sorted(
            evaluation_result["dispatcher_day_deviation_minutes"].items()
        ):
            print(f"  Dispatcher {dispatcher_id}, {day_date}: {deviation:+.1f} min")
