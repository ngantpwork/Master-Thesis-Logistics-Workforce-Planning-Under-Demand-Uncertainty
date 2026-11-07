from __future__ import annotations

from .simulation_entities import Period, World


def apply_greedy_least_loaded_parts_assignment(period: Period) -> None:
    """
    Greedy least-loaded assignment based on number of parts.

    Heuristic:
    1. Sort suppliers by descending number of parts, tie-break by smaller supplier_id.
    2. Assign each supplier to the dispatcher with the currently smallest assigned part load,
       tie-break by smaller dispatcher_id.

    Result:
        period.assignment[supplier_id] = dispatcher_id
    """
    period.clear_assignments()

    if len(period.dispatcher) == 0:
        return

    dispatcher_part_load = {
        dispatcher.dispatcher_id: 0
        for dispatcher in period.dispatcher
    }

    sorted_suppliers = sorted(
        period.supplier,
        key=lambda supplier: (-supplier.n_parts, supplier.supplier_id),
    )

    for supplier in sorted_suppliers:
        chosen_dispatcher = min(
            period.dispatcher,
            key=lambda dispatcher: (
                dispatcher_part_load[dispatcher.dispatcher_id],
                dispatcher.dispatcher_id,
            ),
        )

        period.set_assignment(
            supplier_id=supplier.supplier_id,
            dispatcher_id=chosen_dispatcher.dispatcher_id,
        )

        dispatcher_part_load[chosen_dispatcher.dispatcher_id] += supplier.n_parts


def assign_current_period_with_oem_logic(world: World) -> None:
    """
    Apply the OEM assignment logic to the current period.

    The OEM logic uses the greedy least-loaded heuristic based on the number of parts.
    """
    period = world.get_current_period()
    if period is None:
        raise ValueError("Current period not found in world.")

    if len(period.dispatcher) == 0:
        period.clear_assignments()
        return

    apply_greedy_least_loaded_parts_assignment(period)