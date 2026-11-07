import random
from typing import Optional
import json
from collections import Counter

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
_WEEKDAY_ORDER = {day: idx for idx, day in enumerate(WEEKDAYS)}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _distribute_integer_total(
    rng: random.Random,
    total: int,
    n_bins: int,
    lo: int,
    hi: int,
) -> list[int]:
    if not (n_bins * lo <= total <= n_bins * hi):
        raise ValueError(
            f"Cannot distribute {total} into {n_bins} bins with each in "
            f"[{lo}, {hi}]. Feasible range: [{n_bins * lo}, {n_bins * hi}]."
        )
    values = [lo] * n_bins
    slack = total - n_bins * lo

    order = list(range(n_bins))
    rng.shuffle(order)
    for i in order:
        if slack == 0:
            break
        add = rng.randint(0, min(hi - lo, slack))
        values[i] += add
        slack -= add

    # deterministic second pass to absorb leftover slack
    for i in order:
        if slack == 0:
            break
        room = hi - values[i]
        add = min(room, slack)
        values[i] += add
        slack -= add

    assert slack == 0, "BUG: undistributed slack"
    return values


def _day_counts_from_patterns(patterns: list[list[str]]) -> dict[str, int]:
    """Count total deliveries per weekday across all patterns."""
    counts: dict[str, int] = {d: 0 for d in WEEKDAYS}
    for pattern in patterns:
        for day in pattern:
            counts[day] += 1
    return counts


def _imbalance_ratio(day_counts: dict[str, int]) -> float:
    """
    (max - min) / mean across the five weekdays.
    Returns 0.0 if mean is zero (degenerate case).
    """
    values = list(day_counts.values())
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    return (max(values) - min(values)) / mean


def _sample_patterns_for_day_counts(
    rng: random.Random,
    day_counts: list[int],
) -> list[list[str]]:
    """
    Given a list of per-part day-counts, sample a sorted weekday pattern
    for each part independently.
    """
    return [
        sorted(rng.sample(WEEKDAYS, k=k), key=lambda d: _WEEKDAY_ORDER[d])
        for k in day_counts
    ]


def _resample_one_violating_part(
    rng: random.Random,
    patterns: list[list[str]],
    day_counts: list[int],
    day_totals: dict[str, int],
    max_imbalance: float,
) -> bool:
    """
    Find the part whose current pattern contributes most to imbalance and
    resample *only that part's pattern* (keeping its day-count fixed).

    Returns True if the global imbalance improved after the swap.
    """
    values = list(day_totals.values())
    mean = sum(values) / len(values)
    heavy_day = max(day_totals, key=day_totals.__getitem__)
    light_day = min(day_totals, key=day_totals.__getitem__)

    # Find a part that contains heavy_day but NOT light_day, and whose
    # day-count allows including light_day (it always does since count >= 1).
    candidates = [
        i for i, p in enumerate(patterns)
        if heavy_day in p and light_day not in p
    ]
    if not candidates:
        return False

    idx = rng.choice(candidates)
    old_pattern = patterns[idx]
    k = day_counts[idx]

    # Try up to 20 random resamples; keep the one that reduces max imbalance.
    best_pattern = old_pattern
    best_ratio = _imbalance_ratio(day_totals)

    for _ in range(20):
        new_pattern = sorted(
            rng.sample(WEEKDAYS, k=k), key=lambda d: _WEEKDAY_ORDER[d]
        )
        # Compute updated totals hypothetically
        trial = dict(day_totals)
        for d in old_pattern:
            trial[d] -= 1
        for d in new_pattern:
            trial[d] += 1
        ratio = _imbalance_ratio(trial)
        if ratio < best_ratio:
            best_ratio = ratio
            best_pattern = new_pattern

    if best_pattern is old_pattern:
        return False  # no improvement found

    # Apply the best swap
    for d in old_pattern:
        day_totals[d] -= 1
    for d in best_pattern:
        day_totals[d] += 1
    patterns[idx] = best_pattern
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def sample_world_config_constrained(
    settings,
    target_n_parts: int,
    target_total_deliveries: int,
    min_part_count_per_supplier: int = 3,
    max_part_count_per_supplier: int = 10,
    min_delivery_days_per_part: int = 1,
    max_delivery_days_per_part: int = 5,
    max_daily_imbalance_pct: float = 0.20,
    max_balance_iterations: int = 500,
    random_seed: Optional[int] = None,
    part_counts: Optional[list] = None
) -> dict:
    """
    Sample a world_config with exact cardinality constraints **and** a
    daily-delivery balance guarantee.

    Constraints
    -----------
    * ``len(suppliers) == settings.n_init_suppliers``
    * ``len(parts)     == target_n_parts``
    * ``sum(len(p["delivery_day_pattern"]) for p in parts.values())
         == target_total_deliveries``
    * ``(max_day_total - min_day_total) / mean_day_total
         <= max_daily_imbalance_pct``

    Parameters
    ----------
    max_daily_imbalance_pct:
        Allowed (max − min) / mean ratio across Mon–Fri delivery totals.
        E.g. 0.20 means no day may deviate more than ±~10 % from the mean
        in the worst-case spread sense.
    max_balance_iterations:
        Hard cap on the local-search loop that rebalances patterns.
        If balance is still not achieved, a ValueError is raised so the
        caller can relax the threshold or increase the cap.
    """
    rng = random.Random(random_seed)

    # ── Dispatchers ──────────────────────────────────────────────────────────
    dispatchers: dict[int, dict] = {
        did: {"employment_type": "fulltime" if rng.random() < 0.7 else "parttime"}
        for did in range(1, settings.n_dispatchers + 1)
    }

    # ── Part counts per supplier (fixed, not resampled during balancing) ─────
    n_suppliers = settings.n_init_suppliers
    if part_counts is None:
        part_counts = _distribute_integer_total(
            rng, total=target_n_parts, n_bins=n_suppliers, lo=min_part_count_per_supplier, hi=max_part_count_per_supplier
        )

    # ── Day counts per part (fixed; only *which* days change during balance) ──
    day_counts = _distribute_integer_total(
        rng, total=target_total_deliveries, n_bins=target_n_parts, lo=1, hi=5
    )
    rng.shuffle(day_counts)

    # ── Initial pattern sample ────────────────────────────────────────────────
    patterns = _sample_patterns_for_day_counts(rng, day_counts)
    day_totals = _day_counts_from_patterns(patterns)

    # ── Balance loop ──────────────────────────────────────────────────────────
    for iteration in range(max_balance_iterations):
        ratio = _imbalance_ratio(day_totals)
        if ratio <= max_daily_imbalance_pct:
            break
        _resample_one_violating_part(
            rng, patterns, day_counts, day_totals, max_daily_imbalance_pct
        )
    else:
        final_ratio = _imbalance_ratio(day_totals)
        if final_ratio > max_daily_imbalance_pct:
            raise ValueError(
                f"Could not achieve daily imbalance <= {max_daily_imbalance_pct:.1%} "
                f"after {max_balance_iterations} iterations "
                f"(final ratio: {final_ratio:.3f}). "
                "Try relaxing max_daily_imbalance_pct or increasing max_balance_iterations."
            )

    # ── Assemble output ───────────────────────────────────────────────────────
    parts: dict[int, dict] = {}
    part_id = 1
    pat_idx = 0
    for sid in range(1, n_suppliers + 1):
        for _ in range(part_counts[sid - 1]):
            parts[part_id] = {"delivery_day_pattern": patterns[pat_idx]}
            part_id += 1
            pat_idx += 1

    # ── Sanity checks ─────────────────────────────────────────────────────────
    assert len(parts) == target_n_parts
    assert sum(len(p["delivery_day_pattern"]) for p in parts.values()) == target_total_deliveries

    return {
        "suppliers":   {sid: {"n_parts": part_counts[sid - 1]} for sid in range(1, n_suppliers + 1)},
        "parts":       parts,
        "dispatchers": dispatchers,
        "_debug_day_totals": day_totals,   # remove if unwanted
    }

def check_delivery_day_balance(parts: dict) -> None:
    day_count = Counter()
    for part_data in parts.values():
        for day in part_data['delivery_day_pattern']:
            day_count[day] += 1

    total = sum(day_count.values())
    n_parts = len(parts)
    expected = n_parts * 3 / 5 

    print(f"{'Day':<6} {'Count':>6} {'% of parts':>10} {'Deviation':>10}")
    print("-" * 36)
    for day in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']:
        count = day_count[day]
        pct = count / n_parts * 100
        dev = count - expected
        print(f"{day:<6} {count:>6} {pct:>9.1f}% {dev:>+10.1f}")

    counts = [day_count[d] for d in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']]
    print(f"\nExpected per day : {expected:.1f}")
    print(f"Max - Min        : {max(counts) - min(counts)}")
    print(f"Max deviation    : {max(abs(c - expected) for c in counts):.1f}")