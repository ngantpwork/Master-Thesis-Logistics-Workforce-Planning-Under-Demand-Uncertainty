"""
hierarchical_plots.py
=====================
Visualization module for the hierarchical Beta-Bernoulli supplier/part model.

All public functions accept a ``World`` object (already stepped forward ≥ 1
period) and return a ``matplotlib.figure.Figure``.  They do NOT call
``plt.show()`` — the caller is responsible for displaying or saving.

Public API
----------
plot_beta_distributions(world)          Tab 1: PDF of θ_s and θ_ps
plot_parameter_sensitivity(world)       Tab 2: Variance vs κ_s / κ_p curves + heatmap
plot_supplier_part_scatter(world)       Tab 3: Strip-plot θ_s + scatter θ_ps grouped by supplier
plot_calloff_timeseries(world)          Tab 4: Critical-rate series, bar totals, empirical vs theory

Convenience
-----------
plot_all(world, save_dir=None)          Render all four figures and optionally save to PNG

Usage example
-------------
    from datetime import date
    from simulation_entities import WorldSettings
    from simulation_functions import initialize_world, step_world_forward
    from hierarchical_plots import plot_all

    settings = WorldSettings(
        n_simulation_periods=12,
        current_period=-1,
        simulation_start_date=date(2024, 1, 1),
        weeks_per_period=4,
        n_init_suppliers=8,
        n_dispatchers=3,
        use_hierarchical_theta=True,
        m_global=0.10,
        kappa_s=10.0,
        kappa_p=5.0,
    )
    world = initialize_world(settings, random_seed=42)
    for _ in range(settings.n_simulation_periods):
        step_world_forward(world)

    figs = plot_all(world, save_dir="./figures")
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
from scipy import stats

from .simulation_entities import World, WorldSettings


# ---------------------------------------------------------------------------
# Internal styling helpers
# ---------------------------------------------------------------------------

_PALETTE = [
    "#7F77DD", "#1D9E75", "#D85A30", "#D4537E",
    "#378ADD", "#639922", "#BA7517", "#E24B4A", "#888780",
]

_STYLE_DEFAULTS = dict(
    figure_facecolor="white",
    axes_facecolor="#FAFAFA",
    grid_alpha=0.25,
    spine_color="#CCCCCC",
    title_fontsize=11,
    label_fontsize=10,
    tick_fontsize=9,
)


def _apply_style(ax: plt.Axes) -> None:
    """Apply uniform styling to an Axes object."""
    s = _STYLE_DEFAULTS
    ax.set_facecolor(s["axes_facecolor"])
    ax.grid(True, alpha=s["grid_alpha"], linewidth=0.5)
    ax.tick_params(labelsize=s["tick_fontsize"])
    for spine in ax.spines.values():
        spine.set_color(s["spine_color"])
        spine.set_linewidth(0.5)


def _fig(nrows: int, ncols: int, **kw) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(nrows, ncols, facecolor="white", **kw)
    return fig, np.atleast_2d(axes)


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def _get_settings(world: World) -> WorldSettings:
    return world.settings


def _get_all_periods_sorted(world: World) -> list:
    """Return Period objects sorted by period_id."""
    return sorted(world.period, key=lambda p: p.period_id)


def _iter_suppliers_in_period0(world: World):
    """Yield Supplier objects from period 0 (structural view)."""
    p0 = world.get_period(0)
    if p0 is None:
        return
    yield from p0.supplier


def _supplier_theta(supplier) -> Optional[float]:
    """Return supplier_theta if set (hierarchical path), else None."""
    return getattr(supplier, "supplier_theta", None)


def _beta_pdf_curve(a: float, b: float, n: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, pdf) arrays for Beta(a, b) over (0, 1)."""
    x = np.linspace(0.001, 0.999, n)
    pdf = stats.beta.pdf(x, a, b)
    return x, pdf


# ---------------------------------------------------------------------------
# Figure 1 — Beta distribution PDFs
# ---------------------------------------------------------------------------

def plot_beta_distributions(world: World) -> plt.Figure:
    """
    Figure 1 — Beta distribution PDFs for θ_s and θ_ps.

    Layout (2 × 2):
    ┌──────────────────────┬──────────────────────┐
    │ PDF θ_s (theory)     │ PDF θ_ps (theory)     │
    │  Beta(m·κs, (1-m)·κs)│  Beta(θ̄s·κp,(1-θ̄s)·κp)│
    ├──────────────────────┼──────────────────────┤
    │ Histogram θ_s        │ Histogram θ_ps        │
    │  (empirical from data)│ (empirical from data) │
    └──────────────────────┴──────────────────────┘

    Parameters
    ----------
    world : World
        Simulation world after at least one call to step_world_forward().

    Returns
    -------
    matplotlib.figure.Figure
    """
    s = _get_settings(world)

    fig, axes = _fig(2, 2, figsize=(10, 7))
    fig.suptitle(
        f"Beta distributions  —  m={s.m_global:.2f}  κs={s.kappa_s}  κp={s.kappa_p}",
        fontsize=12, fontweight="normal", y=0.98,
    )

    # ── Collect empirical thetas from period 0 suppliers ───────────────────
    supplier_thetas: list[float] = []
    part_thetas: list[float] = []

    for sup in _iter_suppliers_in_period0(world):
        st = _supplier_theta(sup)
        if st is not None:
            supplier_thetas.append(st)
        for part in sup.part:
            part_thetas.append(part.base_true_part_theta)

    # ── Top-left: PDF θ_s ──────────────────────────────────────────────────
    ax = axes[0, 0]
    _apply_style(ax)
    if s.use_hierarchical_theta:
        a_s, b_s = s.m_global * s.kappa_s, (1 - s.m_global) * s.kappa_s
        x, pdf = _beta_pdf_curve(a_s, b_s)
        ax.plot(x, pdf, color=_PALETTE[0], lw=2, label=f"Beta({a_s:.1f}, {b_s:.1f})")
        ax.axvline(s.m_global, color="#D85A30", lw=1.2, ls="--", label=f"m = {s.m_global:.2f}")
        var_s = s.m_global * (1 - s.m_global) / (s.kappa_s + 1)
        ax.set_title(
            f"θ_s ~ Beta(m·κs, (1−m)·κs)\n"
            f"E[θ_s]={s.m_global:.3f}   Var={var_s:.5f}",
            fontsize=_STYLE_DEFAULTS["title_fontsize"],
        )
        ax.legend(fontsize=8)
    else:
        ax.set_title("θ_s — legacy path (no hierarchical Beta)", fontsize=10)
        ax.text(0.5, 0.5, "use_hierarchical_theta=False", ha="center", va="center",
                transform=ax.transAxes, color="gray", fontsize=9)
    ax.set_xlabel("θ_s", fontsize=_STYLE_DEFAULTS["label_fontsize"])
    ax.set_ylabel("density", fontsize=_STYLE_DEFAULTS["label_fontsize"])
    ax.set_xlim(0, 1)

    # ── Top-right: PDF θ_ps ────────────────────────────────────────────────
    ax = axes[0, 1]
    _apply_style(ax)
    if s.use_hierarchical_theta and supplier_thetas:
        mean_theta_s = float(np.mean(supplier_thetas))
        a_p = mean_theta_s * s.kappa_p
        b_p = (1 - mean_theta_s) * s.kappa_p
        x, pdf = _beta_pdf_curve(a_p, b_p)
        ax.plot(x, pdf, color=_PALETTE[1], lw=2,
                label=f"Beta({a_p:.1f}, {b_p:.1f})\n(θ_s = {mean_theta_s:.3f})")
        ax.axvline(mean_theta_s, color="#D85A30", lw=1.2, ls="--",
                   label=f"E[θ_s] = {mean_theta_s:.3f}")
        var_p = mean_theta_s * (1 - mean_theta_s) / (s.kappa_p + 1)
        ax.set_title(
            f"θ_ps | θ_s ~ Beta(θ_s·κp, (1−θ_s)·κp)\n"
            f"E[θ_ps|θ_s]={mean_theta_s:.3f}   Var={var_p:.5f}",
            fontsize=_STYLE_DEFAULTS["title_fontsize"],
        )
        ax.legend(fontsize=8)
    else:
        ax.set_title("θ_ps — legacy path", fontsize=10)
        ax.text(0.5, 0.5, "use_hierarchical_theta=False", ha="center", va="center",
                transform=ax.transAxes, color="gray", fontsize=9)
    ax.set_xlabel("θ_ps", fontsize=_STYLE_DEFAULTS["label_fontsize"])
    ax.set_ylabel("density", fontsize=_STYLE_DEFAULTS["label_fontsize"])
    ax.set_xlim(0, 1)

    # ── Bottom-left: Histogram θ_s empirical + overlaid theory ────────────
    ax = axes[1, 0]
    _apply_style(ax)
    if supplier_thetas:
        ax.hist(supplier_thetas, bins=min(20, max(5, len(supplier_thetas) // 2)),
                density=True, color=_PALETTE[0], alpha=0.55, edgecolor="white",
                linewidth=0.5, label="empirical")
        if s.use_hierarchical_theta:
            a_s, b_s = s.m_global * s.kappa_s, (1 - s.m_global) * s.kappa_s
            x, pdf = _beta_pdf_curve(a_s, b_s)
            ax.plot(x, pdf, color=_PALETTE[0], lw=2, ls="-", label="theoretical")
        ax.axvline(float(np.mean(supplier_thetas)), color="#D85A30", lw=1.2, ls="--",
                   label=f"mean = {np.mean(supplier_thetas):.3f}")
    ax.set_title("Empirical distribution — θ_s", fontsize=_STYLE_DEFAULTS["title_fontsize"])
    ax.set_xlabel("θ_s", fontsize=_STYLE_DEFAULTS["label_fontsize"])
    ax.set_ylabel("density", fontsize=_STYLE_DEFAULTS["label_fontsize"])
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8)

    # ── Bottom-right: Histogram θ_ps empirical + overlaid theory ──────────
    ax = axes[1, 1]
    _apply_style(ax)
    if part_thetas:
        ax.hist(part_thetas, bins=min(25, max(5, len(part_thetas) // 3)),
                density=True, color=_PALETTE[1], alpha=0.55, edgecolor="white",
                linewidth=0.5, label="empirical")
        if s.use_hierarchical_theta and supplier_thetas:
            mean_ts = float(np.mean(supplier_thetas))
            a_p = mean_ts * s.kappa_p
            b_p = (1 - mean_ts) * s.kappa_p
            x, pdf = _beta_pdf_curve(a_p, b_p)
            ax.plot(x, pdf, color=_PALETTE[1], lw=2, ls="-", label="theoretical")
        ax.axvline(float(np.mean(part_thetas)), color="#D85A30", lw=1.2, ls="--",
                   label=f"mean = {np.mean(part_thetas):.3f}")
    ax.set_title("Empirical distribution — θ_ps", fontsize=_STYLE_DEFAULTS["title_fontsize"])
    ax.set_xlabel("θ_ps", fontsize=_STYLE_DEFAULTS["label_fontsize"])
    ax.set_ylabel("density", fontsize=_STYLE_DEFAULTS["label_fontsize"])
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Parameter sensitivity
# ---------------------------------------------------------------------------

def plot_parameter_sensitivity(world: World) -> plt.Figure:
    """
    Figure 2 — How κ_s and κ_p control variance + 9-regime heatmap.

    Layout (1 × 3):
    ┌──────────────────┬──────────────────┬────────────────────────┐
    │ Var(θ_s) vs κ_s  │ Var(θ_ps) vs κ_p │ Heatmap 9 regimes      │
    │                  │                  │ κ_s × κ_p              │
    └──────────────────┴──────────────────┴────────────────────────┘

    Parameters
    ----------
    world : World

    Returns
    -------
    matplotlib.figure.Figure
    """
    s = _get_settings(world)
    m = s.m_global

    # Reference θ_s for the part-level curve: use empirical mean or m_global
    supplier_thetas = [
        _supplier_theta(sup)
        for sup in _iter_suppliers_in_period0(world)
        if _supplier_theta(sup) is not None
    ]
    ref_theta_s = float(np.mean(supplier_thetas)) if supplier_thetas else m

    fig, axes = _fig(1, 3, figsize=(13, 4.5))
    fig.suptitle(
        f"Parameter sensitivity  —  m={m:.2f}  "
        f"current: κs={s.kappa_s}  κp={s.kappa_p}",
        fontsize=12, fontweight="normal", y=1.01,
    )

    ks_range = np.arange(1, 121, 1, dtype=float)
    kp_range = np.arange(1, 121, 1, dtype=float)
    var_s_curve = m * (1 - m) / (ks_range + 1)
    var_p_curve = ref_theta_s * (1 - ref_theta_s) / (kp_range + 1)

    # ── Left: Var(θ_s) vs κ_s ─────────────────────────────────────────────
    ax = axes[0, 0]
    _apply_style(ax)
    ax.plot(ks_range, var_s_curve, color=_PALETTE[0], lw=2)
    ax.axvline(s.kappa_s, color="#D85A30", lw=1.2, ls="--", label=f"current κs = {s.kappa_s:.0f}")
    current_var_s = m * (1 - m) / (s.kappa_s + 1)
    ax.scatter([s.kappa_s], [current_var_s], color="#D85A30", s=60, zorder=5)
    ax.set_xlabel("κ_s", fontsize=_STYLE_DEFAULTS["label_fontsize"])
    ax.set_ylabel("Var(θ_s)", fontsize=_STYLE_DEFAULTS["label_fontsize"])
    ax.set_title(
        f"Supplier heterogeneity\nVar(θ_s) = m(1−m) / (κs+1)",
        fontsize=_STYLE_DEFAULTS["title_fontsize"],
    )
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    # Annotate three canonical levels
    for ks_mark, label in [(10, "high"), (30, "med"), (100, "low")]:
        v = m * (1 - m) / (ks_mark + 1)
        ax.annotate(
            label, xy=(ks_mark, v), xytext=(ks_mark + 4, v + 0.0008),
            fontsize=7, color="#666666",
            arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.7),
        )

    # ── Middle: Var(θ_ps) vs κ_p ──────────────────────────────────────────
    ax = axes[0, 1]
    _apply_style(ax)
    ax.plot(kp_range, var_p_curve, color=_PALETTE[1], lw=2)
    ax.axvline(s.kappa_p, color="#D85A30", lw=1.2, ls="--", label=f"current κp = {s.kappa_p:.0f}")
    current_var_p = ref_theta_s * (1 - ref_theta_s) / (s.kappa_p + 1)
    ax.scatter([s.kappa_p], [current_var_p], color="#D85A30", s=60, zorder=5)
    ax.set_xlabel("κ_p", fontsize=_STYLE_DEFAULTS["label_fontsize"])
    ax.set_ylabel("Var(θ_ps | θ_s)", fontsize=_STYLE_DEFAULTS["label_fontsize"])
    ax.set_title(
        f"Within-supplier part heterogeneity\nVar(θ_ps|θ_s) = θ_s(1−θ_s) / (κp+1)\n"
        f"(at θ_s = {ref_theta_s:.3f})",
        fontsize=_STYLE_DEFAULTS["title_fontsize"],
    )
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    for kp_mark, label in [(5, "high"), (20, "med"), (100, "low")]:
        v = ref_theta_s * (1 - ref_theta_s) / (kp_mark + 1)
        ax.annotate(
            label, xy=(kp_mark, v), xytext=(kp_mark + 4, v + 0.0003),
            fontsize=7, color="#666666",
            arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.7),
        )

    # ── Right: Heatmap 9 regimes ──────────────────────────────────────────
    ax = axes[0, 2]
    ks_grid = [10, 30, 100]
    kp_grid = [5, 20, 100]
    heat = np.zeros((len(ks_grid), len(kp_grid)))
    for i, ks in enumerate(ks_grid):
        for j, kp in enumerate(kp_grid):
            heat[i, j] = (
                m * (1 - m) / (ks + 1)
                + ref_theta_s * (1 - ref_theta_s) / (kp + 1)
            )

    im = ax.imshow(heat, cmap="Purples", aspect="auto", origin="upper")
    ax.set_xticks(range(len(kp_grid)))
    ax.set_xticklabels([f"κp={v}" for v in kp_grid], fontsize=8)
    ax.set_yticks(range(len(ks_grid)))
    ax.set_yticklabels([f"κs={v}" for v in ks_grid], fontsize=8)
    ax.set_title(
        "Total variance  Var(θ_s) + Var(θ_ps|θ_s)\n9-regime grid  (darker = more heterogeneous)",
        fontsize=_STYLE_DEFAULTS["title_fontsize"],
    )

    # Annotate cells
    for i in range(len(ks_grid)):
        for j in range(len(kp_grid)):
            ax.text(
                j, i, f"{heat[i, j]:.5f}",
                ha="center", va="center",
                fontsize=8, color="white" if heat[i, j] > heat.max() * 0.55 else "#333333",
            )

    # Highlight current regime
    if s.kappa_s in ks_grid and s.kappa_p in kp_grid:
        ci, cj = ks_grid.index(s.kappa_s), kp_grid.index(s.kappa_p)
        rect = mpatches.FancyBboxPatch(
            (cj - 0.48, ci - 0.48), 0.96, 0.96,
            boxstyle="square,pad=0", lw=2, edgecolor="#D85A30", facecolor="none",
        )
        ax.add_patch(rect)

    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — Supplier & Part scatter
# ---------------------------------------------------------------------------

def plot_supplier_part_scatter(world: World) -> plt.Figure:
    """
    Figure 3 — θ_s strip-plot and θ_ps grouped scatter.

    Layout (1 × 2):
    ┌──────────────────────────┬────────────────────────────┐
    │ Strip-plot θ_s           │ Scatter θ_ps               │
    │ (one dot per supplier,   │ (one dot per part, grouped │
    │  m_global reference line)│  by supplier row)          │
    └──────────────────────────┴────────────────────────────┘

    Parameters
    ----------
    world : World

    Returns
    -------
    matplotlib.figure.Figure
    """
    s = _get_settings(world)
    suppliers = list(_iter_suppliers_in_period0(world))

    if not suppliers:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No supplier data found in period 0.",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    n_sup = len(suppliers)
    sup_labels = [f"S{sup.supplier_id}" for sup in suppliers]
    sup_colors = [_PALETTE[i % len(_PALETTE)] for i in range(n_sup)]

    fig, axes = _fig(1, 2, figsize=(12, max(4, n_sup * 0.5 + 2)))
    fig.suptitle(
        f"Supplier & Part theta distribution  —  κs={s.kappa_s}  κp={s.kappa_p}",
        fontsize=12, fontweight="normal",
    )

    rng_jitter = np.random.default_rng(seed=42)

    # ── Left: Strip-plot θ_s ──────────────────────────────────────────────
    ax = axes[0, 0]
    _apply_style(ax)

    for idx, sup in enumerate(suppliers):
        st = _supplier_theta(sup)
        y = idx + 1
        if st is not None:
            ax.scatter(st, y, color=sup_colors[idx], s=90, zorder=4, edgecolors="white",
                       linewidths=0.8)
        else:
            # Legacy path: use supplier_risk_factor as offset from a mean
            legacy_val = sup.supplier_risk_factor
            ax.scatter(legacy_val, y, color=sup_colors[idx], s=90, zorder=4,
                       marker="D", edgecolors="white", linewidths=0.8)

    # Reference line: m_global
    ax.axvline(s.m_global, color="#D85A30", lw=1.5, ls="--",
               label=f"m_global = {s.m_global:.2f}", zorder=3)

    # Whisker: show ± 1 SD from theory
    var_s = s.m_global * (1 - s.m_global) / (s.kappa_s + 1)
    sd_s = math.sqrt(var_s)
    ax.axvspan(
        max(0, s.m_global - 2 * sd_s), min(1, s.m_global + 2 * sd_s),
        alpha=0.07, color=_PALETTE[0], label="±2 SD theory",
    )

    ax.set_yticks(range(1, n_sup + 1))
    ax.set_yticklabels(sup_labels, fontsize=8)
    ax.set_xlabel("θ_s", fontsize=_STYLE_DEFAULTS["label_fontsize"])
    ax.set_title(
        "Supplier-level risk θ_s\n(marker = drawn value, dashed = population mean)",
        fontsize=_STYLE_DEFAULTS["title_fontsize"],
    )
    ax.set_xlim(-0.02, 1.02)
    ax.legend(fontsize=8)

    # ── Right: Scatter θ_ps grouped by supplier row ───────────────────────
    ax = axes[0, 1]
    _apply_style(ax)

    for idx, sup in enumerate(suppliers):
        y_base = idx + 1
        st = _supplier_theta(sup)
        part_thetas = [p.base_true_part_theta for p in sup.part]

        if not part_thetas:
            continue

        jitter = rng_jitter.uniform(-0.22, 0.22, size=len(part_thetas))
        ax.scatter(
            part_thetas, [y_base + j for j in jitter],
            color=sup_colors[idx], alpha=0.75, s=45,
            edgecolors="white", linewidths=0.5, zorder=4,
            label=f"S{sup.supplier_id}",
        )

        # Crosshair marker at θ_s
        if st is not None:
            ax.scatter(
                st, y_base, color=sup_colors[idx], s=140, marker="|",
                linewidths=2.5, zorder=5,
            )

        # Horizontal line spanning part range within supplier
        ax.hlines(
            y_base, min(part_thetas), max(part_thetas),
            colors=sup_colors[idx], alpha=0.3, linewidth=1,
        )

    ax.set_yticks(range(1, n_sup + 1))
    ax.set_yticklabels(sup_labels, fontsize=8)
    ax.set_xlabel("θ_ps", fontsize=_STYLE_DEFAULTS["label_fontsize"])
    ax.set_title(
        "Part-level risk θ_ps per supplier\n(dots = parts, bar = supplier θ_s)",
        fontsize=_STYLE_DEFAULTS["title_fontsize"],
    )
    ax.set_xlim(-0.02, 1.02)
    ax.axvline(s.m_global, color="#D85A30", lw=1, ls="--", alpha=0.5,
               label=f"m_global = {s.m_global:.2f}")
    ax.legend(fontsize=7, ncol=2, loc="lower right")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — Call-off time series
# ---------------------------------------------------------------------------

def plot_calloff_timeseries(world: World) -> plt.Figure:
    """
    Figure 4 — Critical call-off time series, period totals, and empirical distribution.

    Layout (2 × 2):
    ┌────────────────────────────────────────────┬──────────────────────┐
    │ Critical call-off rate per part (line)     │ Posterior mean per   │
    │ — one line per part, colored by supplier   │ part over time       │
    ├────────────────────────────────────────────┴──────────────────────┤
    │ Total critical calloffs per period (bar)   │ θ_ps empirical vs    │
    │                                            │ Beta theoretical     │
    └────────────────────────────────────────────┴──────────────────────┘

    Parameters
    ----------
    world : World
        Must have at least 2 realized periods (world.current_period >= 1).

    Returns
    -------
    matplotlib.figure.Figure
    """
    s = _get_settings(world)
    periods = _get_all_periods_sorted(world)
    realized = [p for p in periods if p.period_id <= world.current_period]

    if not realized:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No realized periods found. Run step_world_forward() first.",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    period_ids = [p.period_id for p in realized]

    # ── Collect per-part series data ──────────────────────────────────────
    # { part_id: { "supplier_id": int, "true_theta": float,
    #              "rates": list[float], "posterior_means": list[float] } }
    part_data: dict[int, dict] = {}
    sup_id_list: list[int] = []

    p0 = world.get_period(0)
    if p0 is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Period 0 not found.", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    for sup in p0.supplier:
        if sup.supplier_id not in sup_id_list:
            sup_id_list.append(sup.supplier_id)
        for part in sup.part:
            part_data[part.part_id] = {
                "supplier_id": sup.supplier_id,
                "true_theta": part.base_true_part_theta,
                "rates": [],
                "posterior_means": [],
            }

    for pid in period_ids:
        period = world.get_period(pid)
        if period is None:
            continue
        for sup in period.supplier:
            for part in sup.part:
                ps = part.get_period_state(pid)
                entry = part_data.get(part.part_id)
                if entry is None:
                    continue
                if ps is not None and ps.n_actual_delivery_calloffs > 0:
                    rate = ps.n_critical_delivery_calloffs / ps.n_actual_delivery_calloffs
                    pm   = ps.posterior_mean
                else:
                    rate = float("nan")
                    pm   = float("nan")
                entry["rates"].append(rate)
                entry["posterior_means"].append(pm)

    # ── Total criticals per period ────────────────────────────────────────
    total_criticals: list[int] = []
    for pid in period_ids:
        period = world.get_period(pid)
        tot = 0
        if period:
            for sup in period.supplier:
                for part in sup.part:
                    ps = part.get_period_state(pid)
                    if ps:
                        tot += ps.n_critical_delivery_calloffs
        total_criticals.append(tot)

    # ── Assign colors per supplier ────────────────────────────────────────
    sup_color_map = {sid: _PALETTE[i % len(_PALETTE)] for i, sid in enumerate(sup_id_list)}
    linestyles = ["-", "--", "-.", ":"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor="white")
    fig.suptitle(
        f"Call-off time series  —  κs={s.kappa_s}  κp={s.kappa_p}  "
        f"periods={world.current_period + 1}",
        fontsize=12, fontweight="normal",
    )

    period_labels = [f"P{pid}" for pid in period_ids]

    # ── Top-left: Observed critical rate per part ─────────────────────────
    ax = axes[0, 0]
    _apply_style(ax)
    part_counter_by_sup: dict[int, int] = defaultdict(int)
    legend_handles: list = []
    added_sups: set = set()

    for part_id, entry in part_data.items():
        sid = entry["supplier_id"]
        col = sup_color_map[sid]
        ls  = linestyles[part_counter_by_sup[sid] % len(linestyles)]
        part_counter_by_sup[sid] += 1

        rates = entry["rates"]
        if len(rates) != len(period_ids):
            continue

        ax.plot(
            period_labels, rates,
            color=col, lw=1.3, ls=ls, alpha=0.75, marker="o", markersize=3,
        )
        # Horizontal dashed true theta line
        ax.axhline(
            entry["true_theta"], color=col, lw=0.7, ls=":", alpha=0.4,
        )
        if sid not in added_sups:
            legend_handles.append(
                mpatches.Patch(color=col, label=f"S{sid}")
            )
            added_sups.add(sid)

    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("critical rate (observed)", fontsize=_STYLE_DEFAULTS["label_fontsize"])
    ax.set_title(
        "Critical call-off rate per part\n(dotted horizontal = true θ_ps)",
        fontsize=_STYLE_DEFAULTS["title_fontsize"],
    )
    ax.tick_params(axis="x", rotation=45)
    ax.legend(handles=legend_handles, fontsize=8, ncol=2, loc="upper right")

    # ── Top-right: Posterior mean per part ────────────────────────────────
    ax = axes[0, 1]
    _apply_style(ax)
    part_counter_by_sup = defaultdict(int)

    for part_id, entry in part_data.items():
        sid = entry["supplier_id"]
        col = sup_color_map[sid]
        ls  = linestyles[part_counter_by_sup[sid] % len(linestyles)]
        part_counter_by_sup[sid] += 1

        pms = entry["posterior_means"]
        if len(pms) != len(period_ids):
            continue

        ax.plot(
            period_labels, pms,
            color=col, lw=1.3, ls=ls, alpha=0.75, marker="s", markersize=3,
        )
        ax.axhline(
            entry["true_theta"], color=col, lw=0.7, ls=":", alpha=0.4,
        )

    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("posterior mean E[θ | data]", fontsize=_STYLE_DEFAULTS["label_fontsize"])
    ax.set_title(
        "Bayesian posterior mean per part\n(dotted horizontal = true θ_ps)",
        fontsize=_STYLE_DEFAULTS["title_fontsize"],
    )
    ax.tick_params(axis="x", rotation=45)

    # ── Bottom-left: Bar chart total criticals per period ─────────────────
    ax = axes[1, 0]
    _apply_style(ax)
    bars = ax.bar(
        period_labels, total_criticals,
        color=_PALETTE[0], alpha=0.65, edgecolor=_PALETTE[0], linewidth=0.5,
    )
    # Trend line
    if len(total_criticals) > 1:
        x_num = np.arange(len(total_criticals))
        z = np.polyfit(x_num, total_criticals, 1)
        p_fit = np.poly1d(z)
        ax.plot(period_labels, p_fit(x_num), color="#D85A30", lw=1.5, ls="--",
                label=f"trend  slope={z[0]:+.1f}/period")
        ax.legend(fontsize=8)

    ax.set_ylabel("total critical calloffs", fontsize=_STYLE_DEFAULTS["label_fontsize"])
    ax.set_title(
        "Total critical call-offs per period",
        fontsize=_STYLE_DEFAULTS["title_fontsize"],
    )
    ax.tick_params(axis="x", rotation=45)

    # ── Bottom-right: Empirical θ_ps distribution vs Beta theory ──────────
    ax = axes[1, 1]
    _apply_style(ax)
    all_part_thetas = [entry["true_theta"] for entry in part_data.values()]

    if all_part_thetas:
        ax.hist(
            all_part_thetas,
            bins=min(25, max(5, len(all_part_thetas) // 2)),
            density=True, color=_PALETTE[1], alpha=0.55,
            edgecolor="white", linewidth=0.5, label="empirical θ_ps",
        )
        if s.use_hierarchical_theta:
            supplier_thetas_list = [
                _supplier_theta(sup)
                for sup in _iter_suppliers_in_period0(world)
                if _supplier_theta(sup) is not None
            ]
            mean_ts = float(np.mean(supplier_thetas_list)) if supplier_thetas_list else s.m_global
            a_p = mean_ts * s.kappa_p
            b_p = (1 - mean_ts) * s.kappa_p
            x, pdf = _beta_pdf_curve(a_p, b_p)
            ax.plot(x, pdf, color=_PALETTE[1], lw=2,
                    label=f"Beta({a_p:.1f}, {b_p:.1f}) theory")

        ax.axvline(float(np.mean(all_part_thetas)), color="#D85A30", lw=1.2, ls="--",
                   label=f"empirical mean = {np.mean(all_part_thetas):.3f}")

    ax.set_xlabel("θ_ps", fontsize=_STYLE_DEFAULTS["label_fontsize"])
    ax.set_ylabel("density", fontsize=_STYLE_DEFAULTS["label_fontsize"])
    ax.set_title(
        "Empirical θ_ps vs Beta theoretical\n(using mean θ_s as conditioning point)",
        fontsize=_STYLE_DEFAULTS["title_fontsize"],
    )
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Convenience: plot_all
# ---------------------------------------------------------------------------

def plot_all(
    world: World,
    save_dir: Optional[str] = None,
    dpi: int = 150,
) -> dict[str, plt.Figure]:
    """
    Render all four figures and optionally save them to PNG files.

    Parameters
    ----------
    world : World
        Simulation world with at least one realized period.
    save_dir : str, optional
        If provided, each figure is saved as a PNG in this directory.
        Filenames: fig1_beta_distributions.png, fig2_parameter_sensitivity.png,
                   fig3_supplier_part_scatter.png, fig4_calloff_timeseries.png
    dpi : int
        Resolution for saved PNG files. Default 150.

    Returns
    -------
    dict mapping short name → Figure object.
    """
    figs = {
        "beta_distributions":    plot_beta_distributions(world),
        "parameter_sensitivity": plot_parameter_sensitivity(world),
        "supplier_part_scatter": plot_supplier_part_scatter(world),
        "calloff_timeseries":    plot_calloff_timeseries(world),
    }

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        for i, (name, fig) in enumerate(figs.items(), start=1):
            path = os.path.join(save_dir, f"fig{i}_{name}.png")
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            print(f"Saved: {path}")

    return figs