import os
import json
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.simulation import World, load_world

def plot_pareto_knee(
    df,
    x_col="n_split_suppliers",
    y_col="deviation",
    xlabel="Number of Split Suppliers",
    ylabel="Deviation",
    title="Pareto Frontier",
    ax=None,
    plot_knee=True
):
    """
    Plot Pareto frontier and highlight knee point.

    Assume both objectives are minimized.
    """

    # ---------- Pareto ----------
    pareto = (
        df.sort_values([x_col, y_col])
          .sort_values(y_col)
          .reset_index(drop=True)
    )
    
    # ---------- Normalize ----------
    x = pareto[x_col].to_numpy(dtype=float)
    y = pareto[y_col].to_numpy(dtype=float)

    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    # ---------- Distance to line ----------
    A = np.array([x_norm[0], y_norm[0]])
    B = np.array([x_norm[-1], y_norm[-1]])

    BA = B - A

    distances = []

    for p in np.column_stack([x_norm, y_norm]):
        # d = abs(np.cross(BA, p - A)) / np.linalg.norm(BA)
        AP = p - A
        
        cross_2d = BA[0] * AP[1] - BA[1] * AP[0]

        d = abs(cross_2d) / np.linalg.norm(BA)
        distances.append(d)

    distances = np.array(distances)

    knee_idx = np.argmax(distances)

    # ---------- Projection of Knee onto AB ----------
    P = np.array([
        x_norm[knee_idx],
        y_norm[knee_idx]
    ])

    BA = B - A

    t = np.dot(P - A, BA) / np.dot(BA, BA)

    proj = A + t * BA

    proj_x = (
        proj[0] * (x.max() - x.min())
        + x.min()
    )

    proj_y = (
        proj[1] * (y.max() - y.min())
        + y.min()
    )

    # ---------- Create ax only if needed ----------
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    # ---------- Pareto ----------
    ax.scatter(
        pareto[x_col],
        pareto[y_col],
        s=70,
        label="Pareto"
    )

    ax.plot(
        pareto[x_col],
        pareto[y_col],
        "--"
    )

    if plot_knee and len(pareto) > 2:
        # ---------- Knee Point ----------
        ax.scatter(
            pareto.loc[knee_idx, x_col],
            pareto.loc[knee_idx, y_col],
            s=180,
            marker="*",
            label="Knee Point"
        )

        ax.annotate(
            "Knee",
            (
                pareto.loc[knee_idx, x_col],
                pareto.loc[knee_idx, y_col],
            ),
            xytext=(-20, -20),
            textcoords="offset points",
        )

        # ---------- A-B Line ----------
        ax.plot(
            [
                pareto.iloc[0][x_col],
                pareto.iloc[-1][x_col]
            ],
            [
                pareto.iloc[0][y_col],
                pareto.iloc[-1][y_col]
            ],
            "--",
            color="gray",
            label="A-B Line"
        )

        # ---------- Perpendicular Distance ----------
        ax.plot(
            [
                pareto.iloc[knee_idx][x_col],
                proj_x
            ],
            [
                pareto.iloc[knee_idx][y_col],
                proj_y
            ],
            color="red",
            linewidth=2,
            label="Perpendicular Distance"
        )

        # ---------- Distance Annotation ----------
        mid_x = (
            pareto.iloc[knee_idx][x_col]
            + proj_x
        ) / 2

        mid_y = (
            pareto.iloc[knee_idx][y_col]
            + proj_y
        ) / 2

        ax.annotate(
            f"d = {distances[knee_idx]:.3f}",
            (mid_x, mid_y),
            xytext=(mid_x + 3, mid_y + 3),
            arrowprops=dict(
                arrowstyle="->",
                color="red"
            ),
            color="red",
            bbox=dict(
                facecolor="white",
                edgecolor="gray",
                alpha=0.85
            )
        )

    # ---------- Formatting ----------
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.grid(alpha=0.3)
    ax.legend()

    ax.set_aspect(
        (x.max() - x.min())
        / (y.max() - y.min())
    )

    return pareto, pareto.iloc[knee_idx], fig


def plot_all_experiments(
    experiment_results,
    x_col="n_split_suppliers",
    y_col="deviation",
    xlabel="Number of Split Suppliers",
    ylabel="Deviation",
    title="Pareto Frontier",
    plot_knee=True
):

    n_experiments = len(experiment_results)

    n_cols = 3
    n_rows = int(np.ceil(n_experiments / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(18, 16)
    )

    axes = np.asarray(axes).reshape(-1)

    results = {}

    for ax, (label, df) in zip(
        axes,
        experiment_results.items()
    ):

        # Vẽ đầy đủ Pareto + Knee + A-B
        pareto, knee, _ = plot_pareto_knee(
            df,
            x_col=x_col,
            y_col=y_col,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            ax=ax,
            plot_knee=plot_knee
        )

        # Đổi title thành experiment
        ax.set_title(label)

        results[label] = {
            "pareto": pareto,
            "knee": knee,
        }

    # Xóa subplot thừa nếu có
    for ax in axes[n_experiments:]:
        ax.remove()

    fig.suptitle(
        "Pareto Frontier and Knee Point - All Experiments",
        fontsize=18
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    return results, fig

def plot_risk(source_data_dir, all_experiments, plot_folder_path):
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
    def _get_settings(world: World):
        return world.settings

    def _iter_suppliers_in_period0(world: World):
        """Yield Supplier objects from period 0 (structural view)."""
        p0 = world.get_period(0)
        if p0 is None:
            return
        yield from p0.supplier

    def _supplier_theta(supplier):
        """Return supplier_theta if set (hierarchical path), else None."""
        return getattr(supplier, "supplier_theta", None)

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
    
    print(f"Total experiments: {len(all_experiments)}")
    for (i, (
        kappa_s,
        kappa_p,
        ks_seed,
        kp_seed,
        general_seed
    )) in enumerate(all_experiments):
        print(f"Running experiment {i+1}/{len(all_experiments)} with ks={kappa_s}, kp={kappa_p}, ks_seed={ks_seed}, kp_seed={kp_seed}") 
        experiment_path = os.path.join(source_data_dir, f"ks_{kappa_s}_kp_{kappa_p}_ksseed_{ks_seed}_kpseed_{kp_seed}_general_seed_{general_seed}")
        world = load_world(os.path.join(experiment_path, "world.json"))

        s = _get_settings(world)

        suppliers = list(_iter_suppliers_in_period0(world))
        n_sup = len(suppliers)

        sup_labels = [f"S{sup.supplier_id}" for sup in suppliers]
        sup_colors = [
            _PALETTE[i % len(_PALETTE)]
            for i in range(n_sup)
        ]

        # ============================================================
        # Figure
        # ============================================================

        fig, axes = _fig(
            1,
            1,
            figsize=(max(8, n_sup * 0.5 + 2), 7),
        )

        ax = axes[0, 0]
        _apply_style(ax)

        rng_jitter = np.random.default_rng(seed=42)

        # ============================================================
        # Plot each supplier
        # ============================================================

        for idx, sup in enumerate(suppliers):

            x_base = idx + 1

            st = _supplier_theta(sup)

            part_thetas = [
                p.base_true_part_theta
                for p in sup.part
            ]

            if not part_thetas:
                continue

            # Jitter theo chiều Y
            jitter = rng_jitter.uniform(
                -0.22,
                0.22,
                size=len(part_thetas),
            )

            ax.scatter(
                [x_base + j for j in jitter],
                part_thetas,
                color=sup_colors[idx],
                alpha=0.75,
                s=45,
                edgecolors="white",
                linewidths=0.5,
                zorder=4,
                label=f"S{sup.supplier_id}",
            )

            # ========================================================
            # Crosshair marker at supplier theta
            # ========================================================

            if st is not None:
                ax.scatter(
                    x_base,
                    st,
                    color=sup_colors[idx],
                    s=140,
                    marker="_",
                    linewidths=2.5,
                    zorder=5,
                )

            # ========================================================
            # Vertical line spanning part range
            # ========================================================

            ax.vlines(
                x_base,
                min(part_thetas),
                max(part_thetas),
                colors=sup_colors[idx],
                alpha=0.3,
                linewidth=1,
            )

        # ============================================================
        # Axes
        # ============================================================

        ax.set_xticks(range(1, n_sup + 1))
        ax.set_xticklabels(
            sup_labels,
            fontsize=8,
            rotation=45,
            ha="right",
        )

        ax.set_ylabel(
            "θ_ps",
            fontsize=_STYLE_DEFAULTS["label_fontsize"],
        )

        ax.set_xlabel(
            "Supplier",
            fontsize=_STYLE_DEFAULTS["label_fontsize"],
        )

        ax.set_title(
            f"Supplier and part theta risk in "
            f"ks={kappa_s}, kp={kappa_p}\n",
            fontsize=_STYLE_DEFAULTS["title_fontsize"],
        )

        ax.set_ylim(-0.02, 1.02)

        # ============================================================
        # Global theta
        # ============================================================

        ax.axhline(
            s.m_global,
            color="#D85A30",
            lw=1,
            ls="--",
            alpha=0.5,
            label=f"m_global = {s.m_global:.2f}",
        )

        fig.savefig(os.path.join(plot_folder_path, f"ks_{kappa_s}_kp_{kappa_p}_risk.png"))

def plot_pareto(source_data_dir, all_experiments, plot_folder_path, plot_knee = False):
    for refining in [False, True]:
        sbt=False
        if refining:
            file_folder = f"pareto_refine_sbt_{sbt}"
            output_prefix = "post"
        else:
            file_folder = f"pareto_sbt_{sbt}"
            output_prefix = "pre"
        experiment_results = {}
        print(f"Total experiments: {len(all_experiments)}")
        for (i, (
            kappa_s,
            kappa_p,
            ks_seed,
            kp_seed,
            general_seed
        )) in enumerate(all_experiments):
            print(f"Running experiment {i+1}/{len(all_experiments)} with ks={kappa_s}, kp={kappa_p}, ks_seed={ks_seed}, kp_seed={kp_seed}") 
            experiment_path = os.path.join(source_data_dir, f"ks_{kappa_s}_kp_{kappa_p}_ksseed_{ks_seed}_kpseed_{kp_seed}_general_seed_{general_seed}")
            
            pair_results = pd.read_csv(os.path.join(experiment_path, file_folder, "pareto_results.csv"),index_col=0)
            # Label cho từng experiment
            label = (
                f"$\\kappa_S={kappa_s}, "
                f"\\kappa_P={kappa_p}$"
            )

            experiment_results[label] = pair_results

        # =====================================================
        # Plot all pareto in a figure
        # =====================================================
        for y_col, ylabel in [("epsilon", "Epsilon"), ("deviation", "Deviation")]:
            if refining and y_col == "epsilon": 
                continue
            _, fig = plot_all_experiments(
                experiment_results,
                y_col=y_col,
                ylabel=ylabel,
                plot_knee=plot_knee
            )

            fig.savefig(
                os.path.join(plot_folder_path, f"all_pareto_{output_prefix}_{y_col}.png"),
                dpi=300,
                bbox_inches="tight"
            )



def main():
    LIST_KAPPA_S = [5, 50, 150]
    LIST_KAPPA_P = [5, 50, 150]

    LIST_RANDOM_SEEDS = [25]

    LIST_DEFAULT_GENERAL_RANDOM_SEED = [50]

    SOURCE_DATA_DIR = "shared_runs/experiment_20260827_230630/data_simulation"


    all_experiments = list(
        itertools.product(
            LIST_KAPPA_S,
            LIST_KAPPA_P,
            LIST_RANDOM_SEEDS,
            LIST_RANDOM_SEEDS,
            LIST_DEFAULT_GENERAL_RANDOM_SEED
        )
    )
    plot_folder_path=os.path.join(SOURCE_DATA_DIR, "report/plots")

    plot_risk(
        source_data_dir=SOURCE_DATA_DIR,
        all_experiments=all_experiments,
        plot_folder_path=plot_folder_path
    )

    plot_pareto(
        source_data_dir=SOURCE_DATA_DIR,
        all_experiments=all_experiments,
        plot_folder_path=plot_folder_path,
        plot_knee=False
    )


if __name__ == "__main__":
    main()