import os
import random
import json
from uuid import uuid4
import matplotlib.pyplot as plt
import pandas as pd
from src.schemas import InputData, ScenarioSet
from src.models.config.part_to_dispatcher_workload_balancing_model_config import (
    PartToDispatcherWorkloadBalancingModelConfig,
)

from src.models.part_to_dispatcher_workload_balancing_model import (
    PartToDispatcherWorkloadBalancingModel,
)
from matplotlib.colors import LinearSegmentedColormap


def run_a_case():

    # Setup input_data
    list_suppliers = [str(i) for i in range(1,7)]

    supplier_info = {}
    for supplier in list_suppliers:
        supplier_info[supplier] = {
            "supplier_id": supplier,
            "parts": [
                str(i) for i in range ((int(supplier) - 1) * 10 + 1, int(supplier) * 10 + 1)
            ],
            "base_true_theta": -1
        }

    part_info = {}
    for i in range(1, 61):
        supplier_id = str((i - 1) // 10 + 1)
        if supplier_id == "1":
            if (i - 1) % 10 <= 7:
                base_true_theta = 0.8
            else:
                base_true_theta = 0.2
        else:
            if (i - 1) % 10 <= 7:
                base_true_theta = 0.2
            else:
                base_true_theta = 0.8
        part_info[str(i)] = {
            "part_id": str(i),
            "supplier_id": supplier_id,
            "n_obs": -1,
            "k_obs": -1,
            "deliveries": {},
            "base_true_theta": base_true_theta
        }

    input_data_raw = {
        "id": str(uuid4()),
        "parts": part_info,
        "suppliers": supplier_info,
        "dispatchers": {
            "1": {
                "dispatcher_id": "1",
                "fte_factor": 0.15,
                "target_workload": {
                "Mon": 180.0,
                }
            },
            "2": {
                "dispatcher_id": "2",
                "fte_factor": 0.15,
                "target_workload": {
                "Mon": 180.0,
                }
            },
            "3": {
                "dispatcher_id": "3",
                "fte_factor": 0.15,
                "target_workload": {
                "Mon": 180.0,
                }
            }
        },
        "rep_days": [
        "Mon",
        ],
    }

    input_data = InputData.model_validate(input_data_raw)

    input_data.save("shared_runs/special/a/input_data.json")

    # Setup scenario set
    critical_handling_time = 30.0
    critical_records = []
    scenario_index = 0
    while True:
        one_scenario = []
        for part in input_data.parts.values():
            if random.random() < part.base_true_theta:
                critical_count = 1
                workload_contribution_min = critical_handling_time
            else:
                critical_count = 0
                workload_contribution_min = 0
            one_scenario.append({
                "part_id": part.part_id,
                "supplier_id": part.supplier_id,
                "day": "Mon",
                "scenario_index": scenario_index,
                "n_deliveries": 1,
                "critical_count": critical_count,
                "workload_contribution_min": workload_contribution_min
            })
        total_critical = sum([ele["critical_count"] for ele in one_scenario])
        if total_critical == 18:
            critical_records.extend(one_scenario)
            scenario_index += 1
            print(f"Generate scenario {scenario_index} successfully")

        if scenario_index == 50:
            break

    scenario_set_raw = {
        "scenario_set_id": str(uuid4()),
        "data_id": input_data.id,
        "config": {
            "rng_seed": 0,
            "n_scenarios": 50,
            "alpha0": 1.0,
            "beta0": 1.0,
            "handling_time": critical_handling_time
        },
        "n_parts": 60,
        "n_rep_days": 1,
        "theta_records": [],
        "critical_records": critical_records,
        "part_stats": [],
        "supplier_stats": [],
    }

    scenario_set = ScenarioSet.model_validate(scenario_set_raw)
    scenario_set.save("shared_runs/special/a/scenario_set.json")

    # Run STD
    std_config = PartToDispatcherWorkloadBalancingModelConfig(
        rng_seed=42,
        cost_deviation=1.0,
        cost_coord=1.0,
        k_max=0,
        time_limit=5 * 60,
        mip_gap=0.01,
    )
    std_model = PartToDispatcherWorkloadBalancingModel(
            input_data=input_data,
            model_config=std_config,
            scenario_set=scenario_set,
            use_scenario_based_target=False,
            use_coord_in_objective=False,
            n_split_exact=False,
            use_underload_penalty=False,
            warm_start_assignment=None
        )
    std_model.solve()
    std_model.save("shared_runs/special/a/models/ptd_k_max_0")

    # Run PTD
    ptd_config = PartToDispatcherWorkloadBalancingModelConfig(
        rng_seed=42,
        cost_deviation=1.0,
        cost_coord=1.0,
        k_max=6,
        time_limit=5 * 60,
        mip_gap=0.01,
    )
    ptd_model = PartToDispatcherWorkloadBalancingModel(
            input_data=input_data,
            model_config=ptd_config,
            scenario_set=scenario_set,
            use_scenario_based_target=False,
            use_coord_in_objective=False,
            n_split_exact=False,
            use_underload_penalty=False,
            warm_start_assignment=None
        )
    ptd_model.solve()
    ptd_model.save("shared_runs/special/a/models/ptd_k_max_6")


def run_b_case():
    # Setup input_data
    list_suppliers = [str(i) for i in range(1,7)]

    supplier_info = {}
    for supplier in list_suppliers:
        supplier_info[supplier] = {
            "supplier_id": supplier,
            "parts": [
                str(i) for i in range ((int(supplier) - 1) * 10 + 1, int(supplier) * 10 + 1)
            ],
            "base_true_theta": -1
        }

    part_info = {}
    for i in range(1, 61):
        supplier_id = str((i - 1) // 10 + 1)
        part_info[str(i)] = {
            "part_id": str(i),
            "supplier_id": supplier_id,
            "n_obs": -1,
            "k_obs": -1,
            "deliveries": {},
            "base_true_theta": 0.3
        }

    input_data_raw = {
        "id": str(uuid4()),
        "parts": part_info,
        "suppliers": supplier_info,
        "dispatchers": {
        "1": {
            "dispatcher_id": "1",
            "fte_factor": 0.15,
            "target_workload": {
            "Mon": 180.0,
            }
        },
        "2": {
            "dispatcher_id": "2",
            "fte_factor": 0.15,
            "target_workload": {
            "Mon": 180.0,
            }
        },
        "3": {
            "dispatcher_id": "3",
            "fte_factor": 0.15,
            "target_workload": {
            "Mon": 180.0,
            }
        }
        },
        "rep_days": [
        "Mon",
        ],
    }
    input_data = InputData.model_validate(input_data_raw)
    input_data.save("shared_runs/special/b/input_data.json")

    # Setup scenario set
    critical_handling_time = 30.0
    critical_records = []
    scenario_index = 0
    while True:
        one_scenario = []
        number_of_critical_per_supplier = {
            ele : 0 for ele in input_data.suppliers.keys()
        }
        for part in input_data.parts.values():
            if random.random() < part.base_true_theta:
                critical_count = 1
                workload_contribution_min = critical_handling_time
                number_of_critical_per_supplier[part.supplier_id] += 1
            else:
                critical_count = 0
                workload_contribution_min = 0
            one_scenario.append({
                "part_id": part.part_id,
                "supplier_id": part.supplier_id,
                "day": "Mon",
                "scenario_index": scenario_index,
                "n_deliveries": 1,
                "critical_count": critical_count,
                "workload_contribution_min": workload_contribution_min
            })

        accept = True
        for ele in number_of_critical_per_supplier.values():
            if ele not in [2,3,4]:
                accept = False
        if not accept:
            continue

        total_critical = sum([ele["critical_count"] for ele in one_scenario])
        if total_critical == 18:
            critical_records.extend(one_scenario)
            scenario_index += 1
            print(f"Generate scenario {scenario_index} successfully")

        if scenario_index == 50:
            break

    scenario_set_raw = {
        "scenario_set_id": str(uuid4()),
        "data_id": input_data.id,
        "config": {
            "rng_seed": 0,
            "n_scenarios": 50,
            "alpha0": 1.0,
            "beta0": 1.0,
            "handling_time": critical_handling_time
        },
        "n_parts": 60,
        "n_rep_days": 1,
        "theta_records": [],
        "critical_records": critical_records,
        "part_stats": [],
        "supplier_stats": [],
    }

    scenario_set = ScenarioSet.model_validate(scenario_set_raw)
    scenario_set.save("shared_runs/special/b/scenario_set.json")

    # Run STD
    std_config = PartToDispatcherWorkloadBalancingModelConfig(
        rng_seed=42,
        cost_deviation=1.0,
        cost_coord=1.0,
        k_max=0,
        time_limit=5 * 60,
        mip_gap=0.01,
    )
    std_model = PartToDispatcherWorkloadBalancingModel(
            input_data=input_data,
            model_config=std_config,
            scenario_set=scenario_set,
            use_scenario_based_target=False,
            use_coord_in_objective=False,
            n_split_exact=False,
            use_underload_penalty=False,
            warm_start_assignment=None
        )
    std_model.solve()
    std_model.save("shared_runs/special/b/models/ptd_k_max_0")

    # Run PTD
    ptd_config = PartToDispatcherWorkloadBalancingModelConfig(
        rng_seed=42,
        cost_deviation=1.0,
        cost_coord=1.0,
        k_max=6,
        time_limit=5 * 60,
        mip_gap=0.01,
    )
    ptd_model = PartToDispatcherWorkloadBalancingModel(
            input_data=input_data,
            model_config=ptd_config,
            scenario_set=scenario_set,
            use_scenario_based_target=False,
            use_coord_in_objective=False,
            n_split_exact=False,
            use_underload_penalty=False,
            warm_start_assignment=None
        )
    ptd_model.solve()
    ptd_model.save("shared_runs/special/b/models/ptd_k_max_6")



def plot_workload_distribution(scenario_set):

    supplier_workload = (
        pd.DataFrame(scenario_set.critical_records)
        .groupby(["supplier_id", "scenario_index"])
        ["workload_contribution_min"]
        .sum()
        .groupby("supplier_id")
        .mean()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(
        supplier_workload.index,
        supplier_workload.values,
    )

    # Add values on top of bars
    for bar, value in zip(bars, supplier_workload.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            f"{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel("Supplier")
    ax.set_ylabel("Workload")
    ax.set_title("Average Workload Distribution by Supplier")

    ax.axhline(
        180,
        linestyle="--",
        label="Average dispatcher workload",
    )

    ax.legend()

    fig.tight_layout()

    return fig

def plot_workload_distribution_by_part_and_supplier(scenario_set):

    part_df = (
        pd.DataFrame(scenario_set.critical_records)
        .groupby(
            ["supplier_id", "part_id"],
            as_index=False,
        )["workload_contribution_min"]
        .mean()
        .sort_values(
            ["supplier_id", "part_id"]
        )
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(15, 6))

    x = range(len(part_df))

    bars = ax.bar(
        x,
        part_df["workload_contribution_min"],
        width=0.8,
    )

    # Value labels
    for bar, value in zip(
        bars,
        part_df["workload_contribution_min"],
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Part labels
    ax.set_xticks(x)
    ax.set_xticklabels(
        part_df["part_id"],
        rotation=90,
    )

    ax.set_xlabel("Part")
    ax.set_ylabel("Average Workload")
    ax.set_title(
        "Average Workload Distribution by Part and Supplier",
        pad=50,
    )

    # Supplier groups
    ymax = max(
        part_df["workload_contribution_min"].max(),
        1,
    )

    start = 0

    for supplier_id, group in part_df.groupby(
        "supplier_id",
        sort=False,
    ):
        end = start + len(group) - 1
        center = (start + end) / 2

        # Supplier name
        ax.text(
            center,
            ymax * 1.20,
            supplier_id,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

        # Supplier bracket
        bracket_y = ymax * 1.14

        ax.plot(
            [start - 0.4, end + 0.4],
            [bracket_y, bracket_y],
            linewidth=1,
        )

        ax.plot(
            [start - 0.4, start - 0.4],
            [
                bracket_y - ymax * 0.025,
                bracket_y,
            ],
            linewidth=1,
        )

        ax.plot(
            [end + 0.4, end + 0.4],
            [
                bracket_y - ymax * 0.025,
                bracket_y,
            ],
            linewidth=1,
        )

        # Separator
        if start > 0:
            ax.axvline(
                start - 0.5,
                linestyle="--",
                alpha=0.5,
            )

        start = end + 1

    ax.set_ylim(
        0,
        ymax * 1.35,
    )

    fig.tight_layout()

    return fig

def plot_dispatcher_workload_in_two_models(output_std, output_ptd):
    ptd_workload = {
        dispatcher: values.get("Mon", 0)
        for dispatcher, values
        in output_ptd["workload_stats"].items()
    }

    std_workload = {
        dispatcher: values.get("Mon", 0)
        for dispatcher, values
        in output_std["workload_stats"].items()
    }

    dispatcher_df = pd.DataFrame({
        "STD": pd.Series(std_workload),
        "PTD": pd.Series(ptd_workload),
    }).fillna(0).sort_index()


    ax = dispatcher_df.plot(
        kind="bar",
        figsize=(10, 5),
        color={
            "PTD": "green",
            "STD": "tab:orange",
        },
    )

    fig = ax.figure

    # ============================================================
    # Add values on top of bars
    # ============================================================

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.0f",
            padding=5,       
            fontsize=10,
        )

    # ============================================================
    # Labels & title
    # ============================================================

    ax.set_xlabel("Dispatcher")
    ax.set_ylabel("Workload (minutes)")

    ax.set_title(
        "Dispatcher Workload: PTD vs STD",
        pad=12,
    )

    plt.xticks(rotation=0)

    # ============================================================
    # Fixed target workload
    # ============================================================

    ax.axhline(
        180,
        linestyle="--",
        label="Fixed target workload",
    )

    # ============================================================
    # Add more space at the top
    # ============================================================

    ymax = dispatcher_df.max().max()
    ax.set_ylim(0, ymax + 15)

    # ============================================================
    # Legend outside the plot
    # ============================================================

    ax.legend(
        title="Model",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
    )

    fig.tight_layout()

    return fig

def plot_supplier_allocation_by_dispatcher(part_to_supplier, output_std, output_ptd):
    def assignment_to_df(assignment, model_name):
        records = []
        for part_id, dispatcher_id in assignment.items():
        
            records.append({
                "model": model_name,
                "dispatcher": f"D{dispatcher_id}",
                "supplier": f"S{part_to_supplier[part_id]}",
                "part": part_id,
            })
        

        return pd.DataFrame(records)

    ptd_assignment = output_ptd["assignment"]
    std_assignment = output_std["assignment"]

    ptd_df = assignment_to_df(
        ptd_assignment,
        "PTD",
    )

    std_df = assignment_to_df(
        std_assignment,
        "STD",
    )

    assignment_df = pd.concat(
        [ptd_df, std_df],
        ignore_index=True,
    )

    supplier_dispatcher_df = (
        assignment_df
        .groupby(
            [
                "model",
                "dispatcher",
                "supplier",
            ]
        )
        .agg(
            n_parts=("part", "nunique")
        )
        .reset_index()
    )

    cmap = LinearSegmentedColormap.from_list(
        "soft_red",
        [
            "#FFF5F3",  # 0 - gần trắng
            "#FDE2DE",  # rất nhạt
            "#F8C5BD",  # nhạt
            "#EFA39A",  # trung bình
            "#D9786D",  # đậm
        ],
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6, 6),
        sharey=True,
    )

    for ax, model in zip(
        axes,
        ["PTD", "STD"],
    ):

        data = (
            supplier_dispatcher_df[
                supplier_dispatcher_df["model"] == model
            ]
            .pivot(
                index="supplier",
                columns="dispatcher",
                values="n_parts",
            )
            .fillna(0)
        )

        ax.imshow(
            data.values,
            cmap=cmap,
            vmin=0,
        )

        ax.set_xticks(range(len(data.columns)))
        ax.set_xticklabels(data.columns)

        ax.set_yticks(range(len(data.index)))
        ax.set_yticklabels(data.index)

        ax.set_xlabel("Dispatcher")
        ax.set_title(model)

        # --------------------------------------------------------
        # Values inside cells
        # --------------------------------------------------------

        for i in range(len(data.index)):
            for j in range(len(data.columns)):

                value = data.iloc[i, j]

                if value > 0:
                    ax.text(
                        j,
                        i,
                        f"{value:.0f}",
                        ha="center",
                        va="center",
                        fontsize=11,
                        fontweight="bold",
                    )

    fig.suptitle(
        "Supplier Allocation by Dispatcher: PTD vs STD",
        fontsize=14,
    )


    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.87,
        bottom=0.10,
        wspace=0.08,
    )

    return fig

def plot_all_for_special_case():
    for case in ['a', 'b']:
        input_data = InputData.load(f"shared_runs/special/{case}/input_data.json")
        scenario_set = ScenarioSet.load(f"shared_runs/special/{case}/scenario_set.json")
        with open(f"shared_runs/special/{case}/models/ptd_k_max_0/solve_result.json") as f1:
            output_std = json.load(f1)
        with open(f"shared_runs/special/{case}/models/ptd_k_max_6/solve_result.json") as f1:
            output_ptd = json.load(f1)
        save_dir = f"shared_runs/special/{case}/plots"
        os.makedirs(save_dir, exist_ok=True)

        fig1 = plot_workload_distribution(scenario_set=scenario_set)
        fig1.savefig(
            os.path.join(save_dir, "workload_distribution.png"),
            dpi=300,
            bbox_inches="tight",
        )

        fig2 = plot_workload_distribution_by_part_and_supplier(scenario_set=scenario_set)
        fig2.savefig(
            os.path.join(save_dir, "workload_distribution_by_part_and_supplier.png"),
            dpi=300,
            bbox_inches="tight",
        )

        fig3 = plot_dispatcher_workload_in_two_models(output_std=output_std, output_ptd=output_ptd)
        fig3.savefig(
            os.path.join(save_dir, "dispatcher_workload.png"),
            dpi=300,
            bbox_inches="tight",
        )

        part_to_supplier = {ele.part_id: ele.supplier_id for ele in input_data.parts.values()}

        fig4 = plot_supplier_allocation_by_dispatcher(
            part_to_supplier=part_to_supplier,
            output_std=output_std,
            output_ptd=output_ptd
        )
        fig4.savefig(
            os.path.join(save_dir, "supplier_allocation_by_dispatcher.png"),
            dpi=300,
            bbox_inches="tight",
        )





def main():

    run_a_case()

    run_b_case()
    plot_all_for_special_case()

if __name__ == '__main__':
    main()