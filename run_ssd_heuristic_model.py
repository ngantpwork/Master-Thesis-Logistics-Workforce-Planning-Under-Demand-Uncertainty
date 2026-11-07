import os
import json
import itertools
import pandas as pd
from src.schemas import InputData, ScenarioSet
from src.models.config.part_to_dispatcher_workload_balancing_model_config import (
    PartToDispatcherWorkloadBalancingModelConfig,
)

from src.models.part_to_dispatcher_workload_balancing_model import (
    PartToDispatcherWorkloadBalancingModel,
)
from src.evaluation.analysis import run_pipeline
from src.models.selective_supplier_decomposition_heuristic import (
    compute_supplier_statistics,
    sort_suppliers_to_split,
    create_split_supplier_set 
)

def aggregation_data_and_analysis(source_data_dir, all_experiments, n_splits = 1, cv_ddof=0, topk=3):
    
    # Aggregation workload data and assignment data
    assignments = []
    workloads = []
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

        input_data = InputData.load(os.path.join(experiment_path, "input_data.json"))
        part_to_supplier = {
            pid: sid
            for sid, supplier in input_data.suppliers.items()
            for pid in supplier.parts
        }

        assignment = pd.read_csv(os.path.join(experiment_path, f"models_pareto/ptd_M4_sbt_False_k_max_equal_{n_splits}/assignment.csv"))
        assignment["config"] = f"ks_{kappa_s}_kp_{kappa_p}"
        assignments.append(assignment)

        scenario_set = ScenarioSet.load(os.path.join(experiment_path, "average_scenario_set.json"))
        one_group_workload = {}
        for key, value in scenario_set.get_part_workload_matrix().items():
            part_id, rep_day, _ = key
            if part_id not in one_group_workload:
                one_group_workload[part_id] = {
                    "config":f"ks_{kappa_s}_kp_{kappa_p}",
                    "supplier_id": part_to_supplier[part_id],
                    "part_id": part_id,
                }
            one_group_workload[part_id][rep_day] = value
        workloads.extend(list(one_group_workload.values()))
    
    assignment_df = pd.concat(assignments)
    assignment_df = assignment_df.rename(columns={
        "dispatcher": "dispatcher_id"
    })
    assignment_df = assignment_df[["config", "supplier_id", "part_id", "dispatcher_id"]]
    assignment_df = assignment_df.astype({
        "config": "string",
        "supplier_id": "string",
        "part_id": "string",
        "dispatcher_id": "string"
    })
    

    workload_df = pd.DataFrame(workloads)
    workload_df = workload_df.fillna(0)
    workload_df = workload_df.astype({
        "config": "string",
        "supplier_id": "string",
        "part_id": "string",
        "Mon": "float64",
        "Tue": "float64",
        "Wed": "float64",
        "Thu": "float64",
        "Fri": "float64",
    })

    # Analysis
    result = run_pipeline(workload_df, assignment_df, cv_ddof=cv_ddof, topk=topk)

    # Save
    save_path = os.path.join(source_data_dir, "analysis")
    os.makedirs(save_path, exist_ok=True)

    assignment_df.to_csv(os.path.join(save_path, "assignment.csv"))
    workload_df.to_csv(os.path.join(save_path, "workload.csv"))
    
    result.feature_table.to_csv(os.path.join(save_path, "feature_table.csv"))
    result.summary_table.to_csv(os.path.join(save_path, "summary_table.csv"))
    result.topk_table.to_csv(os.path.join(save_path, "topk_table.csv"))
    pass

def run_ssd_heuristic_model(source_data_dir, all_experiments):

    workload = pd.read_csv(os.path.join(source_data_dir, "analysis", "workload.csv"), index_col=0)
    supplier_col = "supplier_id"
    part_col = "part_id"
    workload_cols = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    split_supplier_col = "supplier_id_split"

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
        save_path = os.path.join(experiment_path, "selective_supplier_decomposition_heuristic")
        os.makedirs(save_path, exist_ok=True)

        sub_workload = workload[workload.config == f"ks_{kappa_s}_kp_{kappa_p}"]

        supplier_stats = compute_supplier_statistics(
            part_workload=sub_workload,
            workload_cols=workload_cols
        )

        with open(os.path.join(experiment_path, "pareto_refine_sbt_False/knee.json")) as f1:
            n_split_suppliers_knee_point = int(json.load(f1)["n_split_suppliers"])

        supplier_score_df = sort_suppliers_to_split(
            supplier_features=supplier_stats,
            k=n_split_suppliers_knee_point,
            workload_weight=0.5,
            largest_part_share_weight=0.4,
            num_parts_weight=0.1,
            supplier_col="supplier_id",
            workload_col="mean_workload",
            largest_part_share_col="largest_part_share",
            num_parts_col="num_parts"
        )

        selected_suppliers = (
            supplier_score_df[supplier_score_df.selected_for_split == True][supplier_col]
            .tolist()
        )
        split_data = create_split_supplier_set(
            part_workload=sub_workload,
            selected_suppliers=selected_suppliers,
            supplier_col=supplier_col,
            part_col=part_col,
            sub_supplier_col=split_supplier_col,
        )
        new_supplier_to_parts = split_data.groupby("supplier_id_split").agg({
            "part_id": list
        })

        new_input_data = {
            "id": '-1',
            "parts": {
                str(ele["part_id"]): {
                    "part_id": str(ele["part_id"]),
                    "supplier_id": str(ele["supplier_id_split"]),
                    "n_obs": -1,
                    "k_obs": -1,
                    "deliveries": {},
                    "base_true_theta": -1
                } for _, ele in split_data.iterrows()
            },
            "suppliers": {
                str(ele): {
                    "supplier_id": str(ele),
                    "parts": [
                        str(part_id) for part_id in new_supplier_to_parts.loc[ele, 'part_id']
                    ],
                    "base_true_theta": -1
                }
                for ele in new_supplier_to_parts.index
            },

            "dispatchers": {
                "1": {
                    "dispatcher_id": "1",
                    "fte_factor": 0.6,
                    "target_workload": {
                        "Mon": 720.0, "Tue": 720.0, "Wed": 720.0, "Thu": 720.0, "Fri": 720.0
                    }
                },
                "2": {
                    "dispatcher_id": "2",
                    "fte_factor": 0.6,
                    "target_workload": {
                        "Mon": 720.0, "Tue": 720.0, "Wed": 720.0, "Thu": 720.0, "Fri": 720.0
                    }
                },
                "3": {
                    "dispatcher_id": "3",
                    "fte_factor": 0.6,
                    "target_workload": {
                        "Mon": 720.0, "Tue": 720.0, "Wed": 720.0, "Thu": 720.0, "Fri": 720.0
                    }
                }
            },
            "rep_days": ["Mon", "Tue", "Wed", "Thu", "Fri"],
            "prev_assignment": None,
            "created_at": ''
        }
        input_data = InputData.model_validate(new_input_data)

        new_critical_records = []
        for _, ele in split_data.iterrows():
            for rep_day in [
                "Mon",
                "Tue",
                "Wed",
                "Thu",
                "Fri"
            ]:
                new_critical_records.append({
                    "part_id": str(ele["part_id"]),
                    "supplier_id": str(ele["supplier_id_split"]),
                    "day": rep_day,
                    "scenario_index": 0,
                    "n_deliveries": 4.0 if ele[rep_day] > 0 else 0,
                    "critical_count": ele[rep_day] / 30,
                    "workload_contribution_min": ele[rep_day]
                })

        new_scenario_set = {
            "scenario_set_id": "-1",
            "data_id": "-1",
            "config": {
                "rng_seed": 0,
                "n_scenarios": 1,
                "alpha0": 1.0,
                "beta0": 1.0,
                "handling_time": 30.0
            },
            "created_at": "-1",
            "n_parts": 75,
            "n_rep_days": 5,
            "theta_records": [],
            "critical_records": new_critical_records,
            "part_stats": [],
            "supplier_stats": [],
        }
        scenario_set = ScenarioSet.model_validate(new_scenario_set)

        ptd_config = PartToDispatcherWorkloadBalancingModelConfig(
            rng_seed=42,
            cost_deviation=1.0,
            cost_coord=1.0,
            k_max=0,
            time_limit=1*3600,
            mip_gap=0.01,
        )
        model = PartToDispatcherWorkloadBalancingModel(
            input_data=input_data,
            model_config=ptd_config,
            scenario_set=scenario_set,
            use_scenario_based_target=False,
            use_coord_in_objective=False,
            n_split_exact=True,
            use_underload_penalty=False,
            warm_start_assignment=None
        )
        model.solve()
        
        model.save(save_path)
    pass

def main():
    LIST_KAPPA_S = [5, 50, 150]
    LIST_KAPPA_P = [5, 50, 150]

    LIST_RANDOM_SEEDS = [25]

    LIST_DEFAULT_GENERAL_RANDOM_SEED = [50]

    N_SPLITS = 1
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

    aggregation_data_and_analysis(
        source_data_dir=SOURCE_DATA_DIR,
        all_experiments=all_experiments,
        n_splits=N_SPLITS,
        cv_ddof=0,
        topk=3
    )
    run_ssd_heuristic_model(
        source_data_dir=SOURCE_DATA_DIR,
        all_experiments=all_experiments
    )
    pass

if __name__ == "__main__":
    main()