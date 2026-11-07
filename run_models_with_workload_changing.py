import os
import json
import itertools
import pandas as pd
from src.schemas import InputData, ScenarioSet, Dispatcher

from src.models.config.part_to_dispatcher_workload_balancing_model_config import (
    PartToDispatcherWorkloadBalancingModelConfig,
)

from src.models.part_to_dispatcher_workload_balancing_model import (
    PartToDispatcherWorkloadBalancingModel,
)
from src.models.selective_supplier_decomposition_heuristic import (
    compute_supplier_statistics,
    sort_suppliers_to_split,
    create_split_supplier_set 
)

def prepare_data_with_workload_changing(source_data_dir, all_experiments):
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
        workload_changing_dir = os.path.join(experiment_path, "workload_changing")
        os.makedirs(workload_changing_dir, exist_ok=True)
        for scale in [0.8, 0.9, 1.1, 1.2, 0.85, 0.95, 1.05, 1.15]:
            scale_dir = os.path.join(workload_changing_dir, str(scale))
            os.makedirs(scale_dir, exist_ok=True)

            input_data = InputData.load(os.path.join(experiment_path, "input_data.json"))
            input_data.dispatchers = {
                '1': Dispatcher(dispatcher_id='1', fte_factor=0.6 * scale, target_workload={'Mon': 720.0 * scale, 'Tue': 720.0 * scale, 'Wed': 720.0 * scale, 'Thu': 720.0 * scale, 'Fri': 720.0 * scale}),
                '2': Dispatcher(dispatcher_id='2', fte_factor=0.6 * scale, target_workload={'Mon': 720.0 * scale, 'Tue': 720.0 * scale, 'Wed': 720.0 * scale, 'Thu': 720.0 * scale, 'Fri': 720.0 * scale}),
                '3': Dispatcher(dispatcher_id='3', fte_factor=0.6 * scale, target_workload={'Mon': 720.0 * scale, 'Tue': 720.0 * scale, 'Wed': 720.0 * scale, 'Thu': 720.0 * scale, 'Fri': 720.0 * scale})
            }

            input_data.save(os.path.join(scale_dir, "input_data.json"))
            for rerun_model in [False, True]:
                rerun_model_dir = os.path.join(scale_dir, f"rerun_model_{rerun_model}")
                os.makedirs(rerun_model_dir, exist_ok=True)

def run_deterministic_model_with_workload_changing(source_data_dir, all_experiments):
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
        
        scenario_set_path = os.path.join(experiment_path, "average_scenario_set.json")
        scenario_set = ScenarioSet.load(scenario_set_path)

        workload_changing_dir = os.path.join(experiment_path, "workload_changing")
        for scale in [0.85, 0.95, 1.05, 1.15, 0.8, 0.9, 1.1, 1.2]:
            scale_dir = os.path.join(workload_changing_dir, str(scale))

            input_data = InputData.load(os.path.join(scale_dir, "input_data.json"))

            use_scenario_based_target =False
            rerun_model = True
            k_max = 0
                
            ptd_config = PartToDispatcherWorkloadBalancingModelConfig(
                rng_seed=42,
                cost_deviation=1.0,
                cost_coord=1.0,
                k_max=k_max,
                time_limit=1*3600,  # 2 hours time limit for this experiment
                mip_gap=0.01,
            )

            tag = f"M4_sbt_{use_scenario_based_target}_k_max_{ptd_config.k_max}"
            
            save_path =os.path.join(scale_dir, f"rerun_model_{rerun_model}", "deterministic_models", f"ptd_{tag}")
            os.makedirs(save_path, exist_ok=True)            
        
            model = PartToDispatcherWorkloadBalancingModel(
                input_data=input_data,
                model_config=ptd_config,
                scenario_set=scenario_set,
                use_scenario_based_target=use_scenario_based_target,
                use_coord_in_objective=False,
                n_split_exact=False,
                use_underload_penalty=False,
                warm_start_assignment=None
            )
        
            model.solve()
            model.save(save_path)


def run_stochastic_model_with_workload_changing(source_data_dir, all_experiments):
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
        
        scenario_set_path = os.path.join(experiment_path, "scenario_set.json")
        scenario_set = ScenarioSet.load(scenario_set_path)

        workload_changing_dir = os.path.join(experiment_path, "workload_changing")
        for scale in [0.8, 0.9, 1.1, 1.2, 0.85, 0.95, 1.05, 1.15]:
            scale_dir = os.path.join(workload_changing_dir, str(scale))

            input_data = InputData.load(os.path.join(scale_dir, "input_data.json"))
            
            use_scenario_based_target =False

            rerun_model = True
            rerun_model_dir = os.path.join(scale_dir, f"rerun_model_{rerun_model}")

            scenario_set_path = os.path.join(experiment_path, "scenario_set.json")
            scenario_set = ScenarioSet.load(scenario_set_path)

            with open(os.path.join(experiment_path, "pareto_refine_sbt_False/knee.json")) as f1:
                knee_k_max = json.load(f1)["n_split_suppliers"]
            all_k_max = [0, knee_k_max]

            for k_max in all_k_max:
                ptd_config = PartToDispatcherWorkloadBalancingModelConfig(
                    rng_seed=42,
                    cost_deviation=1.0,
                    cost_coord=1.0,
                    k_max=k_max,
                    time_limit=1*3600, 
                    mip_gap=0.01,
                )

                tag = f"M4_sbt_{use_scenario_based_target}_k_max_equal_{ptd_config.k_max}"

                save_path = os.path.join(rerun_model_dir, "models_pareto", f"ptd_{tag}")
                os.makedirs(save_path, exist_ok=True)

                model = PartToDispatcherWorkloadBalancingModel(
                    input_data=input_data,
                    model_config=ptd_config,
                    scenario_set=scenario_set,
                    use_scenario_based_target=use_scenario_based_target,
                    use_coord_in_objective=False,
                    n_split_exact=True,
                    use_underload_penalty=False,
                    warm_start_assignment=None
                )
                model.solve()
                model.save(save_path)

def run_ssd_heuristic_model_with_workload_changing(source_data_dir, all_experiments):

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
        for scale in [0.85, 0.95, 1.05, 1.15, 0.8, 0.9, 1.1, 1.2]:

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
                    "fte_factor": 0.6 * scale,
                    "target_workload": {'Mon': 720.0 * scale, 'Tue': 720.0 * scale, 'Wed': 720.0 * scale, 
                                        'Thu': 720.0 * scale, 'Fri': 720.0 * scale}
                    },
                    "2": {
                    "dispatcher_id": "2",
                    "fte_factor": 0.6 * scale,
                    "target_workload": {'Mon': 720.0 * scale, 'Tue': 720.0 * scale, 'Wed': 720.0 * scale, 
                                        'Thu': 720.0 * scale, 'Fri': 720.0 * scale}
                    },
                    "3": {
                    "dispatcher_id": "3",
                    "fte_factor": 0.6 * scale,
                    "target_workload": {'Mon': 720.0 * scale, 'Tue': 720.0 * scale, 'Wed': 720.0 * scale, 
                                        'Thu': 720.0 * scale, 'Fri': 720.0 * scale}
                    }
                },
                "rep_days": [
                    "Mon",
                    "Tue",
                    "Wed",
                    "Thu",
                    "Fri"
                ],
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
                time_limit=1*3600,  # 2 hours time limit for this experiment
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

            save_path = os.path.join(experiment_path, "workload_changing", str(scale), "rerun_model_True", "selective_supplier_decomposition_heuristic")
            os.makedirs(save_path, exist_ok=True)
            
            model.save(save_path)

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

    # Generate multiple data with workload changing
    prepare_data_with_workload_changing(
        source_data_dir=SOURCE_DATA_DIR,
        all_experiments=all_experiments
    )

    # Run deterministic model with workload changing
    run_deterministic_model_with_workload_changing(
        source_data_dir=SOURCE_DATA_DIR,
        all_experiments=all_experiments
    )

    # Run stochastic model with workload changing
    run_stochastic_model_with_workload_changing(
        source_data_dir=SOURCE_DATA_DIR,
        all_experiments=all_experiments
    )

    # Run SSD-Heuristic model with workload changing
    run_ssd_heuristic_model_with_workload_changing(
        source_data_dir=SOURCE_DATA_DIR,
        all_experiments=all_experiments
    )

if __name__ == "__main__":
    main()