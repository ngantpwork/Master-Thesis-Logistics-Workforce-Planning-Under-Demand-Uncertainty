import os
import json
import itertools
import numpy as np
import pandas as pd
from src.schemas import InputData, ScenarioSet, Dispatcher

from src.models.config.part_to_dispatcher_workload_balancing_model_config import (
    PartToDispatcherWorkloadBalancingModelConfig,
)

from src.models.part_to_dispatcher_workload_balancing_model import (
    PartToDispatcherWorkloadBalancingModel,
)


def run_model_with_average_data(source_data_dir, all_experiments, k_max):

    print(f"Total experiments: {len(all_experiments)}")
    for (i, (
        kappa_s,
        kappa_p,
        ks_seed,
        kp_seed,
        general_seed
    )) in enumerate(all_experiments):

        print(f"Running experiment {i+1}/{len(all_experiments)} with ks={kappa_s}, kp={kappa_p}, ks_seed={ks_seed}, kp_seed={kp_seed}") 
        data_dir = os.path.join(source_data_dir, f"ks_{kappa_s}_kp_{kappa_p}_ksseed_{ks_seed}_kpseed_{kp_seed}_general_seed_{general_seed}")

        input_data_path = os.path.join(data_dir, "input_data.json")
        scenario_set_path = os.path.join(data_dir, "average_scenario_set.json")

        input_data = InputData.load(input_data_path)
        scenario_set = ScenarioSet.load(scenario_set_path)

        for k_max in [0, k_max]: 
            use_scenario_based_target = False
            ptd_config = PartToDispatcherWorkloadBalancingModelConfig(
                rng_seed=42,
                cost_deviation=1.0,
                cost_coord=1.0,
                k_max=k_max,
                time_limit=1*3600,  # 1 hours time limit for this experiment
                mip_gap=0.01,
            )

            tag = f"M4_sbt_{use_scenario_based_target}_k_max_{ptd_config.k_max}"
            
            save_path = os.path.join(data_dir, "deterministic_models", f"ptd_{tag}")
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


def run_model_with_scenario_data_to_get_uptopia_and_nadir(source_data_dir, all_experiments, k_max):

    print(f"Total experiments: {len(all_experiments)}")
    for (i, (
        kappa_s,
        kappa_p,
        ks_seed,
        kp_seed,
        general_seed
    )) in enumerate(all_experiments):

        print(f"Running experiment {i+1}/{len(all_experiments)} with ks={kappa_s}, kp={kappa_p}, ks_seed={ks_seed}, kp_seed={kp_seed}") 
        data_dir = os.path.join(source_data_dir, f"ks_{kappa_s}_kp_{kappa_p}_ksseed_{ks_seed}_kpseed_{kp_seed}_general_seed_{general_seed}")

        input_data_path = os.path.join(data_dir, "input_data.json")
        scenario_set_path = os.path.join(data_dir, "scenario_set.json")

        input_data = InputData.load(input_data_path)
        scenario_set = ScenarioSet.load(scenario_set_path)

        for k_max in [0, k_max]:
            use_scenario_based_target = False
            ptd_config = PartToDispatcherWorkloadBalancingModelConfig(
                rng_seed=42,
                cost_deviation=1.0,
                cost_coord=1.0,
                k_max=k_max,
                time_limit=1*3600,  # 1 hours time limit for this experiment
                mip_gap=0.01,
            )

            tag = f"M4_sbt_{use_scenario_based_target}_k_max_{ptd_config.k_max}"

            save_path = os.path.join(data_dir, "models", f"ptd_{tag}")
            os.makedirs(save_path, exist_ok=True)

            tag = f"M4_sbt_{use_scenario_based_target}_k_max_{ptd_config.k_max}"
                        
            with open(os.path.join(data_dir, "deterministic_models", f"ptd_M4_sbt_{use_scenario_based_target}_k_max_{ptd_config.k_max}/solve_result.json")) as f1:
                warm_start_assignment = json.load(f1)["assignment"]

            model = PartToDispatcherWorkloadBalancingModel(
                input_data=input_data,
                model_config=ptd_config,
                scenario_set=scenario_set,
                use_scenario_based_target=use_scenario_based_target,
                use_coord_in_objective=False,
                n_split_exact=False,
                use_underload_penalty=False,
                warm_start_assignment=warm_start_assignment 
            )
            model.solve()
            model.save(save_path)


def run_pareto_model_with_scenario_data(source_data_dir, all_experiments, k_max):
    print(f"Total experiments: {len(all_experiments)}")
    for (i, (
        kappa_s,
        kappa_p,
        ks_seed,
        kp_seed,
        general_seed
    )) in enumerate(all_experiments):
        print(f"Running experiment {i+1}/{len(all_experiments)} with ks={kappa_s}, kp={kappa_p}, ks_seed={ks_seed}, kp_seed={kp_seed}") 
        data_dir = os.path.join(source_data_dir, f"ks_{kappa_s}_kp_{kappa_p}_ksseed_{ks_seed}_kpseed_{kp_seed}_general_seed_{general_seed}")

        input_data_path = os.path.join(data_dir, "input_data.json")
        scenario_set_path = os.path.join(data_dir, "scenario_set.json")

        input_data = InputData.load(input_data_path)
        scenario_set = ScenarioSet.load(scenario_set_path)

        use_scenario_based_target = False
        with open(os.path.join(data_dir, f"models/ptd_M4_sbt_{use_scenario_based_target}_k_max_{k_max}/solve_result.json"), 'r') as f:
            uptopia_deviation_objective = json.load(f)["obj_value"]

        with open(os.path.join(data_dir, f"models/ptd_M4_sbt_{use_scenario_based_target}_k_max_0/solve_result.json"), 'r') as f:
            nadir_deviation_objective = json.load(f)["obj_value"]

        print(f"Utopia deviation objective: {uptopia_deviation_objective}")
        print(f"Nadir deviation objective: {nadir_deviation_objective}")

        for epsilon in [0.2, 0.4, 0.6, 0.8]:
            max_deviation_objective = uptopia_deviation_objective + epsilon * (nadir_deviation_objective - uptopia_deviation_objective)
            print(f"Running experiment {i+1}/{len(all_experiments)} with ks={kappa_s}, kp={kappa_p}, ks_seed={ks_seed}, kp_seed={kp_seed}, use_scenario_based_target={use_scenario_based_target}, max_deviation_objective={max_deviation_objective}")

            ptd_config = PartToDispatcherWorkloadBalancingModelConfig(
                rng_seed=42,
                cost_deviation=1.0,
                cost_coord=1.0,
                k_max=10,
                time_limit=1*3600,  # 2 hours time limit for this experiment
                mip_gap=0.01,
                pareto_mode=True,
                max_deviation_objective=max_deviation_objective
            )

            if epsilon >= 0.7:
                with open(os.path.join(data_dir, f"models/ptd_M4_sbt_{use_scenario_based_target}_k_max_0/solve_result.json"), 'r') as f1:
                    warm_start_assignment = json.load(f1)["assignment"]
            elif epsilon <= 0.3:
                with open(os.path.join(data_dir, f"models/ptd_M4_sbt_{use_scenario_based_target}_k_max_{k_max}/solve_result.json"), 'r') as f1:
                    warm_start_assignment = json.load(f1)["assignment"]
            else:
                with open(os.path.join(data_dir, "deterministic_models", f"ptd_M4_sbt_{use_scenario_based_target}_k_max_{k_max}/solve_result.json")) as f1:
                    warm_start_assignment = json.load(f1)["assignment"]

            tag = f"M4_sbt_{use_scenario_based_target}_pareto_epsilon_{epsilon}"
            
            save_path = os.path.join(data_dir, "models", f"ptd_{tag}")
            os.makedirs(save_path, exist_ok=True)                
        
            model = PartToDispatcherWorkloadBalancingModel(
                input_data=input_data,
                model_config=ptd_config,
                scenario_set=scenario_set,
                use_scenario_based_target=False,
                use_coord_in_objective=True,
                n_split_exact=False,
                use_underload_penalty=False,
                warm_start_assignment=warm_start_assignment
            )
            model.solve()
            model.save(save_path)



def get_pareto_knee_from_experiments(source_data_dir, all_experiments, k_max):
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

        sbt = False
        model_path = os.path.join(experiment_path, "models")
        save_path = os.path.join(experiment_path, f"pareto_sbt_{sbt}")
        os.makedirs(save_path, exist_ok=True)

        pair_results = []

        with open(os.path.join(model_path, f"ptd_M4_sbt_{sbt}_k_max_0/solve_result.json"), "r") as f:
            solve_result = json.load(f)

            pair_results.append([
                solve_result["obj_value"],
                solve_result["n_split_suppliers"],
                1
            ])

        with open(os.path.join(model_path, f"ptd_M4_sbt_{sbt}_k_max_{k_max}/solve_result.json"), "r") as f:
            solve_result = json.load(f)

            pair_results.append([
                solve_result["obj_value"],
                solve_result["n_split_suppliers"],
                0
            ])

        for i in range(2, 10, 2):
            print(i)
            with open(os.path.join(model_path, f"ptd_M4_sbt_{sbt}_pareto_epsilon_{i/10}/solve_result.json"), "r") as f:
                solve_result = json.load(f)

                pair_results.append([
                    solve_result["raw_objective_components"]["deviation"],
                    solve_result["n_split_suppliers"],
                    i/10
                ])

        pair_results = pd.DataFrame(pair_results, columns=["deviation", "n_split_suppliers", "epsilon"])
        pair_results = pair_results.sort_values(by="deviation").reset_index(drop=True)

        pair_results.to_csv(os.path.join(save_path, "pareto_results.csv"))

        x_col="n_split_suppliers"
        y_col="deviation"
        # ---------- Pareto ----------
        pareto = (
            pair_results.sort_values([x_col, y_col])
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
            AP = p - A
            cross_2d = BA[0] * AP[1] - BA[1] * AP[0]
            d = abs(cross_2d) / np.linalg.norm(BA)
            distances.append(d)

        distances = np.array(distances)

        knee_idx = np.argmax(distances)

        knee = pareto.iloc[knee_idx]

        pareto.to_csv(os.path.join(save_path, "clean_pareto_results.csv"))

        with open(os.path.join(save_path, "knee.json"), "w") as f1:
            json.dump(knee.to_dict(), f1)


def run_model_with_knee_point(source_data_dir, all_experiments):
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

        input_data_path = os.path.join(experiment_path, "input_data.json")
        scenario_set_path = os.path.join(experiment_path, "scenario_set.json")

        input_data = InputData.load(input_data_path)
        scenario_set = ScenarioSet.load(scenario_set_path)


        use_scenario_based_target = False
        all_k_max = list(set(pd.read_csv(os.path.join(experiment_path,f"pareto_sbt_{use_scenario_based_target}/clean_pareto_results.csv"), index_col=0)["n_split_suppliers"].to_list()))
        for k_max in all_k_max:
            ptd_config = PartToDispatcherWorkloadBalancingModelConfig(
                rng_seed=42,
                cost_deviation=1.0,
                cost_coord=1.0,
                k_max=k_max,
                time_limit=1*3600,  # 2 hours time limit for this experiment
                mip_gap=0.01,
            )
            tag = f"M4_sbt_{use_scenario_based_target}_k_max_equal_{ptd_config.k_max}"

            save_path = os.path.join(experiment_path, "models_pareto", f"ptd_{tag}")
            os.makedirs(save_path, exist_ok=True)

            warm_start_assignment = None        

            model = PartToDispatcherWorkloadBalancingModel(
                input_data=input_data,
                model_config=ptd_config,
                scenario_set=scenario_set,
                use_scenario_based_target=use_scenario_based_target,
                use_coord_in_objective=False,
                n_split_exact=True,
                use_underload_penalty=False,
                warm_start_assignment=warm_start_assignment
            )
            model.solve()
            model.save(save_path)

def get_pareto_knee_after_refining_from_experiments(source_data_dir, all_experiments):
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

        sbt = False
        model_paths = os.listdir(os.path.join(
            experiment_path,
            "models_pareto"
        ))
        save_path = os.path.join(experiment_path, f"pareto_refine_sbt_{sbt}")
        os.makedirs(save_path, exist_ok=True)

        pair_results = []

        for model_path in model_paths:
                    
            with open(
                os.path.join(
                    experiment_path,
                    "models_pareto",
                    model_path,
                    "solve_result.json"
                ),
                "r"
            ) as f:

                solve_result = json.load(f)

            pair_results.append([
                solve_result["raw_objective_components"]["deviation"],
                solve_result["n_split_suppliers"]
            ])

        pair_results = pd.DataFrame(pair_results, columns=["deviation", "n_split_suppliers"])
        pair_results = pair_results.sort_values(by="deviation").reset_index(drop=True)

        pair_results.to_csv(os.path.join(save_path, "pareto_results.csv"))

        x_col="n_split_suppliers"
        y_col="deviation"
        # ---------- Pareto ----------
        pareto = (
            pair_results.sort_values([x_col, y_col])
            .groupby(x_col, as_index=False)
            .first()
            .sort_values(x_col)
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
            AP = p - A
            cross_2d = BA[0] * AP[1] - BA[1] * AP[0]
            d = abs(cross_2d) / np.linalg.norm(BA)
            distances.append(d)

        distances = np.array(distances)

        knee_idx = np.argmax(distances)

        knee = pareto.iloc[knee_idx]

        pareto.to_csv(os.path.join(save_path, "clean_pareto_results.csv"))

        with open(os.path.join(save_path, "knee.json"), "w") as f1:
            json.dump(knee.to_dict(), f1)



def main():
    LIST_KAPPA_S = [5, 50, 150]
    LIST_KAPPA_P = [5, 50, 150]

    LIST_RANDOM_SEEDS = [25]

    LIST_DEFAULT_GENERAL_RANDOM_SEED = [50]

    K_MAX = 9
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


    run_model_with_average_data(
        source_data_dir=SOURCE_DATA_DIR,
        all_experiments=all_experiments,
        k_max=K_MAX,
    )

    run_model_with_scenario_data_to_get_uptopia_and_nadir(
        source_data_dir=SOURCE_DATA_DIR,
        all_experiments=all_experiments,
        k_max=K_MAX,
    )

    run_pareto_model_with_scenario_data(
        source_data_dir=SOURCE_DATA_DIR, 
        all_experiments=all_experiments, 
        k_max=K_MAX
    )

    get_pareto_knee_from_experiments(
        source_data_dir=SOURCE_DATA_DIR,
        all_experiments=all_experiments,
        k_max=K_MAX,
    )

    run_model_with_knee_point(
        source_data_dir=SOURCE_DATA_DIR,
        all_experiments=all_experiments,
    )

    get_pareto_knee_after_refining_from_experiments(
        source_data_dir=SOURCE_DATA_DIR,
        all_experiments=all_experiments,
    )

    pass

if __name__ == "__main__":
    main()
