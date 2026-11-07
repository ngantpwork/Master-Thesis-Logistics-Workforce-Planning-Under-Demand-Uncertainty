import os
import json
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from openpyxl.styles import Alignment, Font
from openpyxl.drawing.image import Image as ExcelImage
from src.schemas import ScenarioSet, InputData
from src.evaluation.evaluate_assignment import *

def evaluate_multi_models_function(
    scenario_set,
    dispatchers,
    rep_days,
    deterministic_std_solve_result_path,
    stochastic_std_solve_result_path,
    ssh_heuristic_solve_result_path,
    stochastic_hybrid_epsilon_knee_point_solve_result_path
):
    with open(deterministic_std_solve_result_path) as f1:
        deterministic_std_solve_result = json.load(f1)
        deterministic_std_assignment = deterministic_std_solve_result["assignment"]
        deterministic_std_splits = int(deterministic_std_solve_result["n_split_suppliers"])
        deterministic_std_runtime = deterministic_std_solve_result["solve_time"]
        deterministic_std_mip_gap = deterministic_std_solve_result["mip_gap"]

    deterministic_std_result = evaluate_assignment(
        assignment=deterministic_std_assignment,
        scenario_set=scenario_set,
        dispatchers_active=dispatchers,
        rep_days=rep_days,
        use_underload_penalty=False,
        use_scenario_based_target=False
    )

    with open(stochastic_std_solve_result_path) as f1:
        stochastic_std_solve_result = json.load(f1)
        stochastic_std_assignment = stochastic_std_solve_result["assignment"]
        stochastic_std_splits = int(stochastic_std_solve_result["n_split_suppliers"])
        stochastic_std_runtime = stochastic_std_solve_result["solve_time"]
        stochastic_std_mip_gap = stochastic_std_solve_result["mip_gap"]

    stochastic_std_result = evaluate_assignment(
        assignment=stochastic_std_assignment,
        scenario_set=scenario_set,
        dispatchers_active=dispatchers,
        rep_days=rep_days,
        use_underload_penalty=False,
        use_scenario_based_target=False
    )

    with open(ssh_heuristic_solve_result_path) as f1:
        ssd_heuristic_solve_result = json.load(f1)
        ssd_heuristic_assignment = ssd_heuristic_solve_result["assignment"]
        ssd_heuristic_splits = int(ssd_heuristic_solve_result["n_split_suppliers"])
        ssd_heuristic_runtime = ssd_heuristic_solve_result["solve_time"]
        ssd_heuristic_mip_gap = ssd_heuristic_solve_result["mip_gap"]


    ssd_heuristic_result = evaluate_assignment(
        assignment=ssd_heuristic_assignment,
        scenario_set=scenario_set,
        dispatchers_active=dispatchers,
        rep_days=rep_days,
        use_underload_penalty=False,
        use_scenario_based_target=False
    )

        
    with open(stochastic_hybrid_epsilon_knee_point_solve_result_path) as f1:
        stochastic_hybrid_epsilon_knee_point_solve_result = json.load(f1)
        stochastic_hybrid_epsilon_knee_point_assignment = stochastic_hybrid_epsilon_knee_point_solve_result["assignment"]
        stochastic_hybrid_epsilon_knee_point_splits = int(stochastic_hybrid_epsilon_knee_point_solve_result["n_split_suppliers"])
        stochastic_hybrid_epsilon_knee_point_runtime = stochastic_hybrid_epsilon_knee_point_solve_result["solve_time"]
        stochastic_hybrid_epsilon_knee_point_mip_gap = stochastic_hybrid_epsilon_knee_point_solve_result["mip_gap"]


    stochastic_hybrid_epsilon_knee_point_result = evaluate_assignment(
        assignment=stochastic_hybrid_epsilon_knee_point_assignment,
        scenario_set=scenario_set,
        dispatchers_active=dispatchers,
        rep_days=rep_days,
        use_underload_penalty=False,
        use_scenario_based_target=False
    )

    metrics = pd.DataFrame({
        "Metric": [
            "Mean workload deviation",
            "Median deviation",
            "95th percentile deviation",
            "CVaR 95% deviation",
            "Maximum / Expected peak overload",
            "Number of splits",
            "Runtime",
            "MIP"
        ],
        "D-STD": [
            deterministic_std_result["mean_deviation"],
            deterministic_std_result["median_deviation"],
            deterministic_std_result["p95_deviation"],
            deterministic_std_result["cvar95_deviation"],
            deterministic_std_result["maximum_overload"],
            deterministic_std_splits,
            deterministic_std_runtime,
            deterministic_std_mip_gap
        ],
        "S-STD (Baseline)": [
            stochastic_std_result["mean_deviation"],
            stochastic_std_result["median_deviation"],
            stochastic_std_result["p95_deviation"],
            stochastic_std_result["cvar95_deviation"],
            stochastic_std_result["maximum_overload"],
            stochastic_std_splits,
            stochastic_std_runtime,
            stochastic_std_mip_gap
        ],
        "SSD-H": [
            ssd_heuristic_result["mean_deviation"],
            ssd_heuristic_result["median_deviation"],
            ssd_heuristic_result["p95_deviation"],
            ssd_heuristic_result["cvar95_deviation"],
            ssd_heuristic_result["maximum_overload"],
            ssd_heuristic_splits,
            ssd_heuristic_runtime,
            ssd_heuristic_mip_gap
        ],
        "S-Hybrid at knee point": [
            stochastic_hybrid_epsilon_knee_point_result["mean_deviation"],
            stochastic_hybrid_epsilon_knee_point_result["median_deviation"],
            stochastic_hybrid_epsilon_knee_point_result["p95_deviation"],
            stochastic_hybrid_epsilon_knee_point_result["cvar95_deviation"],
            stochastic_hybrid_epsilon_knee_point_result["maximum_overload"],
            stochastic_hybrid_epsilon_knee_point_splits,
            stochastic_hybrid_epsilon_knee_point_runtime,
            stochastic_hybrid_epsilon_knee_point_mip_gap
        ],
    })

    metrics["Improvement to baseline | D-STD"] = (metrics["S-STD (Baseline)"] - metrics["D-STD"]) / metrics["S-STD (Baseline)"] * 100
    metrics["Improvement to baseline | SSD-H"] = (metrics["S-STD (Baseline)"] - metrics["SSD-H"]) / metrics["S-STD (Baseline)"] * 100
    metrics["Improvement to baseline | S-Hybrid at knee point"] = (metrics["S-STD (Baseline)"] - metrics["S-Hybrid at knee point"]) / metrics["S-STD (Baseline)"] * 100
    metrics.loc[[5,6,7], ["Improvement to baseline | D-STD", 
                          "Improvement to baseline | SSD-H",
                          "Improvement to baseline | S-Hybrid at knee point"]] = None
    
    metrics["95% CI | D-STD"] = [
        format_ci95(
            improvement_ci95(
                baseline=stochastic_std_result["scenario_summary"]["total_deviation"],
                method=deterministic_std_result["scenario_summary"]["total_deviation"],
                metric_func=metric_func,
                n_bootstrap=10000,
                confidence=0.95,
                seed=42
            )
        )
        for metric_func in [np.mean, np.median, lambda x: np.percentile(x, 95), cvar95, np.max]
    ] + [None, None, None]

    metrics["95% CI | SSD-H"] = [
        format_ci95(
            improvement_ci95(
                baseline=stochastic_std_result["scenario_summary"]["total_deviation"],
                method=ssd_heuristic_result["scenario_summary"]["total_deviation"],
                metric_func=metric_func,
                n_bootstrap=10000,
                confidence=0.95,
                seed=42
            )
        )
        for metric_func in [np.mean, np.median, lambda x: np.percentile(x, 95), cvar95, np.max]
    ] + [None, None, None]

    metrics["95% CI | S-Hybrid at knee point"] = [
        format_ci95(
            improvement_ci95(
                baseline=stochastic_std_result["scenario_summary"]["total_deviation"],
                method=stochastic_hybrid_epsilon_knee_point_result["scenario_summary"]["total_deviation"],
                metric_func=metric_func,
                n_bootstrap=10000,
                confidence=0.95,
                seed=42
            )
        )
        for metric_func in [np.mean, np.median, lambda x: np.percentile(x, 95), cvar95, np.max]
    ] + [None, None, None]

    return metrics

def evaluate_multi_models(source_data_dir, all_experiments, save_folder_name="evaluation"):

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

        scenario_set = ScenarioSet.load(os.path.join(experiment_path, "groundtruth.json"))

        input_data = InputData.load(os.path.join(experiment_path, "input_data.json"))
        dispatchers = input_data.dispatchers
        rep_days = input_data.rep_days

        deterministic_std_solve_result_path = os.path.join(experiment_path, "deterministic_models/ptd_M4_sbt_False_k_max_0/solve_result.json")
        stochastic_std_solve_result_path = os.path.join(experiment_path, "models_pareto/ptd_M4_sbt_False_k_max_equal_0/solve_result.json")
        ssh_heuristic_solve_result_path = os.path.join(experiment_path, f"selective_supplier_decomposition_heuristic/solve_result.json")
        
        with open(os.path.join(experiment_path, "pareto_refine_sbt_False/knee.json")) as f1:
            n_split_suppliers_knee_point = int(json.load(f1)["n_split_suppliers"])
        stochastic_hybrid_epsilon_knee_point_solve_result_path = os.path.join(experiment_path, f"models_pareto/ptd_M4_sbt_False_k_max_equal_{n_split_suppliers_knee_point}/solve_result.json")
        
        metrics = evaluate_multi_models_function(
            scenario_set=scenario_set,
            dispatchers=dispatchers,
            rep_days=rep_days,
            deterministic_std_solve_result_path=deterministic_std_solve_result_path,
            stochastic_std_solve_result_path=stochastic_std_solve_result_path,
            ssh_heuristic_solve_result_path=ssh_heuristic_solve_result_path,
            stochastic_hybrid_epsilon_knee_point_solve_result_path=stochastic_hybrid_epsilon_knee_point_solve_result_path
        )

        os.makedirs(os.path.join(experiment_path, save_folder_name), exist_ok=True)
        metrics.to_csv(os.path.join(experiment_path, save_folder_name, f"ks_{kappa_s}_kp_{kappa_p}_compare_report.csv"))
    

def evaluate_multi_models_with_workload_changing_and_not_rerun_model(source_data_dir, all_experiments, save_folder_name="evaluation"):

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
        for scale in [0.8, 0.9, 1.1, 1.2, 0.85, 0.95, 1.05, 1.15]:
            scale_dir = os.path.join(workload_changing_dir, str(scale))
            scenario_set = ScenarioSet.load(os.path.join(experiment_path, "groundtruth.json"))

            input_data = InputData.load(os.path.join(scale_dir, "input_data.json"))
            rerun_model = False
            rerun_model_dir = os.path.join(scale_dir, f"rerun_model_{rerun_model}")
            dispatchers = input_data.dispatchers
            rep_days = input_data.rep_days

            deterministic_std_solve_result_path = os.path.join(experiment_path, "deterministic_models/ptd_M4_sbt_False_k_max_0/solve_result.json")
            stochastic_std_solve_result_path = os.path.join(experiment_path, "models_pareto/ptd_M4_sbt_False_k_max_equal_0/solve_result.json")
            ssh_heuristic_solve_result_path = os.path.join(experiment_path, f"selective_supplier_decomposition_heuristic/solve_result.json")
            
            with open(os.path.join(experiment_path, "pareto_refine_sbt_False/knee.json")) as f1:
                n_split_suppliers_knee_point = int(json.load(f1)["n_split_suppliers"])
            stochastic_hybrid_epsilon_knee_point_solve_result_path = os.path.join(experiment_path, f"models_pareto/ptd_M4_sbt_False_k_max_equal_{n_split_suppliers_knee_point}/solve_result.json")
            
            metrics = evaluate_multi_models_function(
                scenario_set=scenario_set,
                dispatchers=dispatchers,
                rep_days=rep_days,
                deterministic_std_solve_result_path=deterministic_std_solve_result_path,
                stochastic_std_solve_result_path=stochastic_std_solve_result_path,
                ssh_heuristic_solve_result_path=ssh_heuristic_solve_result_path,
                stochastic_hybrid_epsilon_knee_point_solve_result_path=stochastic_hybrid_epsilon_knee_point_solve_result_path
            )

            os.makedirs(os.path.join(rerun_model_dir, save_folder_name), exist_ok=True)
            metrics.to_csv(os.path.join(rerun_model_dir, save_folder_name, f"ks_{kappa_s}_kp_{kappa_p}_scale_{scale}_compare_report.csv"))


def evaluate_multi_models_with_workload_changing_and_rerun_model(source_data_dir, all_experiments, save_folder_name="evaluation"):

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
        for scale in [0.8, 0.9, 1.1, 1.2, 0.85, 0.95, 1.05, 1.15]:
            scale_dir = os.path.join(workload_changing_dir, str(scale))
            scenario_set = ScenarioSet.load(os.path.join(experiment_path, "groundtruth.json"))

            input_data = InputData.load(os.path.join(scale_dir, "input_data.json"))
            rerun_model = True
            rerun_model_dir = os.path.join(scale_dir, f"rerun_model_{rerun_model}")
            dispatchers = input_data.dispatchers
            rep_days = input_data.rep_days

            deterministic_std_solve_result_path = os.path.join(rerun_model_dir, "deterministic_models/ptd_M4_sbt_False_k_max_0/solve_result.json")
            stochastic_std_solve_result_path = os.path.join(rerun_model_dir, "models_pareto/ptd_M4_sbt_False_k_max_equal_0/solve_result.json")
            ssh_heuristic_solve_result_path = os.path.join(rerun_model_dir, f"selective_supplier_decomposition_heuristic/solve_result.json")
            
            with open(os.path.join(experiment_path, "pareto_refine_sbt_False/knee.json")) as f1:
                n_split_suppliers_knee_point = int(json.load(f1)["n_split_suppliers"])
            stochastic_hybrid_epsilon_knee_point_solve_result_path = os.path.join(experiment_path, f"models_pareto/ptd_M4_sbt_False_k_max_equal_{n_split_suppliers_knee_point}/solve_result.json")
            
            metrics = evaluate_multi_models_function(
                scenario_set=scenario_set,
                dispatchers=dispatchers,
                rep_days=rep_days,
                deterministic_std_solve_result_path=deterministic_std_solve_result_path,
                stochastic_std_solve_result_path=stochastic_std_solve_result_path,
                ssh_heuristic_solve_result_path=ssh_heuristic_solve_result_path,
                stochastic_hybrid_epsilon_knee_point_solve_result_path=stochastic_hybrid_epsilon_knee_point_solve_result_path
            )

            os.makedirs(os.path.join(rerun_model_dir, save_folder_name), exist_ok=True)
            metrics.to_csv(os.path.join(rerun_model_dir, save_folder_name, f"ks_{kappa_s}_kp_{kappa_p}_scale_{scale}_compare_report.csv"))

def export_to_report(source_data_dir, all_experiments, evaluation_folder_name="evaluation", save_folder_name="report"):

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
        src_path = os.path.join(experiment_path, evaluation_folder_name, f"ks_{kappa_s}_kp_{kappa_p}_compare_report.csv")
        dest_path = os.path.join(source_data_dir, save_folder_name, f"ks_{kappa_s}_kp_{kappa_p}_compare_report.csv")
        os.system(f"cp {src_path} {dest_path}")

        workload_changing_dir = os.path.join(experiment_path, "workload_changing")
        for scale in [0.8, 0.9, 1.1, 1.2, 0.85, 0.95, 1.05, 1.15]:
            for rerun_model in [False, True]:
                scale_dir = os.path.join(workload_changing_dir, str(scale))
                rerun_model_dir = os.path.join(scale_dir, f"rerun_model_{rerun_model}")
                
                src_path = os.path.join(rerun_model_dir, evaluation_folder_name, f"ks_{kappa_s}_kp_{kappa_p}_scale_{scale}_compare_report.csv")
                dest_path = os.path.join(source_data_dir, save_folder_name, f"ks_{kappa_s}_kp_{kappa_p}_scale_{scale}_rerun_{rerun_model}_compare_report.csv")
                os.system(f"cp {src_path} {dest_path}")

def aggregation_all_report(source_data_dir, all_experiments, report_folder_name="report"):
    all_results = []
    for (ix, (
        kappa_s,
        kappa_p,
        _, _, _
    )) in enumerate(all_experiments):
        for scale in [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]:
            if scale == 1.0:
                file_path = os.path.join(source_data_dir, report_folder_name, f"ks_{kappa_s}_kp_{kappa_p}_compare_report.csv")
                df = pd.read_csv(file_path, index_col=1)
                all_results.append({
                    "kappa_s": kappa_s,
                    "kappa_p": kappa_p,
                    "scale": scale,
                    "metric": "Mean workload deviation",
                    "D-STD": df.loc["Mean workload deviation", "D-STD"],
                    "S-STD (Baseline)": df.loc["Mean workload deviation", "S-STD (Baseline)"],
                    "SSD-H": df.loc["Mean workload deviation", "SSD-H"],
                    "S-Hybrid at knee point": df.loc["Mean workload deviation", "S-Hybrid at knee point"],
                })
                all_results.append({
                    "kappa_s": kappa_s,
                    "kappa_p": kappa_p,
                    "scale": scale,
                    "metric": "CVaR 95% deviation",
                    "D-STD": df.loc["CVaR 95% deviation", "D-STD"],
                    "S-STD (Baseline)": df.loc["CVaR 95% deviation", "S-STD (Baseline)"],
                    "SSD-H": df.loc["CVaR 95% deviation", "SSD-H"],
                    "S-Hybrid at knee point": df.loc["CVaR 95% deviation", "S-Hybrid at knee point"],
                })
            else:
                for rerun in [False, True]:
                    file_path = os.path.join(source_data_dir, report_folder_name, f"ks_{kappa_s}_kp_{kappa_p}_scale_{scale}_rerun_{rerun}_compare_report.csv")
                    df = pd.read_csv(file_path, index_col=1)
                    all_results.append({
                        "kappa_s": kappa_s,
                        "kappa_p": kappa_p,
                        "scale": scale,
                        "rerun": rerun,
                        "metric": "Mean workload deviation",
                        "D-STD": df.loc["Mean workload deviation", "D-STD"],
                        "S-STD (Baseline)": df.loc["Mean workload deviation", "S-STD (Baseline)"],
                        "SSD-H": df.loc["Mean workload deviation", "SSD-H"],
                        "S-Hybrid at knee point": df.loc["Mean workload deviation", "S-Hybrid at knee point"],
                    })
                    all_results.append({
                        "kappa_s": kappa_s,
                        "kappa_p": kappa_p,
                        "scale": scale,
                        "rerun": rerun,
                        "metric": "CVaR 95% deviation",
                        "D-STD": df.loc["CVaR 95% deviation", "D-STD"],
                        "S-STD (Baseline)": df.loc["CVaR 95% deviation", "S-STD (Baseline)"],
                        "SSD-H": df.loc["CVaR 95% deviation", "SSD-H"],
                        "S-Hybrid at knee point": df.loc["CVaR 95% deviation", "S-Hybrid at knee point"],
                    })
    all_results_df = pd.DataFrame(all_results)
    all_results_df["delta_workforce_cap"] = all_results_df["scale"].apply(lambda x: (x - 1.0) * 300 * 1.8 * 20)

    for ix, row in all_results_df.iterrows():
        all_results_df.loc[ix, "delta_D-STD"] = row["D-STD"] - all_results_df.loc[(all_results_df["kappa_s"] == row["kappa_s"]) & (all_results_df["kappa_p"] == row["kappa_p"]) & (all_results_df["scale"] == 1.0) & (all_results_df["metric"] == row["metric"]), "D-STD"].values[0]
        all_results_df.loc[ix, "delta_S-STD (Baseline)"] = row["S-STD (Baseline)"] - all_results_df.loc[(all_results_df["kappa_s"] == row["kappa_s"]) & (all_results_df["kappa_p"] == row["kappa_p"]) & (all_results_df["scale"] == 1.0) & (all_results_df["metric"] == row["metric"]), "S-STD (Baseline)"].values[0]
        all_results_df.loc[ix, "delta_SSD-H"] = row["SSD-H"] - all_results_df.loc[(all_results_df["kappa_s"] == row["kappa_s"]) & (all_results_df["kappa_p"] == row["kappa_p"]) & (all_results_df["scale"] == 1.0) & (all_results_df["metric"] == row["metric"]), "SSD-H"].values[0]
        all_results_df.loc[ix, "delta_S-Hybrid at knee point"] = row["S-Hybrid at knee point"] - all_results_df.loc[(all_results_df["kappa_s"] == row["kappa_s"]) & (all_results_df["kappa_p"] == row["kappa_p"]) & (all_results_df["scale"] == 1.0) & (all_results_df["metric"] == row["metric"]), "S-Hybrid at knee point"].values[0]

    all_results_df.to_csv(os.path.join(source_data_dir, report_folder_name, "all_results_summary.csv"), index=False)


def export_to_excel_and_plot(source_data_dir, all_experiments, report_folder_name="report"):
    all_results_df = pd.read_csv(os.path.join(source_data_dir, report_folder_name, "all_results_summary.csv"))
    
    plot_dir = os.path.join(source_data_dir, report_folder_name, "plots")
    os.makedirs(plot_dir, exist_ok=True)


    for rerun in [False, True]:

        output_file = os.path.join(
            source_data_dir,
            report_folder_name,
            f"workforce_changing_all_rerun_{rerun}.xlsx",
        )

        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:

            for _, (kappa_s, kappa_p, _, _, _) in enumerate(all_experiments):

                df_excel = all_results_df[
                    (all_results_df["kappa_s"] == kappa_s)
                    & (all_results_df["kappa_p"] == kappa_p)
                    & (
                        (all_results_df["rerun"] == rerun)
                        | (all_results_df["scale"] == 1.0)
                    )
                ]
                df_excel = df_excel.drop(columns=["kappa_s", "kappa_p", "rerun"])

                metric_order = {
                    "Mean workload deviation": 0,
                    "CVaR 95% deviation": 1,
                }

                df_excel["_metric_order"] = df_excel["metric"].map(lambda x: metric_order.get(x, 2))

                df_excel = (
                    df_excel
                    .sort_values(
                        ["_metric_order", "scale",],
                        kind="stable",
                    )
                    .drop(columns="_metric_order")
                )
                df_excel = df_excel.rename(
                    columns={
                        "scale": "FTE multiplier",
                    }
                ).reset_index(drop=True)

                # ========================================================
                # Save Excel
                # ========================================================

                sheet_name = f"ks_{kappa_s}_kp_{kappa_p}"

                df_excel.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    index=False,
                )

                # ========================================================
                # Format Excel
                # ========================================================

                ws = writer.book[sheet_name]

                # Header
                for cell in ws[1]:
                    cell.font = Font(bold=True)
                    cell.alignment = Alignment(
                        horizontal="center",
                        vertical="center",
                        wrap_text=True,
                    )

                ws.row_dimensions[1].height = 30

                # ========================================================
                # Column width
                # ========================================================

                widths = {
                    "A": 15,   
                    "B": 28,   
                    "C": 14,   
                    "D": 20,  
                    "E": 24,
                    "F": 24,
                    "G": 18,
                    "H": 24,
                    "I": 24,
                    "J": 24,
                    "K": 24,
                }

                for col, width in widths.items():
                    ws.column_dimensions[col].width = width

                # ========================================================
                # Alignment
                # ========================================================

                for row in ws.iter_rows(min_row=2):
                    for cell in row:
                        cell.alignment = Alignment(
                            horizontal="center",
                            vertical="center",
                        )

                # ========================================================
                # Number format
                # ========================================================

                for row in ws.iter_rows(min_row=2):
                    for cell in row:
                        if isinstance(cell.value, (int, float)):
                            cell.number_format = '#,##0.000'

                # ========================================================
                # Freeze header
                # ========================================================

                ws.freeze_panes = "A2"

                # Auto filter
                ws.auto_filter.ref = ws.dimensions

                # ========================================================
                # PLOT
                # ========================================================

                metrics = df_excel["metric"].dropna().unique()

                for metric in metrics:

                    metric_df = df_excel[
                        df_excel["metric"] == metric
                    ].copy()

                    metric_df = metric_df.sort_values("FTE multiplier")

                    fig, ax = plt.subplots(figsize=(9, 5))

                    # ----------------------------------------------------
                    # D-STD
                    # ----------------------------------------------------

                    ax.plot(
                        metric_df["delta_workforce_cap"],
                        metric_df["D-STD"],
                        # marker="o",
                        linewidth=1.5,
                        label="D-STD",
                    )

                    # ----------------------------------------------------
                    # S-STD Baseline
                    # ----------------------------------------------------

                    ax.plot(
                        metric_df["delta_workforce_cap"],
                        metric_df["S-STD (Baseline)"],
                        # marker="o",
                        linewidth=1.5,
                        label="S-STD (Baseline)",
                    )

                    # ----------------------------------------------------
                    # SSD-H
                    # ----------------------------------------------------

                    ax.plot(
                        metric_df["delta_workforce_cap"],
                        metric_df["SSD-H"],
                        # marker="o",
                        linewidth=1.5,
                        label="SSD-H",
                    )

                    # ----------------------------------------------------
                    # S-Hybrid at knee point
                    # ----------------------------------------------------

                    ax.plot(
                        metric_df["delta_workforce_cap"],
                        metric_df["S-Hybrid at knee point"],
                        # marker="o",
                        linewidth=1.5,
                        label="S-Hybrid at knee point",
                    )

                    # ====================================================
                    # Labels
                    # ====================================================

                    ax.set_xlabel(
                        "Delta Workforce Capacity",
                        fontsize=11,
                    )

                    ax.set_ylabel(
                        metric,
                        fontsize=11,
                    )

                    ax.set_title(
                        f"{metric} - "
                        f"$\\kappa_s$={kappa_s}, "
                        f"$\\kappa_p$={kappa_p}",
                        fontsize=12,
                    )

                    ax.legend(
                        loc="best",
                        frameon=True,
                    )

                    ax.grid(
                        axis="both",
                        alpha=0.3,
                    )
                    # Better x-axis
                    ax.set_xticks(
                        sorted(metric_df["delta_workforce_cap"].unique())
                    )

                    plt.tight_layout()

                    # ====================================================
                    # Save plot
                    # ====================================================

                    metric_filename = (
                        metric
                        .lower()
                        .replace(" ", "_")
                        .replace("%", "pct")
                    )

                    plot_filename = (
                        f"rerun_{rerun}_"
                        f"ks_{kappa_s}_kp_{kappa_p}_"
                        f"{metric_filename}.png"
                    )

                    plot_path = os.path.join(
                        plot_dir,
                        plot_filename,
                    )

                    fig.savefig(
                        plot_path,
                        dpi=200,
                        bbox_inches="tight",
                    )

                    plt.close(fig)

                    # ====================================================
                    # Insert plot into Excel
                    # ====================================================

                    img = ExcelImage(plot_path)

                    # Resize image
                    img.width = 720
                    img.height = 400

                    # ----------------------------------------------------
                    # Find insertion position
                    # ----------------------------------------------------
                    #
                    # Put the first chart below the table.
                    # Subsequent charts are placed further down.
                    #

                    chart_row = len(df_excel) + 4

                    metric_index = list(metrics).index(metric)

                    # Approximately 23 rows per chart
                    chart_row += metric_index * 23

                    img.anchor = f"A{chart_row}"

                    ws.add_image(img)

        print(
            f"Saved: {output_file}"
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
    evaluate_multi_models(
        source_data_dir=SOURCE_DATA_DIR, 
        all_experiments=all_experiments, 
        save_folder_name="evaluation"
    )
    evaluate_multi_models_with_workload_changing_and_not_rerun_model(
        source_data_dir=SOURCE_DATA_DIR,
        all_experiments=all_experiments,
        save_folder_name="evaluation"
    )
    evaluate_multi_models_with_workload_changing_and_rerun_model(
        source_data_dir=SOURCE_DATA_DIR,
        all_experiments=all_experiments,
        save_folder_name="evaluation"
    )

    os.makedirs(os.path.join(SOURCE_DATA_DIR, "report"), exist_ok=True)

    export_to_report(
        source_data_dir=SOURCE_DATA_DIR, 
        all_experiments=all_experiments, 
        evaluation_folder_name="evaluation", 
        save_folder_name="report"
    )

    aggregation_all_report(
        source_data_dir=SOURCE_DATA_DIR, 
        all_experiments=all_experiments, 
        report_folder_name="report"
    )

    export_to_excel_and_plot(
        source_data_dir=SOURCE_DATA_DIR,
        all_experiments=all_experiments,
        report_folder_name="report"
    )

    pass

if __name__ == "__main__":
    main()