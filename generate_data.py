import os
import json
import math
import random
import itertools
from tqdm import tqdm
from uuid import uuid4
from datetime import date, datetime, timezone
import numpy as np
from src.simulation import (
    WorldSettings,
    sample_world_config_constrained,
    check_delivery_day_balance,
    initialize_world,
    step_world_forward,
    create_scenarios_for_current_period,
    world_to_input_data,
    tp3_scenarios_to_scenario_set,
    save_world,
    save_tp3_scenarios,
    load_world,
    tp3_average_scenario_to_scenario_set,
    world_to_input_data,
    create_average_scenario_for_current_period,
    save_tp3_scenarios,
    create_groundtruth_scenarios_for_current_period,
    tp3_scenarios_chunk_to_scenario_set
)
from src.schemas import InputData, ScenarioSet, Dispatcher


def generate_world_config():

    WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]

    def default_delivery_day_pattern_draw(rng: random.Random) -> list[str]:
        n_days = rng.randint(1, 5)
        sampled_days = rng.sample(WEEKDAYS, k=n_days)
        weekday_order = {day: idx for idx, day in enumerate(WEEKDAYS)}
        sampled_days.sort(key=lambda day: weekday_order[day])
        return sampled_days

    def default_supplier_part_count_draw(rng: random.Random) -> int:
        return rng.randint(3, 12)

    sampler_settings = WorldSettings(
        n_simulation_periods=5,
        current_period=0,
        simulation_start_date=date(2025, 1, 6),
        weeks_per_period=4,
        n_init_suppliers=9,
        n_dispatchers=3,
        shortage_concentration_rho=0.0,
        non_stationary_theta_share=0.3,
        # Using Hierarchical Beta-Bernoulli
        use_hierarchical_theta=True,
        m_global=0.4,
        kappa_s=30.0,
        kappa_p=20.0,
        # Use pre
        world_config=None,
        delivery_day_pattern_sampler=default_delivery_day_pattern_draw,
        supplier_part_count_sampler=default_supplier_part_count_draw,
        target_workload_dispatcher_fulltime_per_day=1200,
        target_workload_dispatcher_parttime_per_day=960,
    )

    dict_suppliers_for_9_configs = {
        1: 6,
        2: 6,
        3: 7,
        4: 7,
        5: 8,
        6: 9,
        7: 10,
        8: 11,
        9: 11, 
    }

    part_counts = list(dict_suppliers_for_9_configs.values())

    world_config = sample_world_config_constrained(
        settings=sampler_settings,
        target_n_parts=75,
        target_total_deliveries=75 * 3,
        min_part_count_per_supplier=3,
        max_part_count_per_supplier=10,
        min_delivery_days_per_part=1,
        max_delivery_days_per_part=5,
        max_daily_imbalance_pct=0.01,
        random_seed=20,
        part_counts=part_counts
    )

    totals = world_config["_debug_day_totals"]
    mean = sum(totals.values()) / 5
    print(f"Day totals : {totals}")
    print(f"Mean       : {mean:.1f}")
    print(f"Imbalance  : {(max(totals.values()) - min(totals.values())) / mean:.3f}")
    print(check_delivery_day_balance(world_config['parts']))

    world_config['dispatchers'] = {
        1: {'employment_type': 'parttime'},
        2: {'employment_type': 'parttime'},
        3: {'employment_type': 'parttime'},
    }

    return world_config

def generate_multiple_data_simulation(world_config, save_dir):

    def default_delivery_call_off_rate_draw(rng: random.Random) -> float:
        return 1.0

    world_config.pop("_debug_day_totals", None)

    LIST_KAPPA_S = [5, 50, 150]
    LIST_KAPPA_P = [5, 50, 150]

    LIST_RANDOM_SEEDS = [25]

    DEFAULT_GENERAL_RANDOM_SEED = 50

    NUMBER_OF_SIMULATION_PERIODS = 16

    M_GLOBAL = 0.4

    CRITICAL_HANDLING_TIME = 30  # minutes

    NUMBER_OF_SAMPLES_FOR_BALANCE_CHECK = 5000
    NUMBER_OF_SCENARIOS = 500
    ACCEPTABLE_RANGE = (360 - 360 * 0.0, 360 + 360 * 0.0)  # exactly 360

    all_experiments = list(
        itertools.product(
            LIST_KAPPA_S,
            LIST_KAPPA_P,
            LIST_RANDOM_SEEDS,
            LIST_RANDOM_SEEDS,
        )
    )
    print(f"Total experiments: {len(all_experiments)}")

    for (ix, (
        kappa_s,
        kappa_p,
        ks_seed,
        kp_seed,
    )) in enumerate(all_experiments):

        print(f"Running experiment {ix + 1}/{len(all_experiments)} with kappa_s={kappa_s}, kappa_p={kappa_p}, ks_seed={ks_seed}, kp_seed={kp_seed}")

        settings = WorldSettings(
            n_simulation_periods=NUMBER_OF_SIMULATION_PERIODS,
            current_period=0,
            simulation_start_date=date(2025, 1, 6),
            weeks_per_period=4,
            n_init_suppliers=9,
            n_dispatchers=3,
            shortage_concentration_rho=0.0,
            non_stationary_theta_share=0.0, # Theta no change across periods
            delivery_call_off_rate_sampler=default_delivery_call_off_rate_draw,
            # Using Hierarchical Beta-Bernoulli
            use_hierarchical_theta=True,
            m_global=M_GLOBAL,
            kappa_s=kappa_s,
            kappa_p=kappa_p,
            # Use pre
            world_config=world_config,
            target_workload_dispatcher_fulltime_per_day=1200,
            target_workload_dispatcher_parttime_per_day=960,
        )

        world = initialize_world(
            settings=settings,
            random_seed=DEFAULT_GENERAL_RANDOM_SEED,
            ks_seed=ks_seed,
            kp_seed=kp_seed,
        )

        for _ in range(settings.n_simulation_periods-1):
            step_world_forward(world)   

        final_scenarios_data = []
        retries = 0
        while True:
            print("Generating scenarios for current period...", retries, "times retried")
            random_seed_for_scenarios = random.randint(0, 500)
            scenarios_data = create_scenarios_for_current_period(
                world=world,
                n_scenarios=NUMBER_OF_SAMPLES_FOR_BALANCE_CHECK,
                critical_case_duration_minutes=CRITICAL_HANDLING_TIME,
                source_period_id=world.current_period,
                random_seed=random_seed_for_scenarios,
            )

            number_of_scenarios_in_range = 0
            
            valid_scenarios = []
            for scenario_data in scenarios_data:
                number_of_critical_deliveries = sum([ele['n_critical_delivery_calloffs'] for ele in list(scenario_data['delivery_performance_by_part'].values())])
                if ACCEPTABLE_RANGE[0] <= number_of_critical_deliveries <= ACCEPTABLE_RANGE[1]:
                    number_of_scenarios_in_range += 1
                    valid_scenarios.append(scenario_data)

            print(f"Number of scenarios within ±10% of 360 critical deliveries: {number_of_scenarios_in_range} out of {len(scenarios_data)}")

            final_scenarios_data.extend(valid_scenarios)
            print(f"Total valid scenarios collected so far: {len(final_scenarios_data)}")
            if len(final_scenarios_data) >= NUMBER_OF_SCENARIOS:
                final_scenarios_data = final_scenarios_data[:NUMBER_OF_SCENARIOS]
                break

            retries += 1


        raw_input = world_to_input_data(world=world, period_t=0, period_t1=0)
        input_data = InputData.model_validate(raw_input)

        input_data.dispatchers = {
            '1': Dispatcher(dispatcher_id='1', fte_factor=0.6, target_workload={'Mon': 720.0, 'Tue': 720.0, 'Wed': 720.0, 'Thu': 720.0, 'Fri': 720.0}),
            '2': Dispatcher(dispatcher_id='2', fte_factor=0.6, target_workload={'Mon': 720.0, 'Tue': 720.0, 'Wed': 720.0, 'Thu': 720.0, 'Fri': 720.0}),
            '3': Dispatcher(dispatcher_id='3', fte_factor=0.6, target_workload={'Mon': 720.0, 'Tue': 720.0, 'Wed': 720.0, 'Thu': 720.0, 'Fri': 720.0})
        }

        raw_scenarios = tp3_scenarios_to_scenario_set(
            tp3_scenarios=final_scenarios_data,
            input_data=raw_input,
            handling_time=CRITICAL_HANDLING_TIME,
        )
        scenario_set = ScenarioSet.model_validate(raw_scenarios)
        scenario_set.data_id = input_data.id

        

        save_experiment_dir = os.path.join(save_dir, f"ks_{kappa_s}_kp_{kappa_p}_ksseed_{ks_seed}_kpseed_{kp_seed}_general_seed_{DEFAULT_GENERAL_RANDOM_SEED}")

        save_world(world=world, path=os.path.join(save_experiment_dir, "world.json"))
        save_tp3_scenarios(
            scenarios=final_scenarios_data,
            path=os.path.join(save_experiment_dir, "scenarios.json"),
        )

        input_data.save(os.path.join(save_experiment_dir, "input_data.json"))
        scenario_set.save(os.path.join(save_experiment_dir, "scenario_set.json"))

        print(f"Saved data for experiment with kappa_s={kappa_s}, kappa_p={kappa_p}, ks_seed={ks_seed}, kp_seed={kp_seed} at {save_experiment_dir}")

        print("-" * 50)

def generate_average_scenario(save_experiments_dir):
    LIST_KAPPA_S = [5, 50, 150]
    LIST_KAPPA_P = [5, 50, 150]

    LIST_RANDOM_SEEDS = [25]

    DEFAULT_GENERAL_RANDOM_SEED = 50

    CRITICAL_HANDLING_TIME = 30  # minutes

    all_experiments = list(
        itertools.product(
            LIST_KAPPA_S,
            LIST_KAPPA_P,
            LIST_RANDOM_SEEDS,
            LIST_RANDOM_SEEDS,
        )
    )

    print(f"Total experiments: {len(all_experiments)}")
    for (i, (
        kappa_s,
        kappa_p,
        ks_seed,
        kp_seed,
    )) in enumerate(all_experiments):

        print(f"Running experiment {i+1}/{len(all_experiments)} with ks={kappa_s}, kp={kappa_p}, ks_seed={ks_seed}, kp_seed={kp_seed}") 
        data_dir = os.path.join(save_experiments_dir, f"ks_{kappa_s}_kp_{kappa_p}_ksseed_{ks_seed}_kpseed_{kp_seed}_general_seed_{DEFAULT_GENERAL_RANDOM_SEED}")

        world_path = os.path.join(data_dir, "world.json")
        world = load_world(world_path)

        average_scenario = create_average_scenario_for_current_period(
            world=world,
            critical_case_duration_minutes=CRITICAL_HANDLING_TIME,
            source_period_id=world.current_period
        )

        raw_input = world_to_input_data(world=world, period_t=0, period_t1=0)

        average_scenario_raw = tp3_average_scenario_to_scenario_set(
            average_scenario=average_scenario,
            input_data=raw_input,
            handling_time=CRITICAL_HANDLING_TIME,
        )

        average_scenario_set = ScenarioSet.model_validate(average_scenario_raw)

        save_tp3_scenarios(
            scenarios=[average_scenario],
            path=os.path.join(data_dir, "average_scenario.json"),
        )

        average_scenario_set.save(os.path.join(data_dir, "average_scenario_set.json"))

def generate_ground_truth(save_experiments_dir):

    def convert_keys(obj):
        if isinstance(obj, dict):
            return {
                key.isoformat() if isinstance(key, (date, datetime)) else key: convert_keys(value)
                for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [convert_keys(x) for x in obj]
        elif isinstance(obj, tuple):
            return [convert_keys(x) for x in obj]
        elif isinstance(obj, (date, datetime)):
            return obj.isoformat()
        else:
            return obj

    LIST_KAPPA_S = [5, 50, 150]
    LIST_KAPPA_P = [5, 50, 150]

    LIST_RANDOM_SEEDS = [25]

    CRITICAL_HANDLING_TIME = 30  # minutes

    NUMBER_OF_SAMPLES_FOR_BALANCE_CHECK = 7000
    NUMBER_OF_SCENARIOS = 10000
    SAVE_EVERY_N_SCENARIOS = 1000
    NUMBER_OF_SAVING_TURNS = math.ceil(NUMBER_OF_SCENARIOS / SAVE_EVERY_N_SCENARIOS)

    ACCEPTABLE_RANGE = (360 - 360 * 0.0, 360 + 360 * 0.0)  # exactly 360

    all_experiments = list(
        itertools.product(
            LIST_KAPPA_S,
            LIST_KAPPA_P,
            LIST_RANDOM_SEEDS,
            LIST_RANDOM_SEEDS,
        )
    )

    # Stage 1
    print(f"Total experiments: {len(all_experiments)}")
    for (ix, (
        kappa_s,
        kappa_p,
        ks_seed,
        kp_seed,
    )) in enumerate(all_experiments):
        print(f"Running experiment {ix + 1}/{len(all_experiments)} with kappa_s={kappa_s}, kappa_p={kappa_p}, ks_seed={ks_seed}, kp_seed={kp_seed}")

        experiment_path = os.path.join(save_experiments_dir, f"ks_{kappa_s}_kp_{kappa_p}_ksseed_25_kpseed_25_general_seed_50")
        world = load_world(os.path.join(experiment_path, "world.json"))
        raw_input = world_to_input_data(world=world, period_t=0, period_t1=0)
        input_data = InputData.model_validate(raw_input)

        os.makedirs(os.path.join(experiment_path, "raw_groundtruth"), exist_ok=True)

        final_scenarios_data = []
        retries = 0
        save_ix = 0
        while True:
            print("Generating scenarios for current period...", retries, "times retried")
            random_seed_for_scenarios = random.randint(0, 5000)
            groundtruth_scenarios = create_groundtruth_scenarios_for_current_period(
                world=world,
                n_scenarios=NUMBER_OF_SAMPLES_FOR_BALANCE_CHECK,
                critical_case_duration_minutes=CRITICAL_HANDLING_TIME,
                source_period_id=world.current_period,
                random_seed=random_seed_for_scenarios,
            )

            number_of_scenarios_in_range = 0
            valid_scenarios = []
            for groundtruth_scenario in groundtruth_scenarios:
                number_of_critical_deliveries = sum([ele['n_critical_delivery_calloffs'] for ele in list(groundtruth_scenario['delivery_performance_by_part'].values())])
                if ACCEPTABLE_RANGE[0] <= number_of_critical_deliveries <= ACCEPTABLE_RANGE[1]:
                    number_of_scenarios_in_range += 1
                    valid_scenarios.append(groundtruth_scenario)

            print(f"Number of scenarios within ±0% of 360 critical deliveries: {number_of_scenarios_in_range} out of {len(groundtruth_scenarios)}")

            final_scenarios_data.extend(valid_scenarios)
            print(f"Total valid scenarios collected so far: {save_ix * SAVE_EVERY_N_SCENARIOS + len(final_scenarios_data)}")

            if len(final_scenarios_data) >= SAVE_EVERY_N_SCENARIOS:

                with open(os.path.join(experiment_path, "raw_groundtruth", f"{save_ix}.jsonl"), 'w') as f1:
                    for ele in final_scenarios_data[:SAVE_EVERY_N_SCENARIOS]:
                        line = json.dumps(convert_keys(ele))
                        f1.write(f"{line}\n")

                final_scenarios_data = final_scenarios_data[SAVE_EVERY_N_SCENARIOS:]
                save_ix += 1

                if save_ix >= NUMBER_OF_SAVING_TURNS:
                    break
        
            retries += 1

        # clean
        del final_scenarios_data
        del groundtruth_scenarios
        del valid_scenarios

    # Stage 2
    for (ix, (
        kappa_s,
        kappa_p,
        ks_seed,
        kp_seed,
    )) in enumerate(all_experiments):
        print(f"Running experiment {ix + 1}/{len(all_experiments)} with kappa_s={kappa_s}, kappa_p={kappa_p}, ks_seed={ks_seed}, kp_seed={kp_seed}")

        experiment_path = os.path.join(save_experiments_dir, f"ks_{kappa_s}_kp_{kappa_p}_ksseed_25_kpseed_25_general_seed_50")
        world = load_world(os.path.join(experiment_path, "world.json"))
        raw_input = world_to_input_data(world=world, period_t=0, period_t1=0)
        input_data = InputData.model_validate(raw_input)
        

        theta_records = []
        critical_records = []
        scenario_index_offset = 0

        for ix in tqdm(range(NUMBER_OF_SAVING_TURNS)):
            with open(os.path.join(experiment_path, "raw_groundtruth", f"{ix}.jsonl"), 'r') as f1:
                tp3_scenarios = [json.loads(line.strip()) for line in f1.readlines()]

                chunk = tp3_scenarios_chunk_to_scenario_set(
                    tp3_scenarios,
                    raw_input,
                    CRITICAL_HANDLING_TIME,
                    scenario_index_offset=scenario_index_offset
                )
                # --------------------------------------------------------
                # Merge records
                # --------------------------------------------------------

                theta_records.extend(
                    chunk["theta_records"]
                )

                critical_records.extend(
                    chunk["critical_records"]
                )

                scenario_index_offset += len(tp3_scenarios)

                # --------------------------------------------------------
                # Release memory
                # --------------------------------------------------------

                # del chunk
                # del tp3_scenarios

        # ── Aggregate part_stats ───────────────────────────────────────────────
        bucket_part = {}
        meta_part = {}
        for r in critical_records:
            key = (r["part_id"], r["day"])
            bucket_part.setdefault(key, []).append(r["critical_count"])
            meta_part[key] = (r["supplier_id"], r["n_deliveries"])

        part_stats = []
        for (pid, tau) in sorted(bucket_part):
            arr = np.array(bucket_part[(pid, tau)], dtype=float)
            sid, n_del = meta_part[(pid, tau)]
            part_stats.append({
                "part_id":         pid,
                "supplier_id":     sid,
                "day":             tau,
                "n_deliveries":    n_del,
                "mean_critical":   round(float(np.mean(arr)),          4),
                "std_critical":    round(float(np.std(arr)),           4),
                "p5_critical":     round(float(np.percentile(arr,  5)), 4),
                "p50_critical":    round(float(np.percentile(arr, 50)), 4),
                "p95_critical":    round(float(np.percentile(arr, 95)), 4),
                "mean_workload_min": round(float(np.mean(arr)) * CRITICAL_HANDLING_TIME, 4),
            })

        # ── Aggregate supplier_stats ───────────────────────────────────────────
        # Re-aggregate properly:
        workload_sup_bucket = {}
        for r in critical_records:
            key = (r["supplier_id"], r["day"])
            omega_map = workload_sup_bucket.setdefault(key, {})
            omega_map[r["scenario_index"]] = (
                omega_map.get(r["scenario_index"], 0.0) + r["workload_contribution_min"]
            )

        supplier_stats = []
        for (sid, tau) in sorted(workload_sup_bucket):
            arr = np.array(list(workload_sup_bucket[(sid, tau)].values()), dtype=float)
            supplier_stats.append({
                "supplier_id":         sid,
                "day":                 tau,
                "mean_workload_min":   round(float(np.mean(arr)),          4),
                "std_workload_min":    round(float(np.std(arr)),           4),
                "p5_workload_min":     round(float(np.percentile(arr,  5)), 4),
                "p50_workload_min":    round(float(np.percentile(arr, 50)), 4),
                "p95_workload_min":    round(float(np.percentile(arr, 95)), 4),
            })

        # ── Infer n_parts from suppliers ─────────────────────────────────
        n_parts = len(set(r["part_id"] for r in theta_records))

        raw_scenarios = {
            "scenario_set_id": str(uuid4()),
            "data_id":         raw_input.get("meta", {}).get("data_id", "tp3_world"),
            "config": {
                "n_scenarios":   scenario_index_offset,
                "handling_time": CRITICAL_HANDLING_TIME,
                "alpha0":        1.0,
                "beta0":         1.0,
                "rng_seed":      0,
            },
            "created_at":       datetime.now(timezone.utc).isoformat(),
            "n_parts":          n_parts,
            "n_rep_days":       len(raw_input.get("rep_days", ["Mon", "Tue", "Wed", "Thu", "Fri"])),
            "theta_records":    theta_records,
            "critical_records": critical_records,
            "part_stats":       part_stats,
            "supplier_stats":   supplier_stats,
            # Traceability
            "_source":          "tp3_scenarios_to_scenario_set",
            "_source_period_id":   tp3_scenarios[0]["source_period_id"],
            "_forecast_period_id": tp3_scenarios[0]["forecast_period_id"]
        }

        scenario_set = ScenarioSet.model_validate(raw_scenarios)
        scenario_set.data_id = input_data.id


        scenario_set.save(os.path.join(experiment_path, "groundtruth.json"))

        print("-" * 50)

    pass


def main():
    ROOT_DIR = "shared_runs/experiment_20260827_230630"
    os.makedirs(ROOT_DIR, exist_ok=True)

    # Generate world config
    data_config_dir = os.path.join(ROOT_DIR, "data_config")
    os.makedirs(data_config_dir, exist_ok=True)

    world_config = generate_world_config()
    with open(os.path.join(data_config_dir, "world_config.json"), "w") as f:
        json.dump(world_config, f, indent=4)


    save_experiments_dir = os.path.join(ROOT_DIR, "data_simulation")
    os.makedirs(save_experiments_dir, exist_ok=True)

    # Generate multiple data simulation
    generate_multiple_data_simulation(world_config=world_config, save_dir=save_experiments_dir)

    # Generate average scenario
    generate_average_scenario(save_experiments_dir=save_experiments_dir)

    # Generate ground truth
    generate_ground_truth(save_experiments_dir=save_experiments_dir)




if __name__ == "__main__":
    main()

    
