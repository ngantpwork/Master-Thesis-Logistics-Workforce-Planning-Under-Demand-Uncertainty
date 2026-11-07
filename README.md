# Workload Balancing

This repository contains the experimental framework for **workload balancing and supplier-dispatcher assignment** under deterministic and stochastic settings.

The framework includes:

* Deterministic workload balancing models
* Stochastic workload balancing models
* Selective Supplier Decomposition (SSD) heuristic
* Workload-changing experiments
* Small-scale special cases for hypothesis validation
* Ground-truth generation
* Model evaluation
* Risk and Pareto-frontier visualization

---

## 1. Experiment Overview

The experiments are divided into two stages:

### Stage 1 — Small Cases: Hypothesis Validation

Before running the large-scale experiments, two controlled small cases are used to investigate the hypothesis under simplified settings.

These cases contain:

* 6 suppliers
* 60 parts
* 3 dispatchers
* 1 representative day (`Mon`)
* 50 generated scenarios
* 30 minutes of workload for each critical part

The small cases are designed to create controlled workload distributions and compare:

* **STD** — Standard Supplier-to-Dispatcher assignment (`k_max = 0`)
* **PTD** — Part-to-Dispatcher assignment (`k_max = 6`)

The purpose is to understand how the two assignment structures behave under different supplier-level workload patterns before moving to the larger experiments.

### Stage 2 — Large-Scale Experiments

The main experiments evaluate deterministic, stochastic, and SSD heuristic approaches across different combinations of:

* `κ_s`: supplier-related parameter
* `κ_p`: part-related parameter

Both standard and workload-changing settings are evaluated.

---

# 2. Special Cases

The special cases provide a controlled environment for testing the underlying hypothesis before running the full experimental framework.

They can be executed independently from the large-scale experiment pipeline.

---

## 2.1 Special Case A — Heterogeneous Part Risk Within Suppliers

Special Case A is designed to create **systematic heterogeneity between suppliers and parts**.

There are 6 suppliers with 10 parts each.

For Supplier 1:

* 8 parts have `base_true_theta = 0.8`
* 2 parts have `base_true_theta = 0.2`

For Suppliers 2–6:

* 8 parts have `base_true_theta = 0.2`
* 2 parts have `base_true_theta = 0.8`

Therefore, the high-risk parts are distributed differently across suppliers, creating a structured workload pattern.

The expected structure is:

```text
Supplier 1:
    High-risk parts: 8
    Low-risk parts:  2

Suppliers 2–6:
    High-risk parts: 2
    Low-risk parts:  8
```

Each scenario is generated such that the total number of critical parts is exactly **18**.

A scenario is accepted only when:

```text
Total critical parts = 18
```

A total of **50 scenarios** are generated.

### Purpose

This case tests whether allowing parts from the same supplier to be distributed more flexibly across dispatchers can improve workload balancing when risk is concentrated differently within suppliers.

---

## 2.2 Special Case B — Balanced Supplier-Level Risk

Special Case B removes the heterogeneous part-risk structure from Case A.

All 60 parts use:

```text
base_true_theta = 0.3
```

Thus, all suppliers have the same underlying part-level risk distribution.

However, scenarios are constrained so that each supplier generates between **2 and 4 critical parts**.

The accepted scenario must satisfy:

```text
2 <= critical parts per supplier <= 4
```

and:

```text
Total critical parts = 18
```

Again, **50 scenarios** are generated.

### Purpose

This case provides a more balanced workload structure and serves as a comparison against Case A.

Together, Cases A and B allow the hypothesis to be examined under two different workload-generation structures:

```text
Case A
Heterogeneous risk structure
        │
        ▼
Different workload patterns across suppliers
        │
        ▼
Test flexibility of part-level assignment


Case B
Homogeneous part-level risk
        │
        ▼
Controlled supplier-level workload
        │
        ▼
Test behavior under balanced supplier risk
```

---

## 2.3 Models Used in Special Cases

Both special cases compare two configurations of the same part-to-dispatcher model.

### STD

The standard configuration uses:

```text
k_max = 0
```

and is saved as:

```text
shared_runs/special/<case>/models/ptd_k_max_0/
```

### PTD

The part-to-dispatcher configuration uses:

```text
k_max = 6
```

and is saved as:

```text
shared_runs/special/<case>/models/ptd_k_max_6/
```

Both models use the same:

* Input data
* Scenario set
* Random seed
* Objective configuration
* Time limit
* MIP gap

This keeps the comparison focused on the assignment structure.

---

## 2.4 Running the Special Cases

Run the special-case experiment using the corresponding special-case script.

The workflow is:

```text
Special Case A
      │
      ├── Generate input data
      ├── Generate 50 scenarios
      ├── Run STD
      └── Run PTD

Special Case B
      │
      ├── Generate input data
      ├── Generate 50 scenarios
      ├── Run STD
      └── Run PTD
              │
              ▼
        Generate plots
```

The generated data is stored under:

```text
shared_runs/special/
├── a/
│   ├── input_data.json
│   ├── scenario_set.json
│   ├── models/
│   │   ├── ptd_k_max_0/
│   │   └── ptd_k_max_6/
│   └── plots/
│
└── b/
    ├── input_data.json
    ├── scenario_set.json
    ├── models/
    │   ├── ptd_k_max_0/
    │   └── ptd_k_max_6/
    └── plots/
```

---

## 2.5 Special Case Visualizations

The special cases generate four plots for each case.

### 1. Workload Distribution by Supplier

```text
workload_distribution.png
```

Shows the average workload generated by each supplier across the 50 scenarios.

A horizontal reference line indicates the dispatcher target workload.

---

### 2. Workload Distribution by Part and Supplier

```text
workload_distribution_by_part_and_supplier.png
```

Shows the average workload contribution of each part, grouped by supplier.

This visualization makes the workload heterogeneity between suppliers and parts explicit.

---

### 3. Dispatcher Workload: PTD vs STD

```text
dispatcher_workload.png
```

Compares the resulting dispatcher workloads between:

* STD
* PTD

A fixed target workload is shown as a reference.

This plot directly illustrates how the different assignment structures distribute workload across dispatchers.

---

### 4. Supplier Allocation by Dispatcher

```text
supplier_allocation_by_dispatcher.png
```

Shows how many parts from each supplier are assigned to each dispatcher for:

* PTD
* STD

This provides a direct view of the difference in supplier fragmentation and part-level allocation between the two approaches.

---

## 2.6 Role of the Special Cases

The special cases are **not intended to replace the main experiments**.

Their purpose is to provide a controlled preliminary experiment for the hypothesis.

The progression is:

```text
Controlled Small Cases
        │
        │  Validate / investigate hypothesis
        ▼
Large-Scale Experiments
        │
        │  Evaluate across κ_s × κ_p
        ▼
Workload-Changing Experiments
        │
        ▼
Overall Evaluation
        │
        ▼
Risk & Pareto Analysis
```

The small cases therefore provide an interpretable example of the mechanism being investigated, while the large-scale experiments provide broader empirical evaluation.

---

# 3. Large-Scale Experiment Workflow

After the special cases, the main experiment pipeline can be executed.

```text
Generate World Configuration
          │
          ▼
Generate Experimental Data
          │
          ├── Generate data for (κ_s, κ_p) pairs
          │
          └── Generate ground truth
          │
          ▼
Run Deterministic & Stochastic Models
          │
          ▼
Run SSD Heuristic
          │
          ▼
Run Workload-Changing Experiments
          │
          ▼
Evaluate All Experiments
          │
          ▼
Generate Risk & Pareto Plots
```

---

# 4. Generate Experimental Data

Run:

```bash
python generate_data.py
```

This step:

1. Creates the world configuration.
2. Generates experimental data for each `(κ_s, κ_p)` pair.
3. Generates the corresponding ground truth.

The generated instances are used by the subsequent model experiments.

---

# 5. Run Deterministic and Stochastic Models

Run:

```bash
python run_models.py
```

This script runs the deterministic and stochastic workload-balancing models for the configured `(κ_s, κ_p)` experiments.

---

# 6. Run Selective Supplier Decomposition Heuristic

Run:

```bash
python run_ssd_heuristic_model.py
```

This runs the **Selective Supplier Decomposition (SSD) heuristic** for the configured `(κ_s, κ_p)` experiments.

The SSD heuristic provides an alternative solution approach to the full optimization models.

---

# 7. Workload-Changing Experiments

Run:

```bash
python run_models_with_workload_changing.py
```

This script evaluates the workload-balancing approaches under changing workload conditions.

The following approaches are included:

* Deterministic models
* Stochastic models
* Selective Supplier Decomposition heuristic

The experiments are performed for the configured `(κ_s, κ_p)` combinations.

---

# 8. Evaluate All Experiments

Run:

```bash
python evaluate_all_experiments.py
```

This evaluates the results from all experiments, including:

* Standard experiments
* Workload-changing experiments
* Deterministic models
* Stochastic models
* SSD heuristic

The resulting metrics are used for model comparison and visualization.

---

# 9. Generate Plots

Run:

```bash
python plots.py
```

This generates the main visualizations for the large-scale experiments.

The plots include:

* Supplier risk
* Pareto frontiers
* Other experiment-level comparisons derived from the evaluation results

---

# 10. Complete Experiment Pipeline

For the complete experimental study, the recommended order is:

### Step 1 — Hypothesis validation

Run the special cases first.

```bash
# Special cases
python <special_case_script>.py
```

This generates the controlled Case A and Case B results and their visualizations.

### Step 2 — Generate large-scale experiments

```bash
python generate_data.py
```

### Step 3 — Run deterministic and stochastic models

```bash
python run_models.py
```

### Step 4 — Run SSD heuristic

```bash
python run_ssd_heuristic_model.py
```

### Step 5 — Run workload-changing experiments

```bash
python run_models_with_workload_changing.py
```

### Step 6 — Evaluate all experiments

```bash
python evaluate_all_experiments.py
```

### Step 7 — Generate plots

```bash
python plots.py
```

---

# 11. Experiment Structure

The overall experimental design can be summarized as:

```text
                    Workload Balancing Study
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
       Special Cases                   Large Experiments
              │                               │
       ┌──────┴──────┐                ┌───────┴────────┐
       │             │                │                │
    Case A         Case B        Fixed Workload   Changing Workload
       │             │                │                │
       └──────┬──────┘                └───────┬────────┘
              │                               │
              ▼                               ▼
       Hypothesis Study              κ_s × κ_p Experiments
              │                               │
              └───────────────┬───────────────┘
                              ▼
                         Evaluation
                              │
                              ▼
                     Risk / Pareto Analysis
```

The special cases provide controlled and interpretable examples for the initial hypothesis, while the large-scale experiments test the behavior across a broader range of parameter configurations and workload conditions.
