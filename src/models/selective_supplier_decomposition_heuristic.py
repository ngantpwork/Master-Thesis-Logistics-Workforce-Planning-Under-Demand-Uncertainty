from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any, Sequence
import zipfile
import os
import json
import itertools
import numpy as np
import pandas as pd
from src.schemas import InputData, ScenarioSet
from src.models.config.part_to_dispatcher_workload_balancing_model_config import (
    PartToDispatcherWorkloadBalancingModelConfig,
)

from src.models.part_to_dispatcher_workload_balancing_model import (
    PartToDispatcherWorkloadBalancingModel,
)
# from src.evaluation.evaluate_assignment import *

def compute_supplier_statistics(
    part_workload: pd.DataFrame,
    supplier_col: str = "supplier_id",
    part_col: str = "part_id",
    workload_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Tính statistic cho từng supplier.

    Metrics:
        E_s^total = sum_{t in T} E[W_{s,t}]
        N_s = |P_s|
        largest_part_share
            = max_p(W_p) / sum_p(W_p)

    Returns
    -------
    DataFrame:
        supplier_id
        expected_total_workload
        num_parts
        largest_part_share
    """

    if workload_cols is None:
        raise ValueError(
            "workload_cols must be provided, "
            "e.g. ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']"
        )

    required_cols = [supplier_col, part_col] + list(workload_cols)

    missing_cols = set(required_cols) - set(part_workload.columns)

    if missing_cols:
        raise ValueError(
            f"Missing columns: {missing_cols}"
        )

    # ==============================================================
    # 1. Expected total workload
    #
    # E_s^total = sum_t E[W_s,t]
    # ==============================================================

    supplier_total_workload = (
        part_workload
        .groupby(supplier_col)[list(workload_cols)]
        .sum()
        .mean(axis=1)
        .rename("mean_workload")
    )

    # ==============================================================
    # 2. Number of parts
    #
    # N_s = |P_s|
    # ==============================================================

    supplier_num_parts = (
        part_workload
        .groupby(supplier_col)[part_col]
        .nunique()
        .rename("num_parts")
    )

    # ==============================================================
    # 3. Largest Part Share
    #
    # W_p = sum_t W_{p,t}
    #
    # largest_part_share
    #     = max_p(W_p) / sum_p(W_p)
    # ==============================================================

    part_weekly = part_workload.copy()

    part_weekly["_part_weekly_workload"] = (
        part_weekly[list(workload_cols)]
        .sum(axis=1)
    )

    largest_part_share = (
        part_weekly
        .groupby(supplier_col)["_part_weekly_workload"]
        .agg(["max", "sum"])
        .assign(
            largest_part_share=lambda x: (
                x["max"] / x["sum"]
            ).where(x["sum"] > 0)
        )["largest_part_share"]
        .rename("largest_part_share")
    )

    # ==============================================================
    # 4. Combine
    # ==============================================================

    stats = pd.concat(
        [
            supplier_total_workload,
            supplier_num_parts,
            largest_part_share,
        ],
        axis=1,
    ).reset_index()

    return stats

def sort_suppliers_to_split(
    supplier_features: pd.DataFrame,
    k: int,
    workload_weight: float = 0.6,
    largest_part_share_weight: float = 0.4,
    num_parts_weight: float = 0.0,
    supplier_col: str = "supplier_id",
    workload_col: str = "mean_workload",
    largest_part_share_col: str = "largest_part_share",
    num_parts_col: str = "num_parts"
):
    """
    Select K suppliers for decomposition using SSD-H.

    Selection criteria:
        1. Higher Mean Expected Workload
        2. Lower Largest-Part Share
        3. Higher Number of Parts (optional)

    The selection is based on within-configuration ranks.

    Parameters
    ----------
    supplier_features : pd.DataFrame
        Supplier-level feature table.

    k : int
        Number of suppliers to select for splitting.

    workload_weight : float, default=0.6
        Weight assigned to Mean Expected Workload.

    largest_part_share_weight : float, default=0.4
        Weight assigned to Largest-Part Share.

    num_parts_weight : float, default=0.0
        Optional weight assigned to Number of Parts.
        Set to 0.0 if Number of Parts is used only as a tie-breaker.

    Returns
    -------
    pd.DataFrame
        Supplier ranking and selection results.
    """

    df = supplier_features.copy()

    # --------------------------------------------------
    # 1. Validate inputs
    # --------------------------------------------------

    if k <= 0:
        raise ValueError("k must be greater than 0.")

    if k > len(df):
        raise ValueError(
            f"k={k} exceeds the number of suppliers={len(df)}."
        )

    weights = {
        "workload": workload_weight,
        "largest_part_share": largest_part_share_weight,
        "num_parts": num_parts_weight
    }

    total_weight = sum(weights.values())

    if total_weight <= 0:
        raise ValueError(
            "At least one criterion must have a positive weight."
        )

    # Normalize weights
    workload_weight /= total_weight
    largest_part_share_weight /= total_weight
    num_parts_weight /= total_weight

    # --------------------------------------------------
    # 2. Calculate within-configuration ranks
    #
    # Rank 1 = most desirable for decomposition
    # --------------------------------------------------

    # Higher workload = better
    df["rank_workload"] = df[
        workload_col
    ].rank(
        ascending=False,
        method="average"
    )

    # Lower largest-part share = better
    df["rank_largest_part_share"] = df[
        largest_part_share_col
    ].rank(
        ascending=True,
        method="average"
    )

    # Higher number of parts = better
    df["rank_num_parts"] = df[
        num_parts_col
    ].rank(
        ascending=False,
        method="average"
    )

    # --------------------------------------------------
    # 3. Normalize ranks to [0, 1]
    #
    # Score = 1 for best supplier
    # Score = 0 for worst supplier
    # --------------------------------------------------

    n = len(df)

    if n == 1:

        df["score_workload"] = 1.0
        df["score_largest_part_share"] = 1.0
        df["score_num_parts"] = 1.0

    else:

        df["score_workload"] = (
            n - df["rank_workload"]
        ) / (n - 1)

        df["score_largest_part_share"] = (
            n - df["rank_largest_part_share"]
        ) / (n - 1)

        df["score_num_parts"] = (
            n - df["rank_num_parts"]
        ) / (n - 1)

    # --------------------------------------------------
    # 4. Calculate weighted SSD-H score
    # --------------------------------------------------

    df["ssd_h_score"] = (

        workload_weight
        * df["score_workload"]

        + largest_part_share_weight
        * df["score_largest_part_share"]

        + num_parts_weight
        * df["score_num_parts"]

    )

    # --------------------------------------------------
    # 5. Final ranking
    #
    # Higher SSD-H score = higher priority for splitting
    #
    # Number of parts is also used as a tie-breaker
    # --------------------------------------------------

    df = df.sort_values(
        by=[
            "ssd_h_score",
            num_parts_col,
            supplier_col
        ],
        ascending=[
            False,
            False,
            True
        ]
    ).reset_index(drop=True)

    df["ssd_h_rank"] = range(1, len(df) + 1)

    # --------------------------------------------------
    # 6. Select Top-K suppliers
    # --------------------------------------------------

    df["selected_for_split"] = (
        df["ssd_h_rank"] <= k
    )

    return df

def create_split_supplier_set(
    part_workload: pd.DataFrame,
    selected_suppliers: Sequence,
    supplier_col: str = "supplier_id",
    part_col: str = "part_id",
    sub_supplier_col: str = "supplier_id_split",
) -> pd.DataFrame:
    """
    Tạo supplier set mới S'.

    Với supplier không được split:

        supplier mới = supplier gốc

    Với supplier được split:

        mỗi part trở thành một sub-supplier

        s_p = (s, p)

    Ví dụ:

        Original:

        supplier_id = S1

        parts:
            P1
            P2
            P3

        Sau split:

        S1__P1
        S1__P2
        S1__P3

    Output vẫn giữ part-level structure, chỉ thay đổi supplier grouping.
    """

    df = part_workload.copy()

    selected_suppliers = set(selected_suppliers)

    # Flag supplier được split
    df["is_split_supplier"] = (
        df[supplier_col].isin(selected_suppliers)
    )

    # ----------------------------------------------------------------------
    # Supplier không split:
    #
    # new_supplier = original supplier
    #
    # Supplier split:
    #
    # new_supplier = supplier + part
    # ----------------------------------------------------------------------

    df[sub_supplier_col] = np.where(
        df["is_split_supplier"],

        df[supplier_col].astype(str)
        + "__"
        + df[part_col].astype(str),

        df[supplier_col].astype(str),
    )

    return df