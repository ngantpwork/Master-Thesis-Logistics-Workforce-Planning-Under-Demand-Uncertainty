"""
shybrid_feature_extraction.py
==============================

Trich xuat bang feature (mean / peak / PAR / CV / num_parts / largest-part-share /
day-share / peak-part-share) cho tung supplier trong tung config, kem rank va
percentile rank, va gan nhan split order (First / Later / Never split) dua tren
output S-Hybrid.

INPUT 1 — part-level expected workload (wide hoac long format), toi thieu cac cot:
    config, supplier_id, part_id, Mon, Tue, Wed, Thu, Fri
  (neu la long format thi: config, supplier_id, part_id, weekday, expected_workload
   — dung `--long-format` hoac goi `load_part_workload(..., long_format=True)`)

INPUT 2 — S-Hybrid Pareto assignment, moi dong la 1 (config, solution_id, supplier_id,
part_id) -> dispatcher_id da duoc gan trong solution do. Toi thieu cac cot:
    config, solution_id, supplier_id, part_id, dispatcher_id
  `solution_id` CHI can la 1 gia tri phan biet cac solution/assignment khac
  nhau voi nhau (vd objective value, run id, ten file...) — KHONG can the hien
  thu tu 0/1/2/... Script tu tinh so supplier bi split trong tung solution roi
  tu sap xep tang dan de suy ra thu tu "0-split -> Pareto frontier". Neu file
  cua ban dat ten cot khac, truyen qua `load_assignment(path, solution_col=...)`.
  Neu moi solution nam trong 1 file rieng (khong co cot chung nao), dung
  `load_assignment_from_multiple_files([...])` thay vao.

OUTPUT:
    feature_table   — 1 dong / (config, supplier): tat ca feature + rank + percentile
                       + split_order + split_group (First/Later/Never)
    summary_table   — median percentile theo tung feature, nhom theo split_group
    topk_table      — tan suat First-split supplier nam trong Top-k (k=3 mac dinh)
                       cho tung feature

Chay truc tiep script nay (python3 shybrid_feature_extraction.py) se demo tren
du lieu synthetic de kiem tra logic, khong can file that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]

# ----------------------------------------------------------------------------
# 1. SPLIT-ORDER CLASSIFICATION (tu file assignment)
# ----------------------------------------------------------------------------

def compute_split_order(assignment: pd.DataFrame) -> pd.DataFrame:
    """Voi moi (config, supplier), xac dinh split_group.


    Chi co 1 assignment / config (KHONG co cot phan biet nhieu solution —
    vd assignment_df chi co config, supplier_id, part_id, dispatcher_id,
    moi config chi xuat hien 1 lan duy nhat). Day la truong hop pho bien
    khi ban chi luu lai 1 solution cuoi (vd diem Knee) cho moi bai toan.
    -> CHI tinh duoc BINARY: split_group in {"Split", "NotSplit"}.
        split_order se la NaN (khong co thong tin thu tu).

    
    Tra ve DataFrame: config, supplier_id, split_group.
    """
    disp_counts = (
        assignment.groupby(["config", "supplier_id"])["dispatcher_id"]
        .nunique()
        .rename("n_dispatchers")
        .reset_index()
    )
    disp_counts["split_group"] = np.where(disp_counts["n_dispatchers"] > 1, "Split", "NotSplit")

    return disp_counts[["config", "supplier_id", "split_group"]]


# ----------------------------------------------------------------------------
# 2. FEATURE COMPUTATION (tu file part-level workload)
# ----------------------------------------------------------------------------

def compute_supplier_features(part_wl: pd.DataFrame, cv_ddof: int = 0) -> pd.DataFrame:
    """Tinh cac feature workload/splittability cho tung (config, supplier).

    cv_ddof: 0 = population std (mac dinh, 5 ngay cho la toan bo population trong
             tuan dai dien), 1 = sample std.
    """
    rows = []

    for config, cfg_df in part_wl.groupby("config"):
        # W_{s,tau}: tong workload cua supplier trong tung ngay
        supplier_daily = cfg_df.groupby("supplier_id")[WEEKDAYS].sum()

        # workload toan bo group trong tung ngay (dung cho DayShare)
        day_totals = supplier_daily.sum(axis=0)

        # W_p: tong workload ca tuan cua tung part
        part_weekly_total = cfg_df.assign(W_p=cfg_df[WEEKDAYS].sum(axis=1))

        for s, s_daily in supplier_daily.iterrows():
            values = s_daily.values.astype(float)
            mean_wl = values.mean()
            peak_wl = values.max()
            par = peak_wl / mean_wl if mean_wl > 0 else np.nan
            cv = (values.std(ddof=cv_ddof) / mean_wl) if mean_wl > 0 else np.nan

            peak_day = s_daily.idxmax()
            day_share = (
                s_daily[peak_day] / day_totals[peak_day]
                if day_totals[peak_day] > 0 else np.nan
            )

            s_parts = part_weekly_total[part_weekly_total["supplier_id"] == s]
            num_parts = s_parts["part_id"].nunique()
            total_part_wl = s_parts["W_p"].sum()
            lps = (s_parts["W_p"].max() / total_part_wl) if total_part_wl > 0 else np.nan

            # peak-day largest-part share
            peak_day_part_wl = s_parts[peak_day]
            peak_day_supplier_wl = s_daily[peak_day]
            peak_part_share = (
                peak_day_part_wl.max() / peak_day_supplier_wl
                if peak_day_supplier_wl > 0 else np.nan
            )

            rows.append({
                "config": config,
                "supplier_id": s,
                "mean_workload": mean_wl,
                "peak_workload": peak_wl,
                "peak_to_avg_ratio": par,
                "cv_weekday": cv,
                "peak_day": peak_day,
                "day_share_on_peak": day_share,
                "num_parts": num_parts,
                "largest_part_share": lps,
                "peak_day_largest_part_share": peak_part_share,
            })

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 3. RANK + PERCENTILE RANK (trong tung config, R=1 / percentile=1 la cao nhat)
# ----------------------------------------------------------------------------

RANK_FEATURES = [
    "mean_workload",
    "peak_workload",
    "peak_to_avg_ratio",
    "cv_weekday",
    "day_share_on_peak",
    "num_parts",
    "largest_part_share",
    "peak_day_largest_part_share",
]


def add_ranks(feature_df: pd.DataFrame) -> pd.DataFrame:
    df = feature_df.copy()
    for feat in RANK_FEATURES:
        df[f"{feat}_rank"] = df.groupby("config")[feat].rank(ascending=False, method="average")
        df[f"{feat}_pct"] = df.groupby("config")[feat].rank(ascending=True, pct=True)
    return df


# ----------------------------------------------------------------------------
# 4. SUMMARY TABLES
# ----------------------------------------------------------------------------

def build_summary_table(merged: pd.DataFrame) -> pd.DataFrame:
    """Median percentile rank cho tung feature, nhom theo split_group.
    Tu nhan dien la che do binary (Split/NotSplit) hay day du (First/Later/Never)."""
    pct_cols = [f"{feat}_pct" for feat in RANK_FEATURES]
    summary = merged.groupby("split_group")[pct_cols].median().T
    preferred_order = ["First", "Later", "Never", "Split", "NotSplit"]
    summary = summary.reindex(columns=[c for c in preferred_order if c in summary.columns])
    summary.index.name = "feature"
    return summary.reset_index()


def build_topk_table(
    merged: pd.DataFrame,
    k: int = 3,
    group: str | None = None
) -> pd.DataFrame:
    """
    Tính Mean Precision@k theo từng feature.

    Với mỗi config:
        Precision@k =
            (# supplier thuộc `group` nằm trong Top-k theo feature)
            / k

    Sau đó lấy trung bình Precision@k qua các config.

    Ví dụ:
        Top-3 theo mean_workload:
            S1 = Split
            S2 = NotSplit
            S3 = Split

        Precision@3 = 2 / 3 = 0.667

    Output:
        feature
        mean_precision_at_k
        mean_group_in_topk
    """

    if group is None:
        available = set(merged["split_group"].dropna().unique())
        group = "First" if "First" in available else "Split"

    rows = []

    for feat in RANK_FEATURES:
        rank_col = f"{feat}_rank"

        # Đếm số supplier thuộc group nằm trong Top-k
        # cho từng config
        group_in_topk = (
            merged.assign(
                is_target=merged["split_group"].eq(group),
                is_topk=merged[rank_col].le(k),
            )
            .assign(hit=lambda x: x["is_target"] & x["is_topk"])
            .groupby("config")["hit"]
            .sum()
        )

        # Số supplier thực tế trong Top-k của mỗi config.
        # Dùng để xử lý trường hợp config có ít hơn k supplier
        # hoặc rank ties làm số dòng Top-k khác k.
        topk_size = (
            merged[merged[rank_col] <= k]
            .groupby("config")
            .size()
        )

        # Precision@k theo từng config
        precision_by_config = (
            group_in_topk
            .reindex(merged["config"].unique(), fill_value=0)
            .div(
                topk_size.reindex(
                    merged["config"].unique(),
                    fill_value=0
                )
            )
            .replace([np.inf, -np.inf], np.nan)
        )

        rows.append({
            "feature": feat,
            "group": group,
            f"mean_{group}_in_top{k}": group_in_topk.mean(),
            f"precision_at_{k}": precision_by_config.mean(),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(f"precision_at_{k}", ascending=False)
        .reset_index(drop=True)
    )


# ----------------------------------------------------------------------------
# 5. PIPELINE
# ----------------------------------------------------------------------------

@dataclass
class PipelineResult:
    feature_table: pd.DataFrame
    summary_table: pd.DataFrame
    topk_table: pd.DataFrame


def run_pipeline(part_wl: pd.DataFrame, assignment: pd.DataFrame,
                  cv_ddof: int = 0, topk: int = 3) -> PipelineResult:
    features = compute_supplier_features(part_wl, cv_ddof=cv_ddof)
    features = add_ranks(features)
    split_info = compute_split_order(assignment)

    merged = features.merge(split_info, on=["config", "supplier_id"], how="left")
    merged["split_group"] = merged["split_group"].fillna("Never")

    summary = build_summary_table(merged)
    topk_tbl = build_topk_table(merged, k=topk, group=None)

    return PipelineResult(feature_table=merged, summary_table=summary, topk_table=topk_tbl)
