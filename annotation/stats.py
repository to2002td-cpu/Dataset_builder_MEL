"""
Pure statistics on the annotation master table (no I/O, no plotting).

Every function takes the DataFrames produced by loading.py and returns pandas
objects ready to display or plot. Generic over N annotators.

Usage (from the analysis notebook, cwd = annotation/):
    import stats

    per_ann = stats.label_rates_and_bias(master, annotators)
    K = stats.pairwise_kappa(master, annotators)
    times = stats.cap_outliers(times)          # adds t / is_outlier columns
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

# Landis & Koch (1977) interpretation scale for Cohen's kappa.
KAPPA_SCALE: list[tuple[float, str]] = [
    (0.80, "Almost perfect"),
    (0.60, "Substantial"),
    (0.40, "Moderate"),
    (0.20, "Fair"),
    (0.00, "Slight"),
]


def interpret_kappa(kappa: float) -> str:
    """Landis & Koch label for a kappa value ('Poor' below 0)."""
    for threshold, label in KAPPA_SCALE:
        if kappa >= threshold:
            return label
    return "Poor"


def annotator_volume(master: pd.DataFrame, annotators: list[str]) -> pd.Series:
    """Number of instances labeled by each annotator, descending."""
    return pd.Series(
        {u: int(master[u].notna().sum()) for u in annotators}
    ).sort_values(ascending=False)


def label_counts(master: pd.DataFrame, annotators: list[str]) -> pd.DataFrame:
    """YES / NO / not-annotated counts per annotator."""
    return pd.DataFrame({
        "YES": {u: int((master[u] == "YES").sum()) for u in annotators},
        "NO": {u: int((master[u] == "NO").sum()) for u in annotators},
        "not annotated": {u: int(master[u].isna().sum()) for u in annotators},
    })


def label_rates_and_bias(master: pd.DataFrame, annotators: list[str]) -> pd.DataFrame:
    """Per-annotator YES/NO rates and signed gap (pts) vs the group mean."""
    per_ann = pd.DataFrame({
        "n": {u: master[u].notna().sum() for u in annotators},
        "YES": {u: (master[u] == "YES").sum() for u in annotators},
        "NO": {u: (master[u] == "NO").sum() for u in annotators},
    })
    per_ann["yes_rate"] = (100 * per_ann["YES"] / per_ann["n"]).round(1)
    per_ann["no_rate"] = (100 * per_ann["NO"] / per_ann["n"]).round(1)
    group_yes = 100 * per_ann["YES"].sum() / per_ann["n"].sum()
    group_no = 100 * per_ann["NO"].sum() / per_ann["n"].sum()
    per_ann["bias_yes"] = (per_ann["yes_rate"] - group_yes).round(1)
    per_ann["bias_no"] = (per_ann["no_rate"] - group_no).round(1)
    per_ann.attrs["group_yes"] = group_yes
    per_ann.attrs["group_no"] = group_no
    return per_ann


def pairwise_kappa(master: pd.DataFrame, annotators: list[str]) -> pd.DataFrame:
    """Square Cohen's-kappa matrix computed on each pair's shared instances."""
    K = pd.DataFrame(np.nan, index=annotators, columns=annotators, dtype=float)
    for a in annotators:
        for b in annotators:
            if a == b:
                K.loc[a, b] = 1.0
                continue
            sub = master[master[a].notna() & master[b].notna()]
            if len(sub) >= 2 and sub[a].nunique() * sub[b].nunique() > 1:
                K.loc[a, b] = cohen_kappa_score(sub[a], sub[b])
    return K


def confusion(master: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Label crosstab between two annotators on their shared instances."""
    sub = master[master[a].notna() & master[b].notna()]
    return pd.crosstab(sub[a], sub[b])


def category_summary(master: pd.DataFrame) -> pd.DataFrame:
    """Per category: n, n_multi, YES rate, and unanimity rate among multi."""
    cat = master.groupby("category").agg(
        n=("instance_id", "size"),
        n_multi=("is_multi", "sum"),
    )
    yes_rate, agreement = {}, {}
    for c, g in master.groupby("category"):
        total_votes = g[["n_yes", "n_no"]].sum().sum()
        yes_rate[c] = 100 * g["n_yes"].sum() / total_votes if total_votes else np.nan
        gm = g[g["is_multi"]]
        agreement[c] = 100 * gm["unanimous"].sum() / len(gm) if len(gm) else np.nan
    cat["yes_rate"] = pd.Series(yes_rate).round(0)
    cat["agreement_multi_%"] = pd.Series(agreement).round(0)
    return cat


def kappa_per_category(master: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Cohen's kappa between two annotators, per category (indicative on low n)."""
    rows = []
    for c, gc in master.groupby("category"):
        sub = gc[gc[a].notna() & gc[b].notna()]
        k = np.nan
        if len(sub) >= 2 and sub[a].nunique() * sub[b].nunique() > 1:
            k = cohen_kappa_score(sub[a], sub[b])
        rows.append({"category": c, "n_common": len(sub),
                     "kappa": round(k, 3) if k == k else np.nan})
    return pd.DataFrame(rows)


def labels_long(master: pd.DataFrame, annotators: list[str]) -> pd.DataFrame:
    """Long form: one row per (instance, annotator, label) — for crosstabs."""
    return master.melt(
        id_vars=["instance_id", "category"], value_vars=annotators,
        var_name="annotator", value_name="label",
    ).dropna(subset=["label"])


def fragile_instances(master: pd.DataFrame) -> pd.DataFrame:
    """Multi-annotated instances whose majority flips if one vote changes."""
    return master[master["is_multi"] & (master["margin"] <= 1)].sort_values("margin")


def cap_outliers(times: pd.DataFrame, method: str = "tukey") -> pd.DataFrame:
    """Winsorize decision times; adds `t` (capped) and `is_outlier` columns.
    method: 'tukey' (Q3 + 1.5 IQR), 'p95', or 'none'."""
    times = times.copy()
    if method == "tukey":
        q1, q3 = times["t_raw"].quantile([0.25, 0.75])
        thresh = q3 + 1.5 * (q3 - q1)
    elif method == "p95":
        thresh = times["t_raw"].quantile(0.95)
    else:
        thresh = np.inf
    times["is_outlier"] = times["t_raw"] > thresh
    times["t"] = times["t_raw"].clip(upper=thresh)
    times.attrs["cap_method"] = method
    times.attrs["cap_threshold"] = float(thresh)
    return times


def timing_summary(times: pd.DataFrame) -> pd.DataFrame:
    """Mean/std/median decision time per annotator × category, with ALL rows."""
    agg = {"n": ("t", "count"), "mean_s": ("t", "mean"),
           "std_s": ("t", "std"), "median_s": ("t", "median")}
    by_cat_ann = times.groupby(["annotator", "category"]).agg(**agg).round(1).reset_index()
    by_ann = (times.groupby("annotator").agg(**agg).round(1)
              .reset_index().assign(category="ALL"))
    overall = pd.DataFrame([{
        "annotator": "ALL", "category": "ALL", "n": times["t"].count(),
        "mean_s": round(times["t"].mean(), 1),
        "std_s": round(times["t"].std(), 1),
        "median_s": round(times["t"].median(), 1),
    }])
    cols = ["annotator", "category", "n", "mean_s", "std_s", "median_s"]
    return pd.concat([by_cat_ann, by_ann, overall], ignore_index=True)[cols]


def timing_matrix(times: pd.DataFrame,
                  categories: tuple[str, ...] = ("PERS", "ORG", "LOC")) -> pd.DataFrame:
    """Mean decision time pivot (annotator × category) with ALL margins."""
    base = times.pivot_table(index="annotator", columns="category",
                             values="t", aggfunc="mean")
    cats = [c for c in categories if c in base.columns]
    matrix = base[cats].copy()
    matrix["ALL"] = times.groupby("annotator")["t"].mean()
    row = times.groupby("category")["t"].mean().reindex(cats)
    row["ALL"] = times["t"].mean()
    matrix.loc["ALL"] = row
    return matrix
