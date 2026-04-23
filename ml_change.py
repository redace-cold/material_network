# -*- coding: utf-8 -*-
"""
ML pipeline for BNT-KNN-xSBCT dielectric ceramics.

Dual regression targets:
    1. Wrec  (J/cm^3)   - recoverable energy density
    2. eta   (%)         - efficiency

Pipeline (following AFM 2026, Sun et al.):
    Step 1  Build ~30 weighted-average descriptors + 6 derived structural
            parameters (tolerance factor, octahedral factor, radius variance,
            configurational entropy, electronegativity mismatch, valence
            mismatch).  Occupancy fractions are excluded because they are
            collinear with X inside the main composition series.
    Step 2  Pearson correlation matrix on all descriptors (Figure 1d).
    Step 3  |r| > 0.9 redundancy pruning (keeps representative from each
            correlated cluster) -> ~20 descriptors.
    Step 4  Aggregated feature-importance ranking from 5 tree models on the
            pruned descriptor set -> top 10 (Figure 1e).
    Step 5  Train 7 regressors on the top-10 feature set with repeated
            k-fold CV, rank by Test MAE (Figure 1f).
    Step 6  Best model -> OOF predictions of Wrec and eta; Pareto front +
            combined score for candidate ranking.

Output scope splits:
    main_system  BNT-KNN-xSBCT series (X as a meaningful feature)
    all_data     All rows (X excluded; many occupancy-fraction descriptors
                 re-enabled because they now vary between systems)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


# ---------------------------------------------------------------------------
# Paths & basic config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
LOCAL_XLSX = BASE_DIR / "ML-4-数据收集.xlsx"
INPUT_XLSX = (
    LOCAL_XLSX
    if LOCAL_XLSX.exists()
    else Path(r"D:\wechat_file\xwechat_files\wxid_lmtmmookg5bg11_c185\msg\file\2026-04\ML-4-数据收集.xlsx")
)
SHEET_NAME = "ML描述符"
OUT_DIR = BASE_DIR / "output"

RANDOM_STATE = 42
N_SPLITS = 10          # AFM uses 10-fold
N_REPEATS = 5          # 5 repeats -> 50 folds per model, good for mean ± std

# Dual regression targets - Wrec + eta (W/E dropped as a target; see README)
TARGETS = {
    "Wrec": "Wrec(J/cm3)",
    "eta":  "效率η（%）",
}
TARGET_PLOT_LABELS = {
    "Wrec": r"$W_{rec}$ (J/cm$^3$)",
    "eta":  r"$\eta$ (%)",
}

# Columns that are never features
META_COLS = {"原表行号", "序号", "体系", "计算状态", "计算说明"}
PERFORMANCE_COLS = {"Wrec(J/cm3)", "Eb(kV/cm)", "W/E", "效率η（%）"}

# Occupancy-fraction descriptor prefixes (excluded from the main_system scope
# because they are collinear with X inside the fixed parent formula).
OCC_FRAC_PREFIXES = ("A位分数_", "B位分数_")

# Main composition series markers
MAIN_SYSTEM_MARKERS = (
    "0.7Bi0.5Na0.5TiO3-0.3K0.5Na0.5NbO3",
    "Sr0.7Bi0.2Ca0.1TiO3",
)

# Redundancy threshold. AFM uses |r| > 0.8; we use 0.85 as a compromise that
# keeps derived structural descriptors (t, mu, sigmaA_r ...) distinguishable
# while still dropping trivially duplicated pairs.
REDUNDANCY_THRESHOLD = 0.85
TOP_N_FEATURES = 10

# Model roster
FEATURE_IMPORTANCE_MODELS = [
    "RandomForest", "DecisionTree", "GradientBoosting", "ExtraTrees", "AdaBoost",
]
FEATURE_IMPORTANCE_LABELS = {
    "RandomForest":     "RF",
    "DecisionTree":     "DT",
    "GradientBoosting": "GB",
    "ExtraTrees":       "XGB*",
    "AdaBoost":         "ADAB",
}
FEATURE_IMPORTANCE_COLORS = {
    "RF":   "#1f77b4",
    "DT":   "#ff7f0e",
    "GB":   "#2ca02c",
    "XGB*": "#d62728",
    "ADAB": "#9467bd",
}

ALL_MODEL_NAMES = [
    "RandomForest", "DecisionTree", "SVR", "GradientBoosting",
    "ExtraTrees", "AdaBoost", "Ridge",
]
ALL_MODEL_LABELS = {
    "RandomForest":     "RandomForest",
    "DecisionTree":     "DecisionTree",
    "SVR":              "SVR",
    "GradientBoosting": "GradientBoosting",
    "ExtraTrees":       "XGBoost*",
    "AdaBoost":         "AdaBoost",
    "Ridge":            "Ridge",
}


# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.weight":        "bold",
    "axes.labelweight":   "bold",
    "axes.titleweight":   "bold",
    "axes.linewidth":     1.2,
    "axes.facecolor":     "white",
    "figure.facecolor":   "white",
    "savefig.facecolor":  "white",
    "legend.frameon":     False,
    "mathtext.default":   "regular",
    "axes.unicode_minus": False,
    "axes.grid":          False,
})


@dataclass(frozen=True)
class ScopeConfig:
    name: str
    include_x: bool
    include_occupancy_fractions: bool
    note: str


# ---------------------------------------------------------------------------
# Feature aliasing
# ---------------------------------------------------------------------------
# Short aliases for physical descriptors so axis labels stay readable.
FEATURE_ALIAS_DIRECT = {
    "X":                           "X",
    "A位原始总量":                  "A_sum",
    "B位原始总量":                  "B_sum",
    "A位平均Shannon半径(Å)":        "A_r",
    "B位平均Shannon半径(Å)":        "B_r",
    "A/B平均半径比":                "rA_rB",
    "A-B平均半径差(Å)":             "dr_AB",
    "容忍因子 t":                   "t",
    "八面体因子 μ":                 "mu",
    # A/B paired descriptors. Aliases are 'A_*' / 'B_*' with identical
    # suffix so the Figure 1d legacy view can pair them automatically.
    "A位半径方差(Å^2)":             "A_rvar",
    "B位半径方差(Å^2)":             "B_rvar",
    "A位半径失配 δr":               "A_drm",
    "B位半径失配 δr":               "B_drm",
    "A位构型熵 ΔS_A (J/mol·K)":     "A_S",
    "B位构型熵 ΔS_B (J/mol·K)":     "B_S",
    "总构型熵 ΔS_total (J/mol·K)":  "S_tot",
    "A位平均Pauling电负性":         "A_chi",
    "B位平均Pauling电负性":         "B_chi",
    "Δχ(A-O)":                     "dchi_AO",
    "Δχ(B-O)":                     "dchi_BO",
    "Δχ(A-B)":                     "dchi_AB",
    "A位平均名义价态":              "A_v",
    "B位平均名义价态":              "B_v",
    "A位价态失配 σV_A":             "A_vmis",
    "B位价态失配 σV_B":             "B_vmis",
    "总价态失配 σV_total":          "vmis_tot",
    "A位平均相对原子质量":          "A_M",
    "B位平均相对原子质量":          "B_M",
    "A位平均极化率(bohr^3)":        "A_alpha",
    "B位平均极化率(bohr^3)":        "B_alpha",
    "A位平均第一电离能(eV)":        "A_IE1",
    "B位平均第一电离能(eV)":        "B_IE1",
}


def feature_alias(feature: str, fallback_idx: int | None = None) -> str:
    if feature in FEATURE_ALIAS_DIRECT:
        return FEATURE_ALIAS_DIRECT[feature]
    for prefix, short in (("A位分数_", "fA_"), ("B位分数_", "fB_")):
        if feature.startswith(prefix):
            return f"{short}{feature[len(prefix):]}"
    if fallback_idx is not None:
        return f"D{fallback_idx:02d}"
    return feature


def alias_frame(feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    used: set[str] = set()
    for idx, feature in enumerate(feature_cols, start=1):
        alias = feature_alias(feature, idx)
        base, k = alias, 2
        while alias in used:
            alias, k = f"{base}_{k}", k + 1
        used.add(alias)
        rows.append({
            "Feature":         feature,
            "Feature_Alias":   alias,
            "Descriptor_Code": f"D{idx:02d}",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def _base_regressor(model_name: str, random_state: int) -> Pipeline:
    """Returns a Pipeline (imputer + optional scaler + estimator). Params are
    tuned toward the small-sample regime (~100-150 rows, 10 features)."""
    imp = ("imputer", SimpleImputer(strategy="median"))

    if model_name == "Ridge":
        return Pipeline([imp, ("scaler", StandardScaler()),
                         ("model", Ridge(alpha=1.0, random_state=random_state))])

    if model_name == "SVR":
        return Pipeline([imp, ("scaler", StandardScaler()),
                         ("model", SVR(C=5.0, gamma="scale", epsilon=0.05))])

    if model_name == "RandomForest":
        return Pipeline([imp, ("model", RandomForestRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=4,
            max_features="sqrt", random_state=random_state, n_jobs=1))])

    if model_name == "DecisionTree":
        return Pipeline([imp, ("model", DecisionTreeRegressor(
            max_depth=5, min_samples_leaf=4, random_state=random_state))])

    if model_name == "ExtraTrees":
        return Pipeline([imp, ("model", ExtraTreesRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=4,
            max_features="sqrt", random_state=random_state, n_jobs=1))])

    if model_name == "GradientBoosting":
        return Pipeline([imp, ("model", GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.04, max_depth=3,
            min_samples_leaf=4, subsample=0.8, random_state=random_state))])

    if model_name == "AdaBoost":
        return Pipeline([imp, ("model", AdaBoostRegressor(
            estimator=DecisionTreeRegressor(
                max_depth=3, min_samples_leaf=4, random_state=random_state),
            n_estimators=120, learning_rate=0.05, random_state=random_state))])

    raise ValueError(f"Unknown model: {model_name}")


def make_model(model_name: str, random_state: int = RANDOM_STATE):
    """Wraps the base regressor with target standardization so one config
    works for both Wrec and eta, which live on very different scales."""
    return TransformedTargetRegressor(
        regressor=_base_regressor(model_name, random_state),
        transformer=StandardScaler(),
    )


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------
def initial_feature_columns(df: pd.DataFrame, config: ScopeConfig) -> list[str]:
    """All descriptor columns, respecting the scope config."""
    excluded = set(META_COLS) | set(PERFORMANCE_COLS)
    if not config.include_x:
        excluded.add("X")

    cols: list[str] = []
    for c in df.columns:
        if c in excluded:
            continue
        if not config.include_occupancy_fractions and c.startswith(OCC_FRAC_PREFIXES):
            continue
        values = pd.to_numeric(df[c], errors="coerce")
        # Drop constant columns - they carry no information and break corr().
        if values.notna().sum() < 3 or values.std(ddof=0) < 1e-10:
            continue
        cols.append(c)
    return cols


def pearson_prune(X: pd.DataFrame, y: np.ndarray,
                  threshold: float = REDUNDANCY_THRESHOLD) -> list[str]:
    """Greedy redundancy pruning. Within each |r|>threshold cluster, keep the
    feature with the highest |Pearson r| to the target; drop the rest.

    This mirrors the AFM procedure: 'one representative feature from each
    correlated group is retained'.
    """
    corr = X.corr().abs().fillna(0.0)
    y_corr = pd.Series(
        {c: abs(np.corrcoef(
            X[c].fillna(X[c].median()), y)[0, 1]) if X[c].std() > 0 else 0.0
         for c in X.columns}
    ).fillna(0.0)

    # Rank features by strength of linear association with y (descending).
    ranked = y_corr.sort_values(ascending=False).index.tolist()

    kept: list[str] = []
    dropped: set[str] = set()
    for feat in ranked:
        if feat in dropped:
            continue
        # Everything above the threshold against this feature is redundant.
        for other in ranked:
            if other == feat or other in dropped or other in kept:
                continue
            if corr.loc[feat, other] >= threshold:
                dropped.add(other)
        kept.append(feat)
    return kept


def aggregated_feature_importance(
    X: pd.DataFrame, y: np.ndarray, feature_cols: list[str],
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Per-model normalized importances + an aggregated total. Works on the
    pruned feature set only.
    """
    rows = alias_frame(feature_cols).copy()
    for model_name in FEATURE_IMPORTANCE_MODELS:
        pipe = _base_regressor(model_name, random_state)
        pipe.fit(X[feature_cols], y)
        est = pipe.named_steps["model"]
        imp = np.asarray(getattr(est, "feature_importances_",
                                 np.zeros(len(feature_cols))), dtype=float)
        if len(imp) != len(feature_cols):
            imp = np.zeros(len(feature_cols), dtype=float)
        imp = np.clip(imp, 0, None)
        # Per-model min-max normalization so a single dominant tree does not
        # drown out the others in the stacked plot.
        if imp.max() > 0:
            imp = imp / imp.max()
        rows[FEATURE_IMPORTANCE_LABELS[model_name]] = imp

    importance_cols = [FEATURE_IMPORTANCE_LABELS[m]
                       for m in FEATURE_IMPORTANCE_MODELS]
    rows["Total_Importance"] = rows[importance_cols].sum(axis=1)
    return rows.sort_values("Total_Importance", ascending=False).reset_index(drop=True)




# ---------------------------------------------------------------------------
# Repeated k-fold CV
# ---------------------------------------------------------------------------
def repeated_cv(model, X: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    cv = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    pred_sum = np.zeros(len(y), dtype=float)
    pred_cnt = np.zeros(len(y), dtype=float)
    rows = []
    for fold_id, (tr, te) in enumerate(cv.split(X), start=1):
        m = clone(model)
        m.fit(X.iloc[tr], y[tr])
        tr_pred = m.predict(X.iloc[tr])
        te_pred = m.predict(X.iloc[te])
        pred_sum[te] += te_pred
        pred_cnt[te] += 1
        rows.append({
            "Fold":     fold_id,
            "Train_MAE": mean_absolute_error(y[tr], tr_pred),
            "Test_MAE":  mean_absolute_error(y[te], te_pred),
            "Train_R2":  r2_score(y[tr], tr_pred),
            "Test_R2":   r2_score(y[te], te_pred),
        })
    oof = pred_sum / np.maximum(pred_cnt, 1)
    return pd.DataFrame(rows), oof


def summarize(fold_df: pd.DataFrame, target_key: str, target_col: str,
              model_name: str, y: np.ndarray) -> dict:
    y_range = float(np.nanmax(y) - np.nanmin(y))
    def _mean_std(s: pd.Series) -> tuple[float, float]:
        return float(s.mean()), float(s.std(ddof=1))

    tr_mae_m, tr_mae_s = _mean_std(fold_df["Train_MAE"])
    te_mae_m, te_mae_s = _mean_std(fold_df["Test_MAE"])
    tr_r2_m,  tr_r2_s  = _mean_std(fold_df["Train_R2"])
    te_r2_m,  te_r2_s  = _mean_std(fold_df["Test_R2"])

    norm = (lambda v: v / y_range) if y_range > 0 else (lambda v: np.nan)
    return {
        "Target_Key":               target_key,
        "Target_Column":            target_col,
        "Model":                    model_name,
        "Target_Range":             y_range,
        "Train_MAE_mean":           tr_mae_m,
        "Train_MAE_std":            tr_mae_s,
        "Train_Normalized_MAE":     norm(tr_mae_m),
        "Train_Normalized_MAE_std": norm(tr_mae_s),
        "Test_MAE_mean":            te_mae_m,
        "Test_MAE_std":             te_mae_s,
        "Test_Normalized_MAE":      norm(te_mae_m),
        "Test_Normalized_MAE_std":  norm(te_mae_s),
        "Train_R2_mean":            tr_r2_m,
        "Train_R2_std":             tr_r2_s,
        "Test_R2_mean":             te_r2_m,
        "Test_R2_std":              te_r2_s,
    }


# ---------------------------------------------------------------------------
# Pareto helpers
# ---------------------------------------------------------------------------
def minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-15:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def pareto_maximize(values: np.ndarray) -> np.ndarray:
    """Return a boolean mask of Pareto-optimal rows (maximization)."""
    finite = np.isfinite(values).all(axis=1)
    flags = np.zeros(values.shape[0], dtype=bool)
    if not finite.any():
        return flags
    idx = np.where(finite)[0]
    v = values[finite]
    for i, row in enumerate(v):
        dominated = np.any(
            np.all(v >= row, axis=1) & np.any(v > row, axis=1)
        )
        flags[idx[i]] = not dominated
    return flags


# ---------------------------------------------------------------------------
# Plotting primitives
# ---------------------------------------------------------------------------
def _style_axes(ax):
    ax.set_facecolor("white")
    ax.grid(False)
    ax.minorticks_off()
    ax.tick_params(axis="both", width=1.1, direction="out", labelsize=9)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    for sp in ax.spines.values():
        sp.set_linewidth(1.2)
        sp.set_color("black")


def _style_colorbar(cb):
    cb.outline.set_linewidth(1.0)
    cb.outline.set_edgecolor("black")
    cb.ax.grid(False)
    cb.ax.tick_params(labelsize=8.5, width=1.0, direction="out")
    for lbl in cb.ax.get_yticklabels() + cb.ax.get_xticklabels():
        lbl.set_fontweight("bold")


# ---------------------------------------------------------------------------
# Figure 1d - Pearson correlation heatmap
# ---------------------------------------------------------------------------
def plot_figure1d(X: pd.DataFrame, alias_df: pd.DataFrame, out_path: Path):
    labels = alias_df["Descriptor_Code"].tolist()
    corr = X.corr()
    # Any column that was constant within this scope has already been
    # excluded upstream, so there should be no all-NaN rows here. Still,
    # guard against the rare case with np.where.
    corr_vals = np.where(np.isnan(corr.values), 0.0, corr.values)

    fig, ax = plt.subplots(figsize=(10.5, 9.2))
    # RdBu_r is diverging and centered at zero -> low |r| reads as pale,
    # high |+r| red, high |-r| blue. Much more informative than viridis here.
    im = ax.imshow(corr_vals, vmin=-1, vmax=1, cmap="RdBu_r",
                   aspect="equal", interpolation="nearest")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right",
                       rotation_mode="anchor", fontsize=6.5, fontweight="bold")
    ax.set_yticklabels(labels, rotation=45, ha="right",
                       rotation_mode="anchor", va="center",
                       fontsize=6.5, fontweight="bold")
    ax.tick_params(axis="both", length=0, pad=2)

    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Pearson r", fontweight="bold", fontsize=10)
    _style_axes(ax)
    _style_colorbar(cb)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_figure1d_legacy(X: pd.DataFrame, alias_df: pd.DataFrame, out_path: Path):
    """AFM-style 50-descriptor Pearson heatmap.

    Reconstructs the A/B-site split view used in AFM 2026 (Sun et al.):
    every paired descriptor (A-site weighted average + B-site weighted
    average of the same ion property) is shown as two adjacent columns
    'A_xxx' / 'B_xxx'. This produces the signature checkerboard / mosaic
    pattern where A_x and B_x are typically strongly anti-correlated
    (one goes up as the other goes down along the composition axis),
    giving the AFM-style visual look.

    Uses the viridis colormap to match the original style.
    """
    # Reorder columns so that each A_xxx sits next to its B_xxx partner.
    # We pair features whose aliases share everything except the A_/B_
    # prefix. Any feature without a partner is appended at the end.
    alias_map = dict(zip(alias_df["Feature"], alias_df["Feature_Alias"]))
    cols = list(X.columns)

    paired_order: list[str] = []
    used: set[str] = set()
    for c in cols:
        if c in used:
            continue
        alias = alias_map.get(c, c)
        partner_alias = None
        if alias.startswith("A_"):
            partner_alias = "B_" + alias[2:]
        elif alias.startswith("B_"):
            partner_alias = "A_" + alias[2:]
        partner = None
        if partner_alias is not None:
            for other in cols:
                if other in used or other == c:
                    continue
                if alias_map.get(other, other) == partner_alias:
                    partner = other
                    break
        # Always put A first, then B
        if alias.startswith("B_") and partner is not None:
            paired_order.extend([partner, c])
        else:
            paired_order.append(c)
            if partner is not None:
                paired_order.append(partner)
        used.add(c)
        if partner is not None:
            used.add(partner)

    X_ord = X[paired_order]
    labels = [alias_map.get(c, c) for c in paired_order]
    corr = X_ord.corr()
    corr_vals = np.where(np.isnan(corr.values), 0.0, corr.values)

    n = len(labels)
    side = max(10.0, n * 0.22)
    fig, ax = plt.subplots(figsize=(side, side * 0.92))
    # viridis preserves the original AFM-style aesthetic - yellow at r=+1,
    # purple at r=-1, teal in the low-|r| middle region.
    im = ax.imshow(corr_vals, vmin=-1, vmax=1, cmap="viridis",
                   aspect="equal", interpolation="nearest")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    tick_fs = float(np.clip(260.0 / max(n, 1), 4.5, 9.0))
    ax.set_xticklabels(labels, rotation=90, ha="center",
                       fontsize=tick_fs, fontweight="bold")
    ax.set_yticklabels(labels, rotation=0, va="center",
                       fontsize=tick_fs, fontweight="bold")
    ax.tick_params(axis="both", length=0, pad=2)

    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Pearson r", fontweight="bold", fontsize=10)
    _style_axes(ax)
    _style_colorbar(cb)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1e - Aggregated feature importance (stacked-bar, AFM-style)
# ---------------------------------------------------------------------------
def plot_figure1e(imp_df: pd.DataFrame, out_path: Path, top_n: int = TOP_N_FEATURES):
    top = imp_df.head(top_n).copy()
    model_labels = [FEATURE_IMPORTANCE_LABELS[m] for m in FEATURE_IMPORTANCE_MODELS]

    fig, ax = plt.subplots(figsize=(9.5, 5.0))

    x = np.arange(len(top))
    bottom = np.zeros(len(top), dtype=float)
    for lbl in model_labels:
        if lbl not in top.columns:
            continue
        vals = top[lbl].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, label=lbl,
               color=FEATURE_IMPORTANCE_COLORS[lbl],
               edgecolor="black", linewidth=0.3, width=0.72)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(top["Feature_Alias"].tolist(),
                       rotation=40, ha="right", rotation_mode="anchor",
                       fontsize=10, fontweight="bold")
    ax.set_ylabel("Aggregated importance", fontweight="bold", fontsize=11)
    ax.set_ylim(0, max(bottom.max() * 1.12, 0.1))

    ax.legend(ncol=3, fontsize=9, loc="upper right",
              frameon=True, framealpha=0.85, edgecolor="black")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1f - Train vs Test normalized MAE with error bars
# ---------------------------------------------------------------------------
def plot_figure1f(summary: pd.DataFrame, target_key: str, out_path: Path):
    sub = summary[summary["Target_Key"] == target_key].copy()
    # Rank models by Test MAE (ascending) so the best model sits on the left.
    sub = sub.sort_values(["Test_Normalized_MAE", "Test_Normalized_MAE_std"])
    sub = sub.reset_index(drop=True)

    x = np.arange(len(sub))
    width = 0.38
    labels = [ALL_MODEL_LABELS[m] for m in sub["Model"]]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - width / 2, sub["Train_Normalized_MAE"], width,
           yerr=sub["Train_Normalized_MAE_std"],
           label="Train", color="#1f77b4",
           edgecolor="black", linewidth=0.4,
           capsize=3, error_kw={"elinewidth": 0.9, "ecolor": "black"})
    ax.bar(x + width / 2, sub["Test_Normalized_MAE"], width,
           yerr=sub["Test_Normalized_MAE_std"],
           label="Test", color="#ff7f0e",
           edgecolor="black", linewidth=0.4,
           capsize=3, error_kw={"elinewidth": 0.9, "ecolor": "black"})

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right",
                       rotation_mode="anchor", fontsize=10, fontweight="bold")
    ax.set_ylabel("Normalized MAE (mean ± std)", fontweight="bold", fontsize=11)
    ymax_candidates = [
        (sub["Train_Normalized_MAE"] + sub["Train_Normalized_MAE_std"]).max(),
        (sub["Test_Normalized_MAE"]  + sub["Test_Normalized_MAE_std"]).max(),
    ]
    ymax = max([v for v in ymax_candidates if np.isfinite(v)], default=0.1)
    ax.set_ylim(0, ymax * 1.15)

    ax.legend(loc="upper right", fontsize=10,
              frameon=True, framealpha=0.85, edgecolor="black")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1def combined panel (d | e | f)
# ---------------------------------------------------------------------------
def plot_figure1_combined(X: pd.DataFrame, alias_df: pd.DataFrame,
                          imp_df: pd.DataFrame, summary: pd.DataFrame,
                          target_key: str, out_path: Path,
                          top_n: int = TOP_N_FEATURES):
    fig = plt.figure(figsize=(18.5, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.0, 1.0], wspace=0.30)

    # --- (d) Pearson ---
    ax_d = fig.add_subplot(gs[0, 0])
    labels = alias_df["Descriptor_Code"].tolist()
    corr = X.corr()
    corr_vals = np.where(np.isnan(corr.values), 0.0, corr.values)
    im = ax_d.imshow(corr_vals, vmin=-1, vmax=1, cmap="RdBu_r",
                     aspect="equal", interpolation="nearest")
    ax_d.set_xticks(np.arange(len(labels)))
    ax_d.set_yticks(np.arange(len(labels)))
    ax_d.set_xticklabels(labels, rotation=45, ha="right",
                         rotation_mode="anchor", fontsize=5.5, fontweight="bold")
    ax_d.set_yticklabels(labels, rotation=45, ha="right",
                         rotation_mode="anchor", va="center",
                         fontsize=5.5, fontweight="bold")
    ax_d.tick_params(axis="both", length=0, pad=2)
    cb = fig.colorbar(im, ax=ax_d, fraction=0.040, pad=0.02)
    cb.set_label("Pearson r", fontweight="bold", fontsize=9)
    _style_axes(ax_d)
    _style_colorbar(cb)

    # --- (e) importance ---
    ax_e = fig.add_subplot(gs[0, 1])
    top = imp_df.head(top_n).copy()
    x = np.arange(len(top))
    bottom = np.zeros(len(top), dtype=float)
    for lbl in [FEATURE_IMPORTANCE_LABELS[m] for m in FEATURE_IMPORTANCE_MODELS]:
        if lbl not in top.columns:
            continue
        vals = top[lbl].to_numpy(dtype=float)
        ax_e.bar(x, vals, bottom=bottom, label=lbl,
                 color=FEATURE_IMPORTANCE_COLORS[lbl],
                 edgecolor="black", linewidth=0.3, width=0.72)
        bottom += vals
    ax_e.set_xticks(x)
    ax_e.set_xticklabels(top["Feature_Alias"].tolist(),
                         rotation=40, ha="right", rotation_mode="anchor",
                         fontsize=8.5, fontweight="bold")
    ax_e.set_ylabel("Aggregated importance", fontweight="bold", fontsize=10)
    ax_e.set_ylim(0, max(bottom.max() * 1.12, 0.1))
    ax_e.legend(ncol=3, fontsize=8, loc="upper right",
                frameon=True, framealpha=0.85, edgecolor="black")
    _style_axes(ax_e)

    # --- (f) model selection ---
    ax_f = fig.add_subplot(gs[0, 2])
    sub = summary[summary["Target_Key"] == target_key].copy()
    sub = sub.sort_values(["Test_Normalized_MAE", "Test_Normalized_MAE_std"]).reset_index(drop=True)
    xf = np.arange(len(sub))
    width = 0.38
    ax_f.bar(xf - width / 2, sub["Train_Normalized_MAE"], width,
             yerr=sub["Train_Normalized_MAE_std"],
             label="Train", color="#1f77b4",
             edgecolor="black", linewidth=0.4,
             capsize=3, error_kw={"elinewidth": 0.9, "ecolor": "black"})
    ax_f.bar(xf + width / 2, sub["Test_Normalized_MAE"], width,
             yerr=sub["Test_Normalized_MAE_std"],
             label="Test", color="#ff7f0e",
             edgecolor="black", linewidth=0.4,
             capsize=3, error_kw={"elinewidth": 0.9, "ecolor": "black"})
    ax_f.set_xticks(xf)
    ax_f.set_xticklabels([ALL_MODEL_LABELS[m] for m in sub["Model"]],
                         rotation=30, ha="right", rotation_mode="anchor",
                         fontsize=8.5, fontweight="bold")
    ax_f.set_ylabel("Normalized MAE", fontweight="bold", fontsize=10)
    ymax_vals = [
        (sub["Train_Normalized_MAE"] + sub["Train_Normalized_MAE_std"]).max(),
        (sub["Test_Normalized_MAE"]  + sub["Test_Normalized_MAE_std"]).max(),
    ]
    ymax = max([v for v in ymax_vals if np.isfinite(v)], default=0.1)
    ax_f.set_ylim(0, ymax * 1.15)
    ax_f.legend(loc="upper right", fontsize=9,
                frameon=True, framealpha=0.85, edgecolor="black")
    _style_axes(ax_f)

    # Panel labels (d) (e) (f) - lower-case, bold, upper-left
    for label_txt, ax in zip(("(d)", "(e)", "(f)"), (ax_d, ax_e, ax_f)):
        ax.text(0.01, 0.99, label_txt, transform=ax.transAxes,
                ha="left", va="top", fontsize=12, fontweight="bold")

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)




# ---------------------------------------------------------------------------
# Predicted vs Measured (per target) - no title, y=x line, R^2 + MAE box
# ---------------------------------------------------------------------------
def plot_predicted_vs_measured(scored: pd.DataFrame, target_key: str,
                               out_path: Path):
    target_col = TARGETS[target_key]
    y_true = scored[target_col].astype(float).to_numpy()
    y_pred = scored[f"OOF_Pred_{target_key}"].astype(float).to_numpy()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
    mae = mean_absolute_error(y_true, y_pred) if len(y_true) > 0 else np.nan

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(y_true, y_pred, s=36, c="#1f77b4",
               edgecolor="black", linewidth=0.4, alpha=0.85,
               label="Samples")

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    lims = [lo - pad, hi + pad]
    ax.plot(lims, lims, color="red", linewidth=1.5, label="y = x")
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)

    ax.set_xlabel(f"Measured {TARGET_PLOT_LABELS[target_key]}",
                  fontweight="bold", fontsize=11)
    ax.set_ylabel(f"Predicted {TARGET_PLOT_LABELS[target_key]}",
                  fontweight="bold", fontsize=11)

    txt = f"$R^2$ = {r2:.3f}\nMAE = {mae:.3f}"
    ax.text(0.04, 0.96, txt, transform=ax.transAxes,
            ha="left", va="top", fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35",
                      facecolor="white", edgecolor="black", linewidth=1.0))
    ax.legend(loc="lower right", fontsize=10,
              frameon=True, framealpha=0.85, edgecolor="black")
    _style_axes(ax)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pareto scatter for dual regression
# ---------------------------------------------------------------------------
def plot_dual_objective(scored: pd.DataFrame, out_path: Path):
    x = scored["OOF_Pred_Wrec"].astype(float).to_numpy()
    y = scored["OOF_Pred_eta"].astype(float).to_numpy()
    par = scored["Pareto"].astype(bool).to_numpy()

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(x[~par], y[~par], s=32, c="#7f7f7f",
               edgecolor="black", linewidth=0.3, alpha=0.7,
               label="Non-Pareto")
    ax.scatter(x[par], y[par], s=62, c="#d62728",
               edgecolor="black", linewidth=0.6, alpha=0.95,
               label="Pareto front", marker="D")

    ax.set_xlabel(f"Predicted {TARGET_PLOT_LABELS['Wrec']}",
                  fontweight="bold", fontsize=11)
    ax.set_ylabel(f"Predicted {TARGET_PLOT_LABELS['eta']}",
                  fontweight="bold", fontsize=11)
    ax.legend(loc="lower right", fontsize=10,
              frameon=True, framealpha=0.85, edgecolor="black")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Trend against X (main_system only)
# ---------------------------------------------------------------------------
def plot_x_trends(scored: pd.DataFrame, out_path: Path):
    if "X" not in scored.columns:
        return
    tmp = scored.sort_values("X").copy()
    x = tmp["X"].astype(float).to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for ax, key in zip(axes, ("Wrec", "eta")):
        col = TARGETS[key]
        meas = tmp[col].astype(float).to_numpy()
        pred = tmp[f"OOF_Pred_{key}"].astype(float).to_numpy()
        ax.scatter(x, meas, s=28, c="#1f77b4",
                   edgecolor="black", linewidth=0.3, alpha=0.85,
                   label="Measured")
        ax.plot(x, pred, color="#d62728", linewidth=1.5,
                label="Predicted (OOF)")
        ax.set_xlabel("X", fontweight="bold", fontsize=11)
        ax.set_ylabel(TARGET_PLOT_LABELS[key], fontweight="bold", fontsize=11)
        ax.legend(loc="best", fontsize=9,
                  frameon=True, framealpha=0.85, edgecolor="black")
        _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------
def save_table(df: pd.DataFrame, path_base: Path):
    df.to_csv(path_base.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    try:
        df.to_excel(path_base.with_suffix(".xlsx"), index=False)
    except ValueError:
        # openpyxl can choke on illegal characters in cells; CSV still works.
        pass


# ---------------------------------------------------------------------------
# Run one scope
# ---------------------------------------------------------------------------
def run_scope(df: pd.DataFrame, config: ScopeConfig,
              shared_top_features: list[str] | None = None,
              shared_importance_df: pd.DataFrame | None = None,
              shared_pruned_features: list[str] | None = None) -> dict:
    """Run the full pipeline for a scope.

    shared_top_features:
        If provided, use these features directly for model training
        instead of running Pearson prune + importance aggregation.
        This is needed for `main_system`, where every descriptor is a
        one-to-one function of X (|r| ~ 1 for all pairs), so in-scope
        feature selection is meaningless. We carry over the top features
        selected on `all_data`.

    shared_importance_df, shared_pruned_features:
        Auxiliary outputs from the scope that *did* perform feature
        selection. Used to draw the (e) panel consistently.
    """
    scope_dir = OUT_DIR / config.name
    scope_dir.mkdir(parents=True, exist_ok=True)

    work = df.copy()
    for c in ["X", *PERFORMANCE_COLS]:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")
    # Keep only rows with both primary targets present
    work = work.dropna(subset=list(TARGETS.values())).reset_index(drop=True)

    # ---- Step 1: initial descriptor set ----
    raw_features = initial_feature_columns(work, config)
    X_raw = work[raw_features].apply(pd.to_numeric, errors="coerce")
    alias_raw = alias_frame(raw_features)
    save_table(alias_raw.assign(Scope=config.name),
               scope_dir / "feature_columns_full")

    # Pearson matrix on the full descriptor set (Figure 1d)
    plot_figure1d(X_raw, alias_raw, scope_dir / "Figure1d_Pearson_full.png")
    # Legacy-style heatmap (viridis + feature-name labels) - this is the
    # visual style the AFM-derived pipeline originally produced. It is
    # informative when the descriptor set mixes occupancy fractions with
    # weighted-average physical descriptors.
    plot_figure1d_legacy(
        X_raw, alias_raw,
        scope_dir / "Figure1d_Pearson_full_legacy_viridis.png",
    )
    corr_full = X_raw.corr()
    corr_full.index = alias_raw["Feature_Alias"].tolist()
    corr_full.columns = alias_raw["Feature_Alias"].tolist()
    corr_full.to_csv(scope_dir / "Figure1d_Pearson_full_values.csv",
                     encoding="utf-8-sig")

    y_wrec = work[TARGETS["Wrec"]].astype(float).to_numpy()

    if shared_top_features is None:
        # ---- Step 2: Pearson-based redundancy pruning ----
        pruned = pearson_prune(X_raw, y_wrec, threshold=REDUNDANCY_THRESHOLD)
        pruned = pruned[:max(20, TOP_N_FEATURES + 5)]
        alias_pruned = alias_frame(pruned)
        save_table(alias_pruned.assign(Scope=config.name),
                   scope_dir / "feature_columns_pruned")
        X_pruned = X_raw[pruned]

        # ---- Step 3: aggregated feature importance on the pruned set ----
        imp_df = aggregated_feature_importance(X_pruned, y_wrec, pruned)
        save_table(imp_df, scope_dir / "Figure1e_feature_importance")
        plot_figure1e(imp_df, scope_dir / "Figure1e_feature_importance.png")

        # ---- Step 4: top-N features for model training ----
        top_feats = imp_df["Feature"].head(TOP_N_FEATURES).tolist()
    else:
        # Reuse features selected upstream. Filter to those that exist in
        # this scope's descriptor table (some may have been excluded, e.g.
        # occupancy fractions in main_system).
        top_feats = [f for f in shared_top_features if f in raw_features]
        if len(top_feats) < 3:
            raise RuntimeError(
                f"Only {len(top_feats)} of the shared top features are present "
                f"in scope '{config.name}'. Check column alignment.")

        if shared_importance_df is not None:
            imp_df = shared_importance_df.copy()
            imp_df["Source_Scope"] = "all_data (shared)"
            save_table(imp_df, scope_dir / "Figure1e_feature_importance_shared")
            plot_figure1e(imp_df, scope_dir / "Figure1e_feature_importance.png")
        else:
            imp_df = pd.DataFrame({"Feature": top_feats})

        if shared_pruned_features is not None:
            save_table(
                alias_frame(shared_pruned_features).assign(
                    Scope=f"{config.name} (shared from all_data)"),
                scope_dir / "feature_columns_pruned_shared",
            )

    alias_top = alias_frame(top_feats)
    save_table(alias_top.assign(Scope=config.name),
               scope_dir / "feature_columns_top10")
    X_top = X_raw[top_feats]

    # ---- Step 5: CV-based model comparison (per target) ----
    summary_rows: list[dict] = []
    fold_rows: list[pd.DataFrame] = []
    oof_store: dict[str, dict[str, np.ndarray]] = {}

    for tkey, tcol in TARGETS.items():
        y = work[tcol].astype(float).to_numpy()
        oof_store[tkey] = {}
        for model_name in ALL_MODEL_NAMES:
            model = make_model(model_name)
            fold_df, oof = repeated_cv(model, X_top, y)
            fold_df.insert(0, "Model", model_name)
            fold_df.insert(0, "Target_Column", tcol)
            fold_df.insert(0, "Target_Key", tkey)
            fold_rows.append(fold_df)
            oof_store[tkey][model_name] = oof
            summary_rows.append(
                summarize(fold_df, tkey, tcol, model_name, y)
            )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["Target_Key", "Test_MAE_mean", "Test_MAE_std"]).reset_index(drop=True)
    fold_level = pd.concat(fold_rows, ignore_index=True)
    save_table(summary,    scope_dir / "Figure1f_model_selection_summary")
    save_table(fold_level, scope_dir / "Figure1f_model_selection_folds")

    # Figure 1f - one per target
    plot_figure1f(summary, "Wrec", scope_dir / "Figure1f_model_selection_Wrec.png")
    plot_figure1f(summary, "eta",  scope_dir / "Figure1f_model_selection_eta.png")

    # Figure 1 combined (d|e|f) - use Wrec for the (f) panel by default
    plot_figure1_combined(
        X_raw, alias_raw, imp_df, summary,
        target_key="Wrec",
        out_path=scope_dir / "Figure1_def_combined.png",
    )

    # ---- Step 6: best model per target, build OOF predictions ----
    best_model_per_target: dict[str, str] = {}
    full_models: dict[str, object] = {}
    scored = work.copy()
    for tkey, tcol in TARGETS.items():
        per_target = summary[summary["Target_Key"] == tkey].sort_values(
            ["Test_MAE_mean", "Test_MAE_std"]).reset_index(drop=True)
        best_name = per_target.iloc[0]["Model"]
        best_model_per_target[tkey] = best_name

        scored[f"OOF_Pred_{tkey}"] = oof_store[tkey][best_name]
        y = work[tcol].astype(float).to_numpy()
        scored[f"OOF_Residual_{tkey}"] = y - oof_store[tkey][best_name]

        # Full-fit model for downstream prediction on unseen compositions.
        m = make_model(best_name)
        m.fit(X_top, y)
        full_models[tkey] = m
        scored[f"FullFit_Pred_{tkey}"] = m.predict(X_top)

    selected_df = pd.DataFrame([
        {"Target_Key":       t,
         "Target_Column":    TARGETS[t],
         "Selected_Model":   best_model_per_target[t],
         "Top_N_Features":   TOP_N_FEATURES,
         "CV_Test_MAE_mean": summary.loc[
            (summary.Target_Key == t) &
            (summary.Model == best_model_per_target[t]),
            "Test_MAE_mean"].iloc[0],
         "CV_Test_MAE_std":  summary.loc[
            (summary.Target_Key == t) &
            (summary.Model == best_model_per_target[t]),
            "Test_MAE_std"].iloc[0],
         "CV_Test_R2_mean":  summary.loc[
            (summary.Target_Key == t) &
            (summary.Model == best_model_per_target[t]),
            "Test_R2_mean"].iloc[0]}
        for t in TARGETS
    ])
    save_table(selected_df, scope_dir / "selected_models")

    # ---- Step 7: dual-objective scoring ----
    scored["OOF_Wrec_norm"] = minmax(scored["OOF_Pred_Wrec"])
    scored["OOF_eta_norm"]  = minmax(scored["OOF_Pred_eta"])
    scored["Measured_Wrec_norm"] = minmax(scored[TARGETS["Wrec"]].astype(float))
    scored["Measured_eta_norm"]  = minmax(scored[TARGETS["eta"]].astype(float))

    # 50/50 combined score - documented explicitly; not an objective truth.
    scored["OOF_Combined_Score"] = (
        0.5 * scored["OOF_Wrec_norm"] + 0.5 * scored["OOF_eta_norm"])
    scored["Measured_Combined_Score"] = (
        0.5 * scored["Measured_Wrec_norm"] + 0.5 * scored["Measured_eta_norm"])

    pareto_vals = scored[["OOF_Pred_Wrec", "OOF_Pred_eta"]].to_numpy()
    scored["Pareto"] = pareto_maximize(pareto_vals)
    scored = scored.sort_values("OOF_Combined_Score", ascending=False).reset_index(drop=True)

    pareto_candidates = scored[scored["Pareto"]].sort_values(
        "OOF_Combined_Score", ascending=False).reset_index(drop=True)
    top_candidates = scored.head(20).copy()
    best_predicted = scored.head(1).copy()
    best_measured = scored.sort_values(
        "Measured_Combined_Score", ascending=False).head(1).copy()

    save_table(scored,            scope_dir / "dual_target_scored_samples")
    save_table(pareto_candidates, scope_dir / "pareto_candidates")
    save_table(top_candidates,    scope_dir / "top20_combined_score_candidates")
    save_table(best_predicted,    scope_dir / "best_predicted_candidate")
    save_table(best_measured,     scope_dir / "best_measured_candidate")

    # Plot data backing the Wrec prediction plot
    wrec_plot_data = scored[
        ["序号", "体系", "X", TARGETS["Wrec"], "OOF_Pred_Wrec",
         "OOF_Residual_Wrec", TARGETS["eta"], "OOF_Pred_eta",
         "OOF_Residual_eta", "OOF_Combined_Score", "Pareto"]
    ].copy()
    save_table(wrec_plot_data, scope_dir / "Predicted_vs_Measured_data")

    # ---- Plots ----
    plot_predicted_vs_measured(
        scored, "Wrec", scope_dir / "Predicted_vs_Measured_Wrec.png")
    plot_predicted_vs_measured(
        scored, "eta",  scope_dir / "Predicted_vs_Measured_eta.png")
    plot_dual_objective(scored, scope_dir / "Dual_objective_Pareto.png")

    if config.include_x and "X" in scored.columns:
        plot_x_trends(scored, scope_dir / "X_trends_primary_targets.png")

    # ---- Summary text ----
    if shared_top_features is None:
        pruning_note = f"After Pearson pruning (|r|>{REDUNDANCY_THRESHOLD}): {len(pruned)}"
    else:
        pruning_note = (
            "Feature selection reused from 'all_data' scope. "
            "Pruning was skipped here because every descriptor is a monotone "
            "function of X inside this one-dimensional composition series."
        )
    lines = [
        f"Scope: {config.name}",
        f"Note: {config.note}",
        f"Samples: {len(work)}",
        f"Initial descriptors: {len(raw_features)}",
        pruning_note,
        f"Top-{TOP_N_FEATURES} features used for training:",
    ]
    for f in top_feats:
        lines.append(f"  - {f}  ({feature_alias(f)})")
    lines.append("")
    lines.append("Selected models:")
    for t in TARGETS:
        row = selected_df[selected_df.Target_Key == t].iloc[0]
        lines.append(
            f"  {t} ({TARGETS[t]}): {row['Selected_Model']}, "
            f"Test MAE = {row['CV_Test_MAE_mean']:.4g} ± {row['CV_Test_MAE_std']:.4g}, "
            f"Test R2 = {row['CV_Test_R2_mean']:.3f}")
    lines.extend(["",
        "Scope interpretation:",
        "  - 'Best predicted' / 'Best measured' refer to the best candidate",
        "    in the collected dataset, not a global optimum.",
        "  - Pareto front uses OOF-predicted Wrec and eta.",
        "  - Combined score = 0.5*Wrec_norm + 0.5*eta_norm (min-max scaling).",
    ])
    (scope_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    pruned_to_return = pruned if shared_top_features is None else shared_pruned_features
    return {
        "scope_dir":          scope_dir,
        "scored_xlsx":        scope_dir / "dual_target_scored_samples.xlsx",
        "top20_xlsx":         scope_dir / "top20_combined_score_candidates.xlsx",
        "pareto_xlsx":        scope_dir / "pareto_candidates.xlsx",
        "figure1_combined":   scope_dir / "Figure1_def_combined.png",
        # Objects passed to downstream scopes:
        "top_features":       top_feats,
        "importance_df":      imp_df,
        "pruned_features":    pruned_to_return,
    }


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
def main():
    if not INPUT_XLSX.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_XLSX}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_NAME)

    for c in PERFORMANCE_COLS | {"X"}:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    formula = df["体系"].astype(str)
    main_mask = np.ones(len(df), dtype=bool)
    for marker in MAIN_SYSTEM_MARKERS:
        main_mask &= formula.str.contains(marker, regex=False, na=False).to_numpy()

    # Two scopes:
    #   main_system  - the BNT-KNN-xSBCT continuous series. X is a proper
    #                  composition variable; occupancy fractions are
    #                  collinear with X and removed.
    #   all_data     - the whole dataset. X is meaningless across systems,
    #                  so it is excluded; occupancy fractions re-enabled
    #                  since they now carry real between-system information.
    scopes = [
        ScopeConfig(
            name="main_system",
            include_x=True,
            include_occupancy_fractions=False,
            note=("BNT-KNN-xSBCT series; physical descriptors only "
                  "(occupancy fractions dropped - collinear with X)."),
        ),
        ScopeConfig(
            name="all_data",
            include_x=False,
            include_occupancy_fractions=True,
            note=("All ML descriptor rows; X excluded (not universal); "
                  "occupancy fractions kept to distinguish systems."),
        ),
    ]

    outputs = {}
    # Run `all_data` first - this is where feature selection actually means
    # something (descriptors are not all collinear with a single variable).
    outputs["all_data"] = run_scope(
        df.reset_index(drop=True), scopes[1])

    # `main_system` then reuses the feature set selected on `all_data`.
    # Inside this one-dimensional composition series every descriptor is a
    # monotone function of X, so running the pruning + importance pipeline
    # here would be circular (|r| ~ 1 between any two descriptors, top10
    # would be arbitrary). By sharing features we get a fair CV on this
    # series under the same feature lens used for the cross-system model.
    outputs["main_system"] = run_scope(
        df.loc[main_mask].reset_index(drop=True),
        scopes[0],
        shared_top_features=outputs["all_data"]["top_features"],
        shared_importance_df=outputs["all_data"]["importance_df"],
        shared_pruned_features=outputs["all_data"].get("pruned_features"),
    )

    print("Done.")
    print(f"Output directory: {OUT_DIR.resolve()}")
    for scope_name, paths in outputs.items():
        print(f"[{scope_name}]")
        for k, v in paths.items():
            if isinstance(v, Path):
                print(f"  {k}: {v.resolve()}")


if __name__ == "__main__":
    main()