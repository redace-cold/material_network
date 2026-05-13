from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeRegressor


RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "ML-4-数据收集.xlsx"
OUTPUT_DIR = ROOT / "output" / "wrec_eta_dual_feature_selection"
OLD_TOP10_PATH = Path(
    r"D:\python_project\ml_change\output\all_data\feature_columns_top10.csv"
)

SHEET_NAME = "ML描述符"
WREC_COL = "Wrec(J/cm3)"
ETA_COL = "效率η（%）"
ETA_MIN = 50.0
APPLY_ETA_FILTER = True

REDUNDANCY_THRESHOLD = 0.85
MAX_PEARSON_FEATURES = 20
TOP_N_FEATURES = 10
WREC_WEIGHT = 0.70
ETA_WEIGHT = 0.30

METADATA_COLS = {
    "原表行号",
    "序号",
    "体系",
    "X",
    "计算状态",
    "计算说明",
}
PERFORMANCE_COLS = {
    "Wrec(J/cm3)",
    "Eb(kV/cm)",
    "W/E",
    "效率η（%）",
}


def make_models() -> dict[str, object]:
    return {
        "RF": RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=4,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "DT": DecisionTreeRegressor(
            max_depth=5,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
        ),
        "GB": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.04,
            max_depth=3,
            min_samples_leaf=4,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
        "XGB*": ExtraTreesRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=4,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "ADAB": AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=3, min_samples_leaf=4),
            n_estimators=120,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
        ),
    }


def safe_corr(x: pd.Series, y: np.ndarray) -> float:
    x = pd.to_numeric(x, errors="coerce")
    if x.nunique(dropna=True) <= 1:
        return 0.0
    filled = x.fillna(x.median())
    value = np.corrcoef(filled.to_numpy(dtype=float), y)[0, 1]
    if not np.isfinite(value):
        return 0.0
    return float(value)


def read_workbook() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing data file: {DATA_FILE}")

    df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
    df.columns = [str(c).strip() for c in df.columns]
    for col in (WREC_COL, ETA_COL):
        if col not in df.columns:
            raise ValueError(f"Missing required target column: {col}")

    df[WREC_COL] = pd.to_numeric(df[WREC_COL], errors="coerce")
    df[ETA_COL] = pd.to_numeric(df[ETA_COL], errors="coerce")
    df = df.dropna(subset=[WREC_COL, ETA_COL]).reset_index(drop=True)
    if APPLY_ETA_FILTER:
        df = df[df[ETA_COL] >= ETA_MIN].reset_index(drop=True)
    return df


def descriptor_columns(df: pd.DataFrame) -> list[str]:
    excluded = METADATA_COLS | PERFORMANCE_COLS
    cols: list[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() >= 3 and values.nunique(dropna=True) > 1:
            cols.append(col)
    return cols


def build_feature_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = df[cols].apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.dropna(axis=1, how="all")
    x = x.loc[:, x.nunique(dropna=True) > 1]
    return x.fillna(x.median(numeric_only=True))


def dual_target_pearson_prune(x: pd.DataFrame, y_wrec: np.ndarray, y_eta: np.ndarray) -> pd.DataFrame:
    rows = []
    for col in x.columns:
        r_wrec = abs(safe_corr(x[col], y_wrec))
        r_eta = abs(safe_corr(x[col], y_eta))
        rows.append(
            {
                "Feature": col,
                "Abs_r_Wrec": r_wrec,
                "Abs_r_eta": r_eta,
                "Dual_Target_Corr_Score": WREC_WEIGHT * r_wrec + ETA_WEIGHT * r_eta,
            }
        )

    score_df = pd.DataFrame(rows).sort_values(
        "Dual_Target_Corr_Score", ascending=False
    )
    corr = x[score_df["Feature"].tolist()].corr().abs().fillna(0.0)

    kept: list[str] = []
    dropped: set[str] = set()
    for feat in score_df["Feature"]:
        if feat in dropped:
            continue
        kept.append(feat)
        if len(kept) >= MAX_PEARSON_FEATURES:
            break
        redundant = corr.index[(corr.loc[feat] >= REDUNDANCY_THRESHOLD)].tolist()
        for other in redundant:
            if other != feat:
                dropped.add(other)

    score_df["Pearson_Selected"] = score_df["Feature"].isin(kept)
    score_df["Pearson_Rank"] = score_df["Feature"].map(
        {feature: i + 1 for i, feature in enumerate(kept)}
    )
    return score_df.sort_values(
        ["Pearson_Selected", "Pearson_Rank", "Dual_Target_Corr_Score"],
        ascending=[False, True, False],
    ).reset_index(drop=True)


def normalized_importances(x: pd.DataFrame, y: np.ndarray, prefix: str) -> pd.DataFrame:
    rows = pd.DataFrame({"Feature": x.columns})
    for name, model in make_models().items():
        estimator = clone(model)
        estimator.fit(x, y)
        values = np.asarray(getattr(estimator, "feature_importances_", np.zeros(x.shape[1])), dtype=float)
        values = np.clip(values, 0, None)
        if values.max() > 0:
            values = values / values.max()
        rows[f"{prefix}_{name}"] = values
    model_cols = [c for c in rows.columns if c.startswith(f"{prefix}_")]
    rows[f"{prefix}_Total"] = rows[model_cols].sum(axis=1)
    max_total = rows[f"{prefix}_Total"].max()
    rows[f"{prefix}_Norm"] = rows[f"{prefix}_Total"] / max_total if max_total > 0 else 0.0
    return rows


def dual_target_importance(x: pd.DataFrame, y_wrec: np.ndarray, y_eta: np.ndarray) -> pd.DataFrame:
    wrec_imp = normalized_importances(x, y_wrec, "Wrec")
    eta_imp = normalized_importances(x, y_eta, "eta")
    merged = wrec_imp.merge(eta_imp, on="Feature", how="inner")
    merged["Dual_Target_Importance"] = (
        WREC_WEIGHT * merged["Wrec_Norm"] + ETA_WEIGHT * merged["eta_Norm"]
    )
    return merged.sort_values("Dual_Target_Importance", ascending=False).reset_index(drop=True)


def load_old_top10() -> list[str]:
    if not OLD_TOP10_PATH.exists():
        return []
    old = pd.read_csv(OLD_TOP10_PATH)
    if "Feature" not in old.columns:
        return []
    return old["Feature"].dropna().astype(str).tolist()[:TOP_N_FEATURES]


def compare_top10(new_top10: list[str], old_top10: list[str]) -> pd.DataFrame:
    rows = []
    max_len = max(len(new_top10), len(old_top10), TOP_N_FEATURES)
    for i in range(max_len):
        new_feature = new_top10[i] if i < len(new_top10) else ""
        old_feature = old_top10[i] if i < len(old_top10) else ""
        rows.append(
            {
                "Rank": i + 1,
                "Dual_Target_Top10": new_feature,
                "Old_Wrec_Top10": old_feature,
                "Same_Rank": bool(new_feature and new_feature == old_feature),
                "Dual_Target_In_Old_Top10": bool(new_feature and new_feature in old_top10),
                "Old_In_Dual_Target_Top10": bool(old_feature and old_feature in new_top10),
            }
        )
    return pd.DataFrame(rows)


def save_outputs(
    df: pd.DataFrame,
    features: list[str],
    pearson_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"Feature": features}).to_csv(
        OUTPUT_DIR / "feature_columns_full.csv", index=False, encoding="utf-8-sig"
    )
    pearson_df.to_csv(
        OUTPUT_DIR / "dual_target_pearson_pruned_features.csv", index=False, encoding="utf-8-sig"
    )
    importance_df.to_csv(
        OUTPUT_DIR / "dual_target_feature_importance.csv", index=False, encoding="utf-8-sig"
    )
    importance_df.head(TOP_N_FEATURES).to_csv(
        OUTPUT_DIR / "dual_target_feature_columns_top10.csv", index=False, encoding="utf-8-sig"
    )
    comparison_df.to_csv(
        OUTPUT_DIR / "compare_with_old_wrec_top10.csv", index=False, encoding="utf-8-sig"
    )

    with pd.ExcelWriter(OUTPUT_DIR / "dual_target_feature_selection_results.xlsx") as writer:
        pd.DataFrame({"Feature": features}).to_excel(writer, sheet_name="full_features", index=False)
        pearson_df.to_excel(writer, sheet_name="pearson_pruning", index=False)
        importance_df.to_excel(writer, sheet_name="dual_importance", index=False)
        importance_df.head(TOP_N_FEATURES).to_excel(writer, sheet_name="dual_top10", index=False)
        comparison_df.to_excel(writer, sheet_name="compare_old_top10", index=False)

    old_common = int(comparison_df["Dual_Target_In_Old_Top10"].sum())
    notes = [
        "Dual-target feature selection for Wrec and eta",
        "",
        f"Data file: {DATA_FILE}",
        f"Sheet: {SHEET_NAME}",
        f"Rows used: {len(df)}",
        f"Initial numeric descriptor count: {len(features)}",
        f"Eta filter applied: {APPLY_ETA_FILTER}, eta >= {ETA_MIN}",
        f"Pearson redundancy threshold: |r| >= {REDUNDANCY_THRESHOLD}",
        f"Pearson retained feature count: {int(pearson_df['Pearson_Selected'].sum())}",
        f"Top feature count: {TOP_N_FEATURES}",
        f"Dual-target weighting: {WREC_WEIGHT:.2f}*Wrec + {ETA_WEIGHT:.2f}*eta",
        "",
        f"Overlap with old Wrec-based top10: {old_common}/{TOP_N_FEATURES}",
        "",
        "Important interpretation:",
        "The old ml_change Figure1e top10 was selected mainly against Wrec.",
        "This script selects features by combining Wrec and eta importances.",
        "Exact agreement is not required; differences indicate eta contributes",
        "additional feature preference in the auxiliary screening objective.",
    ]
    (OUTPUT_DIR / "dual_target_feature_selection_notes.txt").write_text(
        "\n".join(notes), encoding="utf-8"
    )


def main() -> None:
    df = read_workbook()
    features = descriptor_columns(df)
    x = build_feature_matrix(df, features)
    y_wrec = df[WREC_COL].to_numpy(dtype=float)
    y_eta = df[ETA_COL].to_numpy(dtype=float)

    pearson_df = dual_target_pearson_prune(x, y_wrec, y_eta)
    pruned = pearson_df.loc[pearson_df["Pearson_Selected"], "Feature"].tolist()
    x_pruned = x[pruned]

    importance_df = dual_target_importance(x_pruned, y_wrec, y_eta)
    new_top10 = importance_df["Feature"].head(TOP_N_FEATURES).tolist()
    old_top10 = load_old_top10()
    comparison_df = compare_top10(new_top10, old_top10)

    save_outputs(df, features, pearson_df, importance_df, comparison_df)

    print("Dual-target Wrec+eta feature selection completed.")
    print(f"Output directory: {OUTPUT_DIR}")
    print("New top10:")
    for i, feature in enumerate(new_top10, start=1):
        print(f"{i:02d}. {feature}")
    if old_top10:
        overlap = sum(feature in old_top10 for feature in new_top10)
        print(f"Overlap with old Wrec top10: {overlap}/{TOP_N_FEATURES}")


if __name__ == "__main__":
    main()
