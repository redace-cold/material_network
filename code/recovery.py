"""Validate complete search histories and plot exact-target recovery."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

STRATEGIES = ["ML (EHVI)", "Coarse-to-fine grid", "Random"]
LABELS = {"ML (EHVI)": "GP–EHVI", "Coarse-to-fine grid": "Grid scan", "Random": "Random"}


def parse_sequence(value):
    return np.array([float(v) for v in str(value).split(";")], dtype=float)


def cohort_median(values):
    values = np.sort(pd.Series(values, dtype=float).fillna(np.inf).to_numpy())
    if len(values) == 0 or not np.isfinite(values[len(values) // 2]):
        return np.nan
    return float(np.median(values))


def validate_runs(runs, pool_x, n_init=3, budget=30):
    required = {"Repeat", "Strategy", "Start_x", "Sequence_x", "N_to_exact_x016"}
    if not required.issubset(runs.columns):
        raise ValueError(f"Missing columns: {required - set(runs.columns)}")
    if set(runs.Strategy) != set(STRATEGIES):
        raise ValueError("Unexpected strategy set")
    if runs.duplicated(["Repeat", "Strategy"]).any():
        raise ValueError("Duplicate run/strategy")
    available = {round(float(v), 6) for v in pool_x}
    for repeat, group in runs.groupby("Repeat"):
        if set(group.Strategy) != set(STRATEGIES):
            raise ValueError(f"Unmatched strategies in repeat {repeat}")
        starts = [parse_sequence(v) for v in group.Start_x]
        if any(len(v) != n_init or not np.array_equal(v, starts[0]) for v in starts):
            raise ValueError(f"Initial sets are not matched in repeat {repeat}")
        for row in group.itertuples():
            seq = parse_sequence(row.Sequence_x)
            if len(seq) != budget or len(set(seq)) != budget:
                raise ValueError("Query sequence has wrong length or repeated compositions")
            if not np.array_equal(seq[:n_init], starts[0]):
                raise ValueError("Initial compositions missing from query count")
            if not {round(float(v), 6) for v in seq}.issubset(available):
                raise ValueError("A query lies outside the measured pool")
            hits = np.flatnonzero(np.isclose(seq, 0.16, atol=1e-10, rtol=0))
            expected = float(hits[0] + 1) if len(hits) else np.nan
            actual = row.N_to_exact_x016
            if not ((np.isnan(expected) and pd.isna(actual)) or expected == actual):
                raise ValueError("Stored hit time differs from query sequence")
    return int(runs.Repeat.nunique())


def tabulate(runs, n_init=3, budget=30):
    curve, summary = [], []
    for strategy in STRATEGIES:
        values = runs.loc[runs.Strategy == strategy, "N_to_exact_x016"].astype(float)
        for n in range(n_init, budget + 1):
            curve.append({"Strategy": LABELS[strategy], "Queries": n,
                          "Recovery_percent": float(100 * (values <= n).mean())})
        median = cohort_median(values)
        summary.append({"Strategy": LABELS[strategy], "Runs": len(values),
                        "Found_within_budget_percent": float(100 * values.notna().mean()),
                        "Median_queries_all_runs": median,
                        "Median_status": "reached" if np.isfinite(median) else f"not reached within {budget} queries",
                        "Median_queries_among_found": float(values.median()) if values.notna().any() else np.nan})
    return pd.DataFrame(curve), pd.DataFrame(summary)


def export(runs_file, output_dir, pool_x, provenance, n_init=3, budget=30):
    import matplotlib.pyplot as plt
    runs = pd.read_csv(runs_file)
    count = validate_runs(runs, pool_x, n_init, budget)
    curve, summary = tabulate(runs, n_init, budget)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    curve.to_csv(output_dir / "figure1f_recovery_values.csv", index=False)
    summary.to_csv(output_dir / "recovery_summary.csv", index=False)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "font.weight": "normal",
                         "axes.labelweight": "normal", "axes.titleweight": "normal", "axes.linewidth": 1.0}):
        fig, ax = plt.subplots(figsize=(4.7, 3.8))
        for strategy, color, style in [("GP–EHVI", "#0072B2", "-"), ("Grid scan", "#D55E00", "--"), ("Random", "#777777", ":")]:
            sub = curve[curve.Strategy == strategy]
            ax.plot(sub.Queries, sub.Recovery_percent, color=color, linestyle=style,
                    linewidth=1.8, marker="o", markersize=3, markerfacecolor="white", label=strategy)
        ax.set(xlabel="Queried compositions", ylabel="Recovery of x = 0.16 (%)", xlim=(n_init, budget), ylim=(0, 105))
        ax.legend(frameon=False, loc="lower right")
        ax.tick_params(direction="out")
        fig.tight_layout()
        for ext in ["png", "pdf", "svg"]:
            fig.savefig(output_dir / f"figure1f_recovery.{ext}", dpi=300)
        plt.close(fig)
    (output_dir / "provenance.json").write_text(json.dumps({"mode": provenance, "source": str(runs_file),
        "runs_per_strategy": count, "budget_including_initial": budget, "initial_compositions": n_init,
        "curve": "Empirical cumulative frequencies at integer queries; no smoothing or fitted curve."}, indent=2), encoding="utf-8")
    return summary
