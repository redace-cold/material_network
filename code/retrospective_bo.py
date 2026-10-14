# -*- coding: utf-8 -*-
"""Retrospective search of the measured 101-composition pool.

The archived search algorithm and default seeds/budgets are retained. Only
queried measurements enter model fits and acquisition. The exact x=0.16
recovery metric corresponds to main-text Fig. 1f. The additional 6.40 J/cm3
peak-region threshold is a post hoc full-pool descriptor, not an independent
validation target. The GP structure is shared with ml_v3, but this benchmark
has its own optimizer restarts, Monte Carlo count and reference-point margins.

Run: python code/reproduce.py retrospective
"""
from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

import ml_v3 as M

warnings.filterwarnings("ignore", category=ConvergenceWarning)

N_INIT = 3
N_REPEATS = 100
BUDGET = 30
X_BEST = 0.16
PEAK_REGION_WREC = 6.40  # smooth-trend peak (~6.49) minus about one noise-floor MAE
MC_SAMPLES = 1000
OUT = M.OUT_DIR.parent / "B_retrospective_101pool"


def load_pool() -> pd.DataFrame:
    df = M.load_data()
    pool = df[df.is_main].sort_values(M.COL_X).reset_index(drop=True)
    return pool[[M.COL_X, M.COL_EB, M.COL_WE, M.COL_WREC, M.COL_ETA]]


def fit_models(pool: pd.DataFrame, idx: list[int]) -> dict:
    x = pool.loc[idx, M.COL_X].to_numpy(float)
    return {k: M.LineGP(restarts=1).fit(x, pool.loc[idx, c].to_numpy(float))
            for k, c in M.COMPONENTS.items()}


def ehvi_next(pool: pd.DataFrame, idx: list[int], rng: np.random.Generator) -> int:
    models = fit_models(pool, idx)
    cand = [i for i in range(len(pool)) if i not in idx]
    xc = pool.loc[cand, M.COL_X].to_numpy(float)
    pred = M.predict_objectives(models, xc, np.full(len(xc), M.Y_MEASURED))
    measured = pool.loc[idx, [M.COL_WREC, M.COL_ETA]].to_numpy(float)
    # Reference uses revealed measurements only, minus fixed margins.
    ref = measured.min(axis=0) - np.array([0.5, 2.0])
    front = M.pareto_front(measured)
    span = np.maximum(front.max(axis=0) - ref, 1e-9)
    f_s = (front - ref) / span
    z = rng.standard_normal((MC_SAMPLES, 2))
    mean = pred[["Wrec_mean", "eta_mean"]].to_numpy()
    std = pred[["Wrec_std", "eta_std"]].to_numpy()
    score = np.array([M.hv_improvement(f_s, (mean[j] + z * std[j] - ref) / span).mean()
                      for j in range(len(cand))])
    return cand[int(np.argmax(score))]


def grid_order(pool: pd.DataFrame) -> list[int]:
    """Coarse-to-fine scan: step 0.04, then fill with 0.02, 0.01, 0.006, 0.002."""
    x = pool[M.COL_X].round(4).to_numpy()
    order: list[int] = []
    for step in (0.04, 0.02, 0.01, 0.006, 0.002):
        for target in np.round(np.arange(0.05, 0.2501, step), 4):
            i = int(np.argmin(np.abs(x - target)))
            if i not in order:
                order.append(i)
    return order


def run_strategy(pool: pd.DataFrame, start: list[int], strategy: str,
                 rng: np.random.Generator, grid: list[int]) -> list[int]:
    idx = list(start)
    while len(idx) < BUDGET:
        if strategy == "ML (EHVI)":
            nxt = ehvi_next(pool, idx, rng)
        elif strategy == "Random":
            nxt = int(rng.choice([i for i in range(len(pool)) if i not in idx]))
        elif strategy == "Coarse-to-fine grid":
            nxt = next(i for i in grid if i not in idx)
        else:
            raise ValueError(strategy)
        idx.append(nxt)
    return idx


def experiments_to(pool: pd.DataFrame, idx: list[int], cond) -> float:
    for n, i in enumerate(idx, start=1):
        if cond(pool.loc[i]):
            return n
    return np.nan  # not reached within budget


def _cohort_median(values, budget):
    ordered = np.sort(values.fillna(np.inf).to_numpy(float))
    n = len(ordered)
    if n == 0:
        return np.nan
    lo, hi = (n - 1) // 2, n // 2
    if not np.isfinite(ordered[hi]):
        return np.nan  # not reached/identifiable within the budget
    return float((ordered[lo] + ordered[hi]) / 2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pool = load_pool()
    best_i = int(np.argmin(np.abs(pool[M.COL_X] - X_BEST)))
    assert pool.loc[best_i, M.COL_WREC] == pool[M.COL_WREC].max()
    grid = grid_order(pool)
    hit = lambda r: np.isclose(r[M.COL_X], X_BEST)
    peak = lambda r: r[M.COL_WREC] >= PEAK_REGION_WREC

    rows, traces = [], []
    for rep in range(N_REPEATS):
        start_rng = np.random.default_rng(1000 + rep)
        start = [int(i) for i in start_rng.choice(len(pool), N_INIT, replace=False)]
        for strategy in ("ML (EHVI)", "Random", "Coarse-to-fine grid"):
            rng = np.random.default_rng(2000 + rep)
            idx = run_strategy(pool, start, strategy, rng, grid)
            best_so_far = np.maximum.accumulate(pool.loc[idx, M.COL_WREC].to_numpy())
            rows.append({"Repeat": rep, "Strategy": strategy,
                         "Start_x": ";".join(f"{pool.loc[i, M.COL_X]:.3f}" for i in start),
                         "N_to_exact_x016": experiments_to(pool, idx, hit),
                         "N_to_peak_region": experiments_to(pool, idx, peak),
                         "Sequence_x": ";".join(f"{pool.loc[i, M.COL_X]:.3f}" for i in idx)})
            traces.append(pd.DataFrame({"Repeat": rep, "Strategy": strategy,
                                        "N": np.arange(1, BUDGET + 1),
                                        "Best_Wrec_so_far": best_so_far}))
        if (rep + 1) % 10 == 0:
            print(f"  {rep + 1}/{N_REPEATS} repeats done", flush=True)

    res = pd.DataFrame(rows)
    tr = pd.concat(traces, ignore_index=True)
    res.to_csv(OUT / "runs.csv", index=False, encoding="utf-8-sig")
    tr.to_csv(OUT / "best_so_far_traces.csv", index=False, encoding="utf-8-sig")

    def summarize(col):
        g = res.groupby("Strategy")[col]
        return pd.DataFrame({
            "found_within_budget_%": g.apply(lambda s: 100 * s.notna().mean()),
            # Failed runs remain censored above BUDGET for the cohort metric.
            "median_N_all_runs": g.apply(lambda s: _cohort_median(s, BUDGET)),
            "median_N_among_found": g.median(),
            "mean_N_among_found": g.mean(),
            "P25_N_among_found": g.quantile(0.25),
            "P75_N_among_found": g.quantile(0.75),
        })

    summ = pd.concat({"exact x=0.16": summarize("N_to_exact_x016"),
                      "peak region (Wrec>=6.40)": summarize("N_to_peak_region")}, axis=1)
    summ.to_csv(OUT / "summary.csv", encoding="utf-8-sig")

    paper_scan = [0.10, 0.12, 0.14, 0.16, 0.18]
    print(f"\nBudget {BUDGET} experiments, start = {N_INIT} random compositions, "
          f"{N_REPEATS} repeats.")
    print("Paper's scan (0.10, 0.12, ..., 0.18) reaches x = 0.16 at experiment "
          f"{paper_scan.index(0.16) + 1} of {len(paper_scan)} — but that grid was chosen "
          "knowing the series; it is shown for reference, not as a blind strategy.\n")
    print(summ.round(1).to_string())

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for strategy, color in (("ML (EHVI)", "C3"), ("Random", "C7"), ("Coarse-to-fine grid", "C0")):
        s = tr[tr.Strategy == strategy].groupby("N")["Best_Wrec_so_far"]
        med, lo, hi = s.median(), s.quantile(0.25), s.quantile(0.75)
        ax.plot(med.index, med, color=color, lw=1.8, label=strategy)
        ax.fill_between(med.index, lo, hi, color=color, alpha=0.18)
    ax.axhline(pool[M.COL_WREC].max(), color="k", ls="--", lw=1, label="x = 0.16 (6.87)")
    ax.axhline(PEAK_REGION_WREC, color="k", ls=":", lw=1, label="peak region (6.40)")
    ax.set_xlabel("Number of experiments")
    ax.set_ylabel(r"Best measured $W_{rec}$ so far (J/cm$^3$)")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "best_so_far.png", dpi=300)
    plt.close(fig)
    print(f"\nOutputs in {OUT}")


if __name__ == "__main__":
    main()
