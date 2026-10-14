# -*- coding: utf-8 -*-
"""ML-guided design for (1-x)[(1-y)BNT - yKNN] - xSBCT ceramics.

Steps:

  1. Load the data (46 literature rows + 101 measured BNT-KNN-xSBCT
     compositions, y = 0.30) and estimate the measurement-noise floor of the
     series from neighbouring compositions.
  2. Validate models honestly and save the comparison:
       - blocked CV inside the series: contiguous x blocks of 0.02 are held
         out together, so a test point never has a near-duplicate neighbour
         in the training set;
       - edge extrapolation: x < 0.09 and x > 0.21 held out;
       - 10-fold over literature rows (every literature row is a different
         system), for the descriptor models only.
     Candidates: descriptor models (GP / GB / RF on four perovskite
     descriptors, trained on all rows) and the composition GP used for
     design (trained on the series only).
  3. Design model. Wrec is modelled as Eb x (W/E) and eta directly, each with
     a Gaussian process along x. Off the measured line (y != 0.30) there is
     no data, so the model states its assumption explicitly: the best guess
     is the value on the measured line at the same x, and the uncertainty
     grows with |y - 0.30| on the same correlation length learned along x
     (composition-space isotropy). Descriptor models are not used for the
     y-direction because they do not transfer across systems (step 2).
  4. Recommend the next batch inside a trust region of about one correlation
     length around the measured line, by Monte-Carlo expected hypervolume
     improvement (EHVI) on (Wrec, eta). Points closer than EXCLUSION_RADIUS
     to an earlier pick are skipped so the batch spreads out.
  5. Flag measured points that sit far from the smooth trend (> 3 noise SD)
     as candidates for replicate samples.

Run: python code/reproduce.py exploratory
This optional analysis proposes untested compositions; it is not the manuscript benchmark.
"""
from __future__ import annotations

import warnings
from pathlib import Path
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import descriptors as D

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_XLSX = Path(os.environ.get("MATERIAL_INPUT", BASE_DIR.parent / "data" / "ML-4-数据收集.xlsx"))
OUT_DIR = Path(os.environ.get("MATERIAL_RESULTS", BASE_DIR.parent / "results")) / "C_EXPLORATORY_UNVALIDATED"
SEED = 42

ETA_MIN = 50.0  # rows below this efficiency are outside the application window
COL_X = "X"
COL_WREC = "Wrec(J/cm3)"
COL_ETA = "效率η（%）"
COL_EB = "Eb(kV/cm)"
COL_WE = "W/E"
TARGETS = {"Wrec": COL_WREC, "eta": COL_ETA}
TARGET_LABELS = {"Wrec": r"$W_{rec}$ (J/cm$^3$)", "eta": r"$\eta$ (%)"}
# Wrec = Eb x (W/E). Eb and W/E each vary more smoothly with composition
# than their product; modelling them separately lowered the blocked-CV error
# of Wrec in the model comparison (validation_summary.csv).
COMPONENTS = {"Eb": COL_EB, "WE": COL_WE, "eta": COL_ETA}
MAIN_MARKERS = ("0.7Bi0.5Na0.5TiO3-0.3K0.5Na0.5NbO3", "Sr0.7Bi0.2Ca0.1TiO3")

# Descriptor set for the comparison models (validation only).
FEATURES = [
    "容忍因子 t",
    "总构型熵 ΔS_total (J/mol·K)",
    "A位平均极化率(bohr^3)",
    "B位平均名义价态",
]

# Design space: x = SBCT content, y = KNN fraction in the BNT-KNN parent.
Y_MEASURED = 0.30
TRUST_RADIUS_Y = 0.04  # about one correlation length learned along x
X_GRID = np.round(np.arange(0.05, 0.2501, 0.005), 4)
Y_GRID = np.round(np.arange(Y_MEASURED - TRUST_RADIUS_Y,
                            Y_MEASURED + TRUST_RADIUS_Y + 1e-9, 0.01), 4)
BATCH_SIZE = 3
EXCLUSION_RADIUS = 0.02
N_MC = 4000
REPLICATE_SIGMA = 3.0

BLOCK_WIDTH = 0.02
EDGE_LOW, EDGE_HIGH = 0.09, 0.21


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_XLSX, sheet_name="ML描述符")
    for c in [COL_X, COL_WREC, COL_ETA, COL_EB, COL_WE, *FEATURES]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[COL_WREC, COL_ETA, COL_EB, COL_WE, *FEATURES])
    df = df[df[COL_ETA] >= ETA_MIN].reset_index(drop=True)
    formula = df["体系"].astype(str)
    main = np.ones(len(df), dtype=bool)
    for marker in MAIN_MARKERS:
        main &= formula.str.contains(marker, regex=False).to_numpy()
    df["is_main"] = main
    return df


def noise_floor(df: pd.DataFrame) -> pd.DataFrame:
    """Measurement noise of the evenly spaced series from second differences
    of neighbouring compositions (Rice estimator): a smooth trend cancels,
    composition-to-composition scatter remains."""
    m = df[df.is_main].sort_values(COL_X)
    rows = []
    for key, col in TARGETS.items():
        y = m[col].to_numpy(float)
        d2 = y[2:] - 2 * y[1:-1] + y[:-2]
        sd = float(np.sqrt(np.mean(d2 ** 2) / 6))
        rows.append({"Target": key, "Noise_SD": sd,
                     "MAE_floor": sd * np.sqrt(2 / np.pi)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def _gp(n_dims: int, restarts: int = 3) -> TransformedTargetRegressor:
    kernel = (ConstantKernel(1.0, (1e-2, 1e2))
              * Matern(length_scale=np.ones(n_dims) if n_dims > 1 else 1.0,
                       nu=2.5, length_scale_bounds=(1e-2, 1e3))
              + WhiteKernel(0.05, (1e-5, 1.0)))
    gp = GaussianProcessRegressor(kernel, n_restarts_optimizer=restarts,
                                  random_state=SEED)
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), gp),
        transformer=StandardScaler())


def _gb():
    return GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.04, max_depth=3,
        min_samples_leaf=4, subsample=0.8, random_state=SEED)


def _rf():
    return RandomForestRegressor(
        n_estimators=300, max_depth=6, min_samples_leaf=4,
        max_features="sqrt", random_state=SEED, n_jobs=1)


# name -> (factory, input columns, trained on the series only?)
MODELS = {
    "GP_composition": (lambda: _gp(1), [COL_X], True),
    "GP_descriptors": (lambda: _gp(len(FEATURES)), FEATURES, False),
    "GB_descriptors": (_gb, FEATURES, False),
    "RF_descriptors": (_rf, FEATURES, False),
}
DESIGN_MODEL = "GP_composition"


class LineGP:
    """GP along x for one property, extended to y != 0.30 by assumption.

    On the measured line this is an ordinary GP posterior. Off the line the
    mean stays at the line value and the latent variance adds
    s^2 (1 - rho(|y - 0.30|)^2), with s^2 and the Matern-5/2 correlation rho
    taken from the kernel learned along x. This is the posterior of
    f(x, y) = g(x) + h(x, y), h being an independent GP with the same
    hyper-parameters and pinned to zero on the measured line.
    """

    def __init__(self, restarts: int = 5):
        self.model = _gp(1, restarts)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LineGP":
        self.model.fit(x.reshape(-1, 1), y)
        gp = self.model.regressor_.steps[-1][1]
        x_scale = self.model.regressor_.steps[0][1].scale_[0]
        self.y_scale = self.model.transformer_.scale_[0]
        self.signal_var = gp.kernel_.k1.k1.constant_value
        self.noise_var = gp.kernel_.k2.noise_level
        self.length_scale = gp.kernel_.k1.k2.length_scale * x_scale
        return self

    def predict(self, x: np.ndarray, y_knn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pipe = self.model.regressor_
        mean_s, std_s = pipe.predict(x.reshape(-1, 1), return_std=True)
        latent_var = np.maximum(std_s ** 2 - self.noise_var, 0.0)
        r = np.abs(y_knn - Y_MEASURED) / self.length_scale
        rho = (1 + np.sqrt(5) * r + 5 * r ** 2 / 3) * np.exp(-np.sqrt(5) * r)
        latent_var = latent_var + self.signal_var * (1 - rho ** 2)
        mean = self.model.transformer_.inverse_transform(mean_s.reshape(-1, 1)).ravel()
        return mean, np.sqrt(latent_var) * self.y_scale


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _predict_all(name: str, df: pd.DataFrame, train: np.ndarray,
                 test: np.ndarray) -> dict[str, np.ndarray]:
    """Wrec two ways (direct, and as Eb x W/E) plus eta."""
    factory, cols, series_only = MODELS[name]
    if series_only:
        train = train & df.is_main.to_numpy()
    X = df[cols].to_numpy(float)

    def fp(col):
        m = factory()
        m.fit(X[train], df[col].to_numpy(float)[train])
        return m.predict(X[test])

    return {"Wrec": fp(COL_EB) * fp(COL_WE), "Wrec_direct": fp(COL_WREC), "eta": fp(COL_ETA)}


def validate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    main = df.is_main.to_numpy()
    x = df[COL_X].to_numpy(float)
    block = np.where(main, np.floor((x - 0.05) / BLOCK_WIDTH + 1e-9), -1).astype(int)
    edge = main & ((x < EDGE_LOW) | (x > EDGE_HIGH))
    lit_idx = np.where(~main)[0]
    lit_tests = []
    for chunk in np.array_split(np.random.default_rng(SEED).permutation(lit_idx), 10):
        t = np.zeros(len(df), dtype=bool)
        t[chunk] = True
        lit_tests.append(t)
    splits = {"block": [block == g for g in np.unique(block[main])],
              "edge": [edge], "lit": lit_tests}

    eval_targets = {"Wrec": COL_WREC, "Wrec_direct": COL_WREC, "eta": COL_ETA}
    summary, preds = [], []
    for name, (_, _, series_only) in MODELS.items():
        out = {kind: {k: np.full(len(df), np.nan) for k in eval_targets} for kind in splits}
        for kind, tests in splits.items():
            if kind == "lit" and series_only:
                continue  # a series-only model cannot predict other systems
            for test in tests:
                p = _predict_all(name, df, ~test, test)
                for k in eval_targets:
                    out[kind][k][test] = p[k]
        for key, col in eval_targets.items():
            y = df[col].to_numpy(float)
            pb, pe, pl = out["block"][key], out["edge"][key], out["lit"][key]
            has_lit = np.isfinite(pl[~main]).all()
            summary.append({
                "Model": name, "Target": key,
                "Blocked_MAE": mean_absolute_error(y[main], pb[main]),
                "Blocked_R2": r2_score(y[main], pb[main]),
                "Edge_MAE": mean_absolute_error(y[edge], pe[edge]),
                "Literature_10fold_MAE": mean_absolute_error(y[~main], pl[~main]) if has_lit else np.nan,
                "Literature_10fold_R2": r2_score(y[~main], pl[~main]) if has_lit else np.nan,
            })
            preds.append(pd.DataFrame({
                "Model": name, "Target": key, "is_main": main, "X": x,
                "Measured": y, "Pred_Blocked": pb, "Pred_Edge": pe,
                "Pred_Literature10fold": pl,
            }))
            s = summary[-1]
            print(f"  {name:15s} {key:11s} blocked MAE={s['Blocked_MAE']:.3f} "
                  f"R2={s['Blocked_R2']:.3f} | edge MAE={s['Edge_MAE']:.3f} | "
                  f"lit MAE={s['Literature_10fold_MAE']:.3f}", flush=True)
    return pd.DataFrame(summary), pd.concat(preds, ignore_index=True)


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------
def pareto_front(points: np.ndarray) -> np.ndarray:
    """Non-dominated rows (maximization), sorted by the first objective."""
    keep = [i for i, p in enumerate(points)
            if not np.any(np.all(points >= p, axis=1) & np.any(points > p, axis=1))]
    front = points[keep]
    return front[np.argsort(front[:, 0])]


def hv_improvement(front: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Hypervolume improvement of each point in `s` (n, 2) over a 2-D
    maximization front, both shifted so the reference point is 0.

    Improvement = box [0, s1] x [0, s2] minus the part the front already
    dominates. With the front sorted by f1 ascending (f2 descending) the
    dominated area of a staircase is sum((a_i - a_{i-1}) b_i); clipping the
    front to the box keeps that order, so this vectorizes over samples.
    """
    s = np.maximum(s, 0.0)
    a = np.maximum(np.minimum(front[None, :, 0], s[:, :1]), 0.0)
    b = np.maximum(np.minimum(front[None, :, 1], s[:, 1:]), 0.0)
    da = np.diff(np.concatenate([np.zeros((len(s), 1)), a], axis=1), axis=1)
    return np.maximum(s[:, 0] * s[:, 1] - (da * b).sum(axis=1), 0.0)


def ehvi(mean: np.ndarray, std: np.ndarray, front: np.ndarray,
         ref: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Monte-Carlo EHVI with independent objectives, scaled by the measured
    range. The same normal draws serve every candidate (common random
    numbers) so the ranking is stable."""
    span = np.maximum(front.max(axis=0) - ref, 1e-9)
    f_s = (front - ref) / span
    z = rng.standard_normal((N_MC, 2))
    return np.array([hv_improvement(f_s, (mean[i] + z * std[i] - ref) / span).mean()
                     for i in range(len(mean))])


def fit_design_models(df: pd.DataFrame) -> dict[str, LineGP]:
    m = df[df.is_main]
    models = {}
    for key, col in COMPONENTS.items():
        g = LineGP().fit(m[COL_X].to_numpy(float), m[col].to_numpy(float))
        models[key] = g
        print(f"  {key:3s}: correlation length along x = {g.length_scale:.4f}, "
              f"noise SD = {np.sqrt(g.noise_var) * g.y_scale:.4g}", flush=True)
    return models


def predict_objectives(models: dict[str, LineGP], x: np.ndarray,
                       y_knn: np.ndarray) -> pd.DataFrame:
    """Wrec = Eb x (W/E) with independent Gaussian components:
    var = mu_E^2 s_W^2 + mu_W^2 s_E^2 + s_E^2 s_W^2."""
    (mE, sE), (mW, sW), (mH, sH) = (models[k].predict(x, y_knn) for k in COMPONENTS)
    return pd.DataFrame({
        "Eb_mean": mE, "Eb_std": sE, "WE_mean": mW, "WE_std": sW,
        "Wrec_mean": mE * mW,
        "Wrec_std": np.sqrt(mE ** 2 * sW ** 2 + mW ** 2 * sE ** 2 + sE ** 2 * sW ** 2),
        "eta_mean": mH, "eta_std": sH,
    })


def recommend(df: pd.DataFrame, models: dict[str, LineGP]) -> tuple[pd.DataFrame, pd.DataFrame]:
    gx, gy = np.meshgrid(X_GRID, Y_GRID)
    grid = pd.DataFrame({"x": gx.ravel(), "y": gy.ravel()})
    grid = pd.concat([grid, predict_objectives(models, grid.x.to_numpy(), grid.y.to_numpy())],
                     axis=1)

    measured = df.loc[df.is_main, [COL_WREC, COL_ETA]].to_numpy(float)
    ref = measured.min(axis=0)
    front = pareto_front(measured)
    mean = grid[["Wrec_mean", "eta_mean"]].to_numpy()
    std = grid[["Wrec_std", "eta_std"]].to_numpy()
    grid["EHVI"] = ehvi(mean, std, front, ref, np.random.default_rng(SEED))
    grid["Delta_y"] = (grid.y - Y_MEASURED).round(4)

    # The measured line is already sampled every 0.002 in x; new information
    # lies off it.
    candidates = grid[~np.isclose(grid.y, Y_MEASURED)].sort_values("EHVI", ascending=False)
    picks: list[int] = []
    for i, r in candidates.iterrows():
        if all(np.hypot(r.x - grid.at[j, "x"], r.y - grid.at[j, "y"]) >= EXCLUSION_RADIUS
               for j in picks):
            picks.append(i)
        if len(picks) == BATCH_SIZE:
            break

    rec = grid.loc[picks].copy()
    rec.insert(0, "Rank", range(1, len(picks) + 1))
    rec["Formula"] = [
        f"{1 - r.x:.3f}[{1 - r.y:.2f}Bi0.5Na0.5TiO3-{r.y:.2f}K0.5Na0.5NbO3]-{r.x:.3f}Sr0.7Bi0.2Ca0.1TiO3"
        for r in rec.itertuples()
    ]
    return grid, rec


def replicate_check(df: pd.DataFrame, models: dict[str, LineGP],
                    noise: pd.DataFrame) -> pd.DataFrame:
    """Measured points far from the smooth trend, in units of noise SD."""
    m = df[df.is_main].sort_values(COL_X)
    x = m[COL_X].to_numpy(float)
    pred = predict_objectives(models, x, np.full(len(x), Y_MEASURED))
    rows = []
    for key, col in TARGETS.items():
        sd = noise.loc[noise.Target == key, "Noise_SD"].iloc[0]
        trend = pred[f"{key}_mean"].to_numpy()
        z = (m[col].to_numpy(float) - trend) / sd
        for xi, yi, mu, zi in zip(x, m[col], trend, z):
            if abs(zi) > REPLICATE_SIGMA:
                rows.append({"Target": key, "x": xi, "Measured": yi,
                             "Trend": mu, "Deviation_in_noise_SD": zi})
    return pd.DataFrame(rows, columns=["Target", "x", "Measured", "Trend",
                                       "Deviation_in_noise_SD"])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "axes.linewidth": 1.1,
    "mathtext.default": "regular",
    "axes.unicode_minus": False,
})


def plot_validation(preds: pd.DataFrame, summary: pd.DataFrame, noise: pd.DataFrame):
    sub = preds[(preds.Model == DESIGN_MODEL) & preds.is_main]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax, key in zip(axes, TARGETS):
        s = sub[sub.Target == key]
        sc = ax.scatter(s.Measured, s.Pred_Blocked, s=22, c=s.X, cmap="viridis",
                        edgecolor="k", linewidth=0.3)
        lo = min(s.Measured.min(), s.Pred_Blocked.min())
        hi = max(s.Measured.max(), s.Pred_Blocked.max())
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "r-", lw=1.2)
        row = summary[(summary.Model == DESIGN_MODEL) & (summary.Target == key)].iloc[0]
        floor = noise.loc[noise.Target == key, "MAE_floor"].iloc[0]
        ax.text(0.04, 0.96,
                f"Blocked CV (held-out x blocks of {BLOCK_WIDTH})\n"
                f"$R^2$ = {row.Blocked_R2:.3f},  MAE = {row.Blocked_MAE:.3f}\n"
                f"noise-floor MAE ≈ {floor:.3f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="k"))
        ax.set_xlabel(f"Measured {TARGET_LABELS[key]}")
        ax.set_ylabel(f"Predicted {TARGET_LABELS[key]}")
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(sc, ax=ax, label="x", shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "validation_parity_blocked.png", dpi=300)
    plt.close(fig)


def plot_series_fit(df: pd.DataFrame, models: dict[str, LineGP], flags: pd.DataFrame):
    xs = np.linspace(0.05, 0.25, 401)
    pred = predict_objectives(models, xs, np.full(len(xs), Y_MEASURED))
    m = df[df.is_main]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, key in zip(axes, TARGETS):
        mu, sd = pred[f"{key}_mean"], pred[f"{key}_std"]
        ax.fill_between(xs, mu - 1.96 * sd, mu + 1.96 * sd, color="C0", alpha=0.25,
                        label="95% interval of the trend")
        ax.plot(xs, mu, "C0-", lw=1.5, label="GP trend (y = 0.30)")
        ax.scatter(m[COL_X], m[TARGETS[key]], s=12, c="k", label="Measured")
        f = flags[flags.Target == key]
        if len(f):
            ax.scatter(f.x, f.Measured, s=90, facecolors="none", edgecolors="red",
                       linewidths=1.5, label=f"> {REPLICATE_SIGMA:.0f} noise SD from trend")
        ax.set_xlabel("x (SBCT content)")
        ax.set_ylabel(TARGET_LABELS[key])
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "series_fit_y030.png", dpi=300)
    plt.close(fig)


def plot_design_maps(df: pd.DataFrame, grid: pd.DataFrame, rec: pd.DataFrame):
    panels = [("Wrec_mean", r"Predicted $W_{rec}$ (J/cm$^3$)", "viridis"),
              ("eta_mean", r"Predicted $\eta$ (%)", "viridis"),
              ("Wrec_std", r"Uncertainty of $W_{rec}$ (1$\sigma$)", "magma"),
              ("EHVI", "EHVI (acquisition)", "cividis")]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6))
    xm = df.loc[df.is_main, COL_X]
    for ax, (col, label, cmap) in zip(axes.ravel(), panels):
        Z = grid.pivot(index="y", columns="x", values=col)
        im = ax.pcolormesh(Z.columns, Z.index, Z.values, cmap=cmap, shading="nearest")
        fig.colorbar(im, ax=ax, label=label)
        ax.plot(xm, np.full(len(xm), Y_MEASURED), "w.", ms=3, label="Measured (y = 0.30)")
        ax.scatter(rec.x, rec.y, marker="*", s=200, c="red", edgecolor="k", label="Recommended")
        for r in rec.itertuples():
            ax.annotate(str(r.Rank), (r.x, r.y), xytext=(5, 4), textcoords="offset points",
                        color="red", fontsize=10, fontweight="bold")
        ax.set_xlabel("x (SBCT content)")
        ax.set_ylabel("y (KNN fraction in parent)")
    axes[0, 0].legend(loc="lower left", fontsize=7, framealpha=0.85)
    fig.tight_layout()
    fig.suptitle("UNTESTED proposals: assumed isotropic extension from y = 0.30", fontsize=9)
    fig.savefig(OUT_DIR / "exploratory_design_maps_UNTESTED.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def save(df: pd.DataFrame, name: str):
    df.to_csv(OUT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/6] Checking descriptor calculator against the data sheet")
    worst = D.validate(DATA_XLSX)["max_rel_dev"].max()
    if not np.isfinite(worst) or worst > 1e-6:
        raise RuntimeError(f"Descriptor mismatch (max relative deviation {worst:.2e}).")

    print("[2/6] Loading data and estimating the noise floor")
    df = load_data()
    noise = noise_floor(df)
    save(noise, "noise_floor")
    print(f"  rows: {len(df)} (series {int(df.is_main.sum())}, literature {int((~df.is_main).sum())})")
    print(noise.round(4).to_string(index=False))

    print("[3/6] Validating models")
    summary, preds = validate(df)
    save(summary, "validation_summary")
    save(preds, "validation_predictions")
    plot_validation(preds, summary, noise)

    print("[4/6] Fitting the design model")
    models = fit_design_models(df)

    print("[5/6] Scoring the design space and checking for outliers")
    grid, rec = recommend(df, models)
    flags = replicate_check(df, models, noise)
    save(grid, "design_space_predictions")
    save(rec, "exploratory_offline_candidates_UNTESTED")
    save(flags, "replicate_candidates")

    print("[6/6] Plotting")
    plot_series_fit(df, models, flags)
    plot_design_maps(df, grid, rec)

    print("\nUNTESTED, assumption-based exploratory candidates:")
    cols = ["Rank", "x", "y", "Wrec_mean", "Wrec_std", "eta_mean", "eta_std", "EHVI"]
    print(rec[cols].round(4).to_string(index=False))
    print("\nMeasured points to replicate (far from the smooth trend):")
    print(flags.round(3).to_string(index=False) if len(flags) else "  none")
    print(f"\nOutputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
