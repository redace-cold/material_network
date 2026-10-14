# -*- coding: utf-8 -*-
"""Perovskite descriptors from A/B-site occupancy.

Reproduces the formulas in the '计算说明' sheet of ML-4-数据收集.xlsx so that
descriptors for *new* (unmeasured) compositions are computed exactly the same
way as for the measured rows. `validate()` checks this against the sheet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

R_GAS = 8.314463
R_O = 1.40
CHI_O = 3.44

ION_COLS = {
    "valence": "名义价态",
    "mass": "wa/wb 相对原子质量",
    "radius": "raa/rba Shannon离子半径 (Å)",
    "polar": "dpa/dpb 极化率 (bohr^3)",
    "chi": "snap/enbp Pauling电负性",
    "ie1": "feia/feib 第一电离能 (eV)",
}

# Descriptor column names, identical to the 'ML描述符' sheet.
DESCRIPTOR_COLS = [
    "A位平均Shannon半径(Å)", "B位平均Shannon半径(Å)", "A/B平均半径比", "A-B平均半径差(Å)",
    "容忍因子 t", "八面体因子 μ", "A位半径方差(Å^2)", "B位半径方差(Å^2)",
    "A位半径失配 δr", "B位半径失配 δr",
    "A位构型熵 ΔS_A (J/mol·K)", "B位构型熵 ΔS_B (J/mol·K)", "总构型熵 ΔS_total (J/mol·K)",
    "A位平均Pauling电负性", "B位平均Pauling电负性", "Δχ(A-O)", "Δχ(B-O)", "Δχ(A-B)",
    "A位平均名义价态", "B位平均名义价态", "A位价态失配 σV_A", "B位价态失配 σV_B", "总价态失配 σV_total",
    "A位平均相对原子质量", "B位平均相对原子质量", "A位平均极化率(bohr^3)", "B位平均极化率(bohr^3)",
    "A位平均第一电离能(eV)", "B位平均第一电离能(eV)",
]


def load_ion_table(xlsx_path) -> pd.DataFrame:
    ion = pd.read_excel(xlsx_path, sheet_name="离子特征参数")
    ion = ion.dropna(subset=["元素", "位点"])
    out = ion[["元素", "位点"]].copy()
    for key, col in ION_COLS.items():
        out[key] = pd.to_numeric(ion[col], errors="coerce")
    return out.set_index(["位点", "元素"])


def _site_stats(frac: dict[str, float], ion: pd.DataFrame, site: str) -> dict:
    total = sum(frac.values())
    x = {el: v / total for el, v in frac.items() if v > 0}
    p = ion.loc[site].loc[list(x)]
    w = np.array(list(x.values()))
    r = p["radius"].to_numpy()
    v = p["valence"].to_numpy()
    r_bar = float(w @ r)
    v_bar = float(w @ v)
    return {
        "r": r_bar,
        "rvar": float(w @ (r - r_bar) ** 2),
        "drm": float(np.sqrt(w @ (1 - r / r_bar) ** 2)),
        "S": float(-R_GAS * (w @ np.log(w))),
        "chi": float(w @ p["chi"].to_numpy()),
        "v": v_bar,
        "vmis": float(np.sqrt(w @ (v - v_bar) ** 2)),
        "mass": float(w @ p["mass"].to_numpy()),
        "polar": float(w @ p["polar"].to_numpy()),
        "ie1": float(w @ p["ie1"].to_numpy()),
        "x": x,
        "raw_total": total,
    }


def descriptors(a_frac: dict[str, float], b_frac: dict[str, float],
                ion: pd.DataFrame) -> dict:
    """All descriptors plus normalized site fractions for one composition."""
    A = _site_stats(a_frac, ion, "A")
    B = _site_stats(b_frac, ion, "B")
    d = {
        "A位平均Shannon半径(Å)": A["r"],
        "B位平均Shannon半径(Å)": B["r"],
        "A/B平均半径比": A["r"] / B["r"],
        "A-B平均半径差(Å)": A["r"] - B["r"],
        "容忍因子 t": (A["r"] + R_O) / (np.sqrt(2) * (B["r"] + R_O)),
        "八面体因子 μ": B["r"] / R_O,
        "A位半径方差(Å^2)": A["rvar"],
        "B位半径方差(Å^2)": B["rvar"],
        "A位半径失配 δr": A["drm"],
        "B位半径失配 δr": B["drm"],
        "A位构型熵 ΔS_A (J/mol·K)": A["S"],
        "B位构型熵 ΔS_B (J/mol·K)": B["S"],
        "总构型熵 ΔS_total (J/mol·K)": A["S"] + B["S"],
        "A位平均Pauling电负性": A["chi"],
        "B位平均Pauling电负性": B["chi"],
        "Δχ(A-O)": abs(CHI_O - A["chi"]),
        "Δχ(B-O)": abs(CHI_O - B["chi"]),
        "Δχ(A-B)": abs(A["chi"] - B["chi"]),
        "A位平均名义价态": A["v"],
        "B位平均名义价态": B["v"],
        "A位价态失配 σV_A": A["vmis"],
        "B位价态失配 σV_B": B["vmis"],
        "总价态失配 σV_total": None,  # filled below
        "A位平均相对原子质量": A["mass"],
        "B位平均相对原子质量": B["mass"],
        "A位平均极化率(bohr^3)": A["polar"],
        "B位平均极化率(bohr^3)": B["polar"],
        "A位平均第一电离能(eV)": A["ie1"],
        "B位平均第一电离能(eV)": B["ie1"],
    }
    d["总价态失配 σV_total"] = _total_vmis(A, B)
    for el, v in A["x"].items():
        d[f"A位分数_{el}"] = v
    for el, v in B["x"].items():
        d[f"B位分数_{el}"] = v
    return d


def _total_vmis(A: dict, B: dict) -> float:
    # The sheet does not spell out how the A and B mismatches combine; the
    # quadratic mean reproduces every row (see validate()).
    return float(np.sqrt((A["vmis"] ** 2 + B["vmis"] ** 2) / 2))


def bnt_knn_sbct_sites(x: float, y: float = 0.3) -> tuple[dict, dict]:
    """(1-x)[(1-y)Bi0.5Na0.5TiO3 - y K0.5Na0.5NbO3] - x Sr0.7Bi0.2Ca0.1TiO3.

    y = 0.3 is the measured series in the paper.
    """
    a = {
        "Bi": 0.5 * (1 - y) * (1 - x) + 0.2 * x,
        "Na": 0.5 * (1 - x),
        "K": 0.5 * y * (1 - x),
        "Sr": 0.7 * x,
        "Ca": 0.1 * x,
    }
    b = {
        "Ti": (1 - y) * (1 - x) + x,
        "Nb": y * (1 - x),
    }
    return a, b


def validate(xlsx_path) -> pd.DataFrame:
    """Recompute descriptors for every sheet row from its site fractions and
    report the largest relative deviation per descriptor."""
    ion = load_ion_table(xlsx_path)
    df = pd.read_excel(xlsx_path, sheet_name="ML描述符")
    a_cols = [c for c in df.columns if c.startswith("A位分数_")]
    b_cols = [c for c in df.columns if c.startswith("B位分数_")]
    occupancy = df[a_cols + b_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(occupancy).all():
        raise ValueError("Non-finite site occupancy in descriptor input.")
    rows = []
    for _, r in df.iterrows():
        a = {c.split("_", 1)[1]: float(r[c]) for c in a_cols if pd.to_numeric(r[c], errors="coerce") > 0}
        b = {c.split("_", 1)[1]: float(r[c]) for c in b_cols if pd.to_numeric(r[c], errors="coerce") > 0}
        rows.append(descriptors(a, b, ion))
    calc = pd.DataFrame(rows)
    report = []
    for c in DESCRIPTOR_COLS:
        ref = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        got = calc[c].to_numpy(float)
        if not np.isfinite(ref).all() or not np.isfinite(got).all():
            raise ValueError(f"Non-finite descriptor values: {c}")
        rel = np.abs(got - ref) / np.maximum(np.abs(ref), 1e-9)
        report.append({"descriptor": c, "max_rel_dev": np.max(rel),
                       "rows_over_1e-3": int(np.sum(rel > 1e-3))})
    return pd.DataFrame(report)
