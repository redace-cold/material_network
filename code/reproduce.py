"""Portable entrypoints for manuscript analyses and explicitly optional exploration."""
from pathlib import Path
import argparse, hashlib, importlib.metadata, json, os, sys, time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "openpyxl"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_manifest():
    manifest = json.loads((ROOT / "data/MANIFEST.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        if sha(path) != entry["sha256"]:
            raise ValueError(f"Input/reference checksum mismatch: {entry['path']}")


def validate_inputs(out):
    import numpy as np
    import pandas as pd
    import descriptors as D
    import ml_v2 as V
    import ml_v3 as M
    raw = pd.read_excel(M.DATA_XLSX, sheet_name="ML描述符")
    valid = M.load_data()
    v2_valid = V.prepare_scope_work(raw)
    pool = valid.loc[valid.is_main].sort_values(M.COL_X)
    if (len(raw), len(valid), len(v2_valid), len(pool)) != (148, 147, 147, 101):
        raise ValueError("Unexpected row counts; review and version the input dataset explicitly")
    if not np.allclose(pool[M.COL_X], np.arange(0.05, 0.25001, 0.002)):
        raise ValueError("Target compositions do not match the 101-point pool")
    descriptor_check = D.validate(M.DATA_XLSX)
    if not np.isfinite(descriptor_check.max_rel_dev).all() or descriptor_check.max_rel_dev.max() > 1e-6:
        raise ValueError("Descriptor recomputation mismatch")
    out.mkdir(parents=True, exist_ok=True)
    descriptor_check.to_csv(out / "descriptor_check.csv", index=False)
    optimum = pool.loc[np.isclose(pool[M.COL_X], 0.16)].iloc[0]
    report = {"raw_rows": len(raw), "eligible_rows": len(valid), "target_rows": len(pool),
              "target_rows_overlap_descriptor_dataset": True, "target_x": 0.16,
              "target_Wrec": float(optimum[M.COL_WREC]), "target_eta": float(optimum[M.COL_ETA]),
              "target_is_max_Wrec": bool(optimum[M.COL_WREC] == pool[M.COL_WREC].max()),
              "target_is_max_eta": bool(optimum[M.COL_ETA] == pool[M.COL_ETA].max()),
              "descriptor_max_relative_error": float(descriptor_check.max_rel_dev.max())}
    (out / "input_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return pool


def retrospective(out, repeats=100, budget=30, mc=1000):
    import retrospective_bo as R
    if not (repeats >= 1 and 3 <= budget <= 101 and mc >= 1):
        raise ValueError("Invalid retrospective parameters")
    R.OUT = out / "B_retrospective_101pool"
    R.N_REPEATS, R.BUDGET, R.MC_SAMPLES = repeats, budget, mc
    R.main()
    import recovery
    return recovery.export(R.OUT / "runs.csv", R.OUT, R.load_pool()[R.M.COL_X],
                           "fresh recomputation", budget=budget)


def smoke_descriptors(out):
    import numpy as np
    import pandas as pd
    import ml_v2 as V
    import ml_v3 as M
    frame = M.load_data().loc[lambda d: d.is_main].copy()
    columns = pd.read_csv(ROOT / "data/reference/descriptors/main_system/feature_columns_top10.csv")
    # Use the exported physical names; column naming is checked explicitly.
    name_col = next(c for c in columns if columns[c].isin(frame.columns).all())
    names = columns[name_col].tolist()
    if len(names) != 7:
        raise ValueError("Expected seven target-system predictors")
    previous = (V.N_SPLITS, V.N_REPEATS)
    V.N_SPLITS, V.N_REPEATS = 2, 1
    try:
        metrics, predictions = V.repeated_cv(V.make_model("GradientBoosting"), frame[names], frame[M.COL_WREC].to_numpy(float))
        if not np.isfinite(predictions).all():raise ValueError("Non-finite smoke predictions")
        dest=out / "SMOKE_descriptor_model";dest.mkdir(exist_ok=True)
        metrics.to_csv(dest / "two_fold_smoke_metrics.csv", index=False)
    finally:
        V.N_SPLITS, V.N_REPEATS = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["paper", "descriptors", "retrospective", "figures", "check", "smoke", "exploratory"])
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--input", type=Path, default=ROOT / "data/ML-4-数据收集.xlsx")
    parser.add_argument("--repeats", type=int, default=100, help="Retrospective mode only; manuscript setting is 100")
    parser.add_argument("--budget", type=int, default=30, help="Retrospective mode only; includes initial three")
    parser.add_argument("--mc-samples", type=int, default=1000)
    parser.add_argument("--controls", action="store_true", help="Include optional v2 control comparisons")
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    os.environ["MATERIAL_INPUT"] = str(args.input.resolve())
    os.environ["MATERIAL_RESULTS"] = str(out)
    os.environ["MPLCONFIGDIR"] = str(out / ".matplotlib")
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ.setdefault(name, "1")
    check_manifest()
    record = {"mode": args.mode, "python": sys.version, "dependencies": {n: importlib.metadata.version(n) for n in PACKAGES},
              "input_sha256": sha(args.input), "settings": vars(args).copy(), "status": "running"}
    record["settings"]={k:str(v) if isinstance(v,Path) else v for k,v in record["settings"].items()}
    start=time.time()
    try:
        pool=validate_inputs(out)
        if args.mode in ["paper", "descriptors"]:
            import ml_v2
            ml_v2.main(run_controls=args.controls)
        if args.mode == "paper":
            retrospective(out, 100, 30, 1000)
        elif args.mode == "retrospective":
            retrospective(out,args.repeats,args.budget,args.mc_samples)
        elif args.mode == "figures":
            import recovery
            recovery.export(ROOT / "data/reference/retrospective/runs.csv",out / "ARCHIVED_figure1f",pool["X"],
                            "replot from archived 100-run histories; models NOT rerun")
        elif args.mode == "smoke":
            smoke_descriptors(out)
            retrospective(out,2,5,64)
            (out / "SMOKE_ONLY.txt").write_text("Environment/entrypoint check only: 2 CV folds, 2 retrospective runs, budget 5, MC 64. Not manuscript results.\n",encoding="utf-8")
        elif args.mode == "exploratory":
            import ml_v3
            ml_v3.main()
            (ml_v3.OUT_DIR / "UNVALIDATED.txt").write_text("Off-line y proposals are assumption-based and have not been experimentally tested.\n",encoding="utf-8")
        check_manifest()
        record["status"]="succeeded"
    except Exception as exc:
        record["status"]="failed";record["error"]=repr(exc);raise
    finally:
        record["elapsed_seconds"]=time.time()-start
        (out / "run_manifest.json").write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Completed {args.mode}; results: {out}",flush=True)


if __name__ == "__main__":
    main()
