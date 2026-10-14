# BNT–KNN–xSBCT: reproducible machine-learning analyses

Python code and the input workbook used for descriptor regression and a retrospective sequential-search benchmark. The package preserves the source measurements, search policies, random seeds, and default search budget from the author's archived analysis. Packaging changes and corrected statistical labels are recorded in [CHANGELOG.md](CHANGELOG.md).

The repository contains the current reproducibility package. Superseded workflow folders and their outputs have been removed from the current branch; previous commits retain their history. The input workbook and reference CSVs required by the current workflow are in `data/`.

## Run

Use Python 3.12 and install the pinned dependencies in a virtual environment:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r environment/requirements.txt
python code/reproduce.py check
python -m unittest discover -s tests -v
python code/reproduce.py smoke
python code/reproduce.py paper
```

The final command recomputes both descriptor-model analyses and all 100 paired retrospective runs. Outputs go to `results/`; inputs are never overwritten. On Linux, `bash code/run` performs the same full run. No account, API key, remote model, or internet connection is needed at runtime after installing dependencies.

`environment/requirements.txt` pins the six scientific dependencies to the versions used by the source analysis. `environment/requirements.lock.txt` additionally pins the transitive packages installed during local verification; install that file to reproduce the complete tested dependency set. The Docker recipe uses this full lock file.

| Command | Purpose |
| --- | --- |
| `check` | Check input hashes, row counts, target grid, and recomputed descriptors. |
| `smoke` | Quick execution check: two CV folds, two search runs, five queries, 64 acquisition samples. These are **not** manuscript results. |
| `paper` | Full descriptor analysis and the 100-run, 30-query benchmark. |
| `descriptors` | Descriptor analysis only; add `--controls` for the original optional control comparisons. |
| `retrospective` | Recompute the benchmark; defaults: 100 runs, 30 queries, 1,000 acquisition samples. |
| `figures` | Replot the archived 100-run histories; **does not refit any models**. |
| `exploratory` | Optional off-line composition proposals under an assumed extension away from the measured line; **not experimentally validated**. |

Pass `--output PATH` to use a separate output directory. For a shorter algorithm check, use `retrospective --repeats 2`; retain the default budget and Monte Carlo count when comparing with the corresponding archived runs. Non-default settings do not reproduce the manuscript's 100-run statistic.

## Analyses and evidence

1. **Descriptor regression (`A_descriptor_models_v2`)**: two targets, seven candidate algorithms, and repeated 10-fold cross-validation with five repeats. Both the combined dataset and the target-system subset are exported. The subset is part of the combined dataset; it is not an external validation set. Feature selection is performed before cross-validation, so these scores are not nested-CV estimates.
2. **Retrospective sequential search (`B_retrospective_101pool`)**: simulated sequential revelation of existing measurements for 101 target-system compositions. GP–EHVI, a fixed coarse-to-fine scan, and random sampling use matched initial sets. This assesses selection efficiency on a finite measured pool; it is not evidence of prospective, model-directed experiments.
3. **Optional exploration (`C_EXPLORATORY_UNVALIDATED`)**: source code for proposing compositions outside the measured line. No new experiments were performed for these proposals. This analysis is excluded from the default paper run.

The workbook contains 148 raw rows. The existing efficiency criterion (`eta >= 50%`) leaves 147 modeling rows, including all 101 target-system rows. These 101 compositions span `x = 0.05…0.25` in increments of 0.002 at fixed parent fraction `y = 0.30`. The author confirmed that their measurements are independent experimental records and that they overlap the descriptor dataset. Preparation/test-condition equivalence and original measurement records must be documented by the authors; this package does not establish them.

`data/reference/` contains archived output CSVs for comparison, not newly generated results. File hashes are in `data/MANIFEST.json`. See [workflow and interpretation](docs/workflow.md), [figure mapping and unresolved discrepancies](docs/manuscript_alignment.md), and [Code Ocean setup](docs/code_ocean.md).

## Archived Fig. 1f values

| Strategy | Recovery after 10 queries | Recovery after 30 queries | Median queries, all 100 runs |
| --- | ---: | ---: | ---: |
| GP–EHVI | 57% | 100% | 10 |
| Grid scan | 5% | 100% | 20 |
| Random | 11% | 36% | Not reached within 30 |

All counts include the initial three compositions. GP–EHVI reaches 100% by query 17; the specified grid scan reaches 100% by query 20. The random strategy's conditional median of 15 among its 36 successful runs is **not** the median for all 100 runs. The 57% and 36% values describe different query budgets and must not be compared as an equal-budget advantage.

## Reproducibility and licensing

The package records dependency versions, input checksums, parameters, and execution status in each `run_manifest.json`. Read `docs/validation.md` for actual checks performed and their limits. A successful local run does not establish a successful Code Ocean run; execute a Reproducible Run on that platform before sharing its reviewer link.

No new license is assigned by this update. Preserve and review the existing repository's license and the authors' rights to redistribute code and input data before public release. This snapshot does not create a public release, DOI, or published capsule.
