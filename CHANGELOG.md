# Update prepared 2026-10-14

- At the author's request, removed the superseded `01_wrec_only_workflow/`, `02_wrec_eta_dual_workflow/`, historical illustration assets, and old overview from the current branch. Only the current package remains; prior commits preserve the old versions.

- Imported the author's current descriptor, composition-GP, descriptor-calculation, and retrospective-search source files, with source hashes retained in `docs/source_provenance.json`.
- Replaced machine-specific data/output paths with package-relative paths and optional environment overrides.
- Added one command for full paper analyses, separate commands for validation/replotting, and explicit opt-in for untested off-line proposals.
- Added pinned dependency versions, a Docker environment recipe, a Code Ocean run entrypoint, input hashes, and execution manifests.
- Corrected the label of Extra Trees, previously displayed as XGB/XGBoost, without changing its estimator.
- Distinguished the all-run median from statistics conditional on successful runs. Random's all-run median is not reached within 30 queries.
- Added finite-value checks for descriptor validation; invalid or missing numeric cells now fail explicitly.
- Added query-history validation, leakage regression tests, and empirical recovery plots with exported numerical values.
- Corrected optional control-table feature counts to reflect the features actually used.
- Documented dataset overlap, non-nested feature selection, figure-number mapping, and unresolved manuscript/data discrepancies.

The workbook measurements, original default random seeds, GP acquisition rule, grid order, random-selection rule, and 100-run/30-query protocol were preserved. No new material experiments were performed. GitHub publication and Code Ocean publication require separate verified actions; preparing this package does not perform either.
