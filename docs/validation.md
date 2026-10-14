# Validation performed on 2026-10-14

The package was executed locally on Windows with Python 3.12.14 and freshly installed pinned scientific dependencies.

| Check | Result | Elapsed time on this machine |
| --- | --- | ---: |
| inputs | Passed | 14.2 s |
| tests | Passed | 3.9 s |
| smoke | Passed | 5.3 s |
| figures | Passed | 3.7 s |
| seed_probe | Passed | 7.3 s |
| full_retrospective | Passed | 163.9 s |
| full_descriptors | Passed | 120.0 s |

The four unit tests cover archived recovery rates and censored medians, detection of mismatched initial sets, invariance of the next GP acquisition to changes in all unqueried outcome values, and rejection of a nonnumeric descriptor input.

The full retrospective recomputation used 100 repeats for each of three strategies, three initial compositions, a 30-query budget, and 1,000 acquisition samples. All **300 complete query sequences and both recorded hit-time fields matched the archived results exactly**.

The complete descriptor runs used 10 folds and five repeats in both scopes. Their selected features match the archive, but some model scores differ from the older saved CSVs. The maximum absolute difference in mean test normalized MAE is 0.000284666398. The archived and fresh scores are preserved separately, and all per-model differences appear in `comparison.json`.

To investigate, the unmodified original descriptor source was also executed in the same local environment, overriding only host-specific paths and disabling optional post-analysis controls. Its numeric model-comparison outputs match the packaged source with rtol=1e-9 and atol=1e-12 in both scopes. This shows that the packaging edits did not introduce the observed score differences. The precise cause of the difference from the older archived output is not established by this check; cross-platform/library numerical behavior is a possible contributor, not a confirmed explanation. Do not claim byte-exact reproduction of the archived descriptor-model results.

The workbook checksum remained unchanged, and its Git blob hash also equals the workbook already present in both historical repository workflows. Fourteen core functions/classes used for modeling, acquisition, grid order, random selection, and CV have identical abstract syntax trees to the source snapshot. Only packaging, checks, plot labels, output organization, and reporting were intentionally changed as listed in the changelog.

Input checks confirmed 148 raw rows, 147 eligible modeling rows, and 101 target-system rows. The largest relative deviation in descriptor recomputation was 4.5925613927305025e-08, below the 1e-6 threshold. The workbook records Wrec=6.87 and eta=86.0 at x=0.16; these are the maxima of both target columns in this pool.

Numerical evidence and the fresh recovery plot are saved under `verification/2026-10-14/`. Archived input/reference CSVs remain unchanged under `data/reference/`. The figure was visually checked for legible labels, legend, and complete plot borders. Scientific values, rather than pixel identity, are the criterion for descriptor-plot reproduction; font availability can change rendering across Windows and Linux.

## Limits of the verification

The two complete scientific workflows were invoked separately through their supported CLI modes; the combined `paper` entrypoint was inspected but not redundantly rerun. The optional control comparisons and untested exploratory recommendations were not executed in this packaging check. **Docker and Code Ocean execution have not been tested.** Their environment recipe and instructions are provided for a subsequent platform Reproducible Run; no capsule, public release, or DOI was created by these tests.

Matching prior results verifies the implementation and packaging. It does not resolve the scientific and manuscript-data issues documented in [manuscript_alignment.md](manuscript_alignment.md).
