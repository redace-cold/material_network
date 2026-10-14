# Workflow and statistical interpretation

## Descriptor regression

`ml_v2.py` reads the workbook's `ML描述符` sheet, excludes rows with missing regression targets and `eta < 50%`, and processes two scopes. The combined scope contains 147 rows; the main-system scope contains 101 of those rows.

The combined scope starts with 54 nonconstant predictors, excludes the cross-system composition variable `X`, removes redundant predictors at absolute Pearson correlation 0.85, and uses weighted target correlations (0.70 for Wrec, 0.30 for eta) to retain up to 20 predictors. Five tree models provide importance rankings for the final ten. The main-system calculation reuses these selected predictors and drops three occupancy fractions, leaving seven. Physical feature names in exported CSVs are authoritative; local descriptor codes are not a universal numbering scheme.

Regression algorithms are random forest, decision tree, gradient boosting, extra trees, AdaBoost, SVR, and ridge regression. RepeatedKFold uses 10 folds, five repeats, and seed 42. Imputation is fitted on each training fold; SVR and ridge also standardize predictors, and the target is standardized within each fitted model. Normalized MAE is MAE divided by the target's full-scope range. Error bars describe the sample standard deviation across 50 fold scores (`ddof=1`), not a confidence interval.

Feature selection uses the full combined dataset before this cross-validation. Consequently, the reported CV error is conditional on those selected features and may be optimistic. The 101-row scope overlaps the 147-row scope; neither this subset comparison nor the separate search analysis establishes independent external validation of the descriptor model. Model selection outcomes must be reported by scope and target; the archived code does not support saying GB is best for both targets in every scope.

## Retrospective search

`retrospective_bo.py` uses only the 101 measured target-system compositions. For repeat r, the initial three distinct candidates are sampled with seed `1000+r`; each strategy's subsequent random generator starts at `2000+r`. The total budget of 30 includes the initial three. All three strategies share the same initial compositions in a repeat, and sample without replacement.

The GP strategy fits three one-dimensional Gaussian processes along x to breakdown field Eb, q = Wrec/Eb, and eta. It forms Wrec predictive moments from the product Eb × q and uses a Gaussian approximation for acquisition. Kernel hyperparameters are refitted using revealed data, with one optimizer restart and optimizer seed 42. Latent-function uncertainty excludes the observation-noise term. Expected hypervolume improvement is estimated from 1,000 Monte Carlo draws over Wrec and eta; the reference point and scaling use revealed observations only. No measured outcome from an unqueried composition enters fitting or acquisition.

The grid baseline follows a **fixed** order: scan x from 0.05 to 0.25 at steps 0.04, 0.02, 0.01, 0.006, and 0.002, map positions to the nearest pool composition, remove duplicates, and skip already queried candidates. It does not refine the grid in response to measured performance. Its rise near queries 18–20 occurs because x=0.16 appears late in this predefined order; matched random starts cause small shifts in when it is encountered. This is a property of the specified baseline and target, not a universal limit of grid search.

Random selection samples uniformly from the remaining candidates. With a budget of 30 out of 101 distinct candidates, its theoretical probability of including any one fixed target is 30/101, approximately 29.7%. The archived result of 36/100 is a finite-sample realization, not a theoretical guarantee or evidence that random search cannot eventually recover the target.

The primary metric is the fraction of all runs that have queried x=0.16 by each query count. x=0.16 is designated retrospectively from the complete measured pool; it is not supplied as a selection objective to GP–EHVI. The recorded result supports more efficient recovery of this target under this protocol. It does not prove universal superiority, causation in the original experiments, transfer to another material family, or discovery of a previously unmeasured composition.

An additional source-script threshold, Wrec >= 6.40 J/cm³, describes a peak region derived from the full-pool trend. It is a post hoc secondary analysis and is not the metric plotted in main-text Fig. 1f.

Failed runs are right-censored at the query budget. `Median_queries_all_runs` is reported only if the usual empirical median can be determined within that budget; otherwise its status is `not reached`. Statistics conditional on successful runs are separately labeled `among_found`. Archived summaries are preserved unchanged for provenance and can contain the old ambiguous label `median_N`; use the corrected generated summaries when reporting results.

## Plotting

`recovery.py` validates complete sequences, matched initial sets, candidate membership, and recorded hit times before plotting. The plot connects empirical frequencies at integer query counts. No curve fitting or smoothing changes the empirical rates. It exports the plotted numbers and PNG/PDF/SVG versions.

The `figures` mode reads archived search histories and explicitly records that no models were rerun. The `retrospective` and `paper` modes instead generate fresh search histories and plot those new histories. A difference between the two is evidence to investigate, not a reason to replace the fresh output with archived values.
