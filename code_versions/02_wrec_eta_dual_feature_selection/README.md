# Updated Version: Wrec+eta Dual-Target Feature Selection

This folder contains the updated machine-learning workflow.

## Main Difference

In this version, the top-10 descriptor selection combines `Wrec` and eta. The
default feature-selection weighting is:

```text
0.70 * Wrec + 0.30 * eta
```

This design keeps `Wrec` as the primary target while allowing eta to act as an
auxiliary target during feature selection. The workflow then reuses the
selected descriptors for the `all_data` and `main_system` analyses.

## Workflow

```text
all_data
  -> dual-target Pearson redundancy pruning
  -> dual-target aggregated tree-model feature importance
  -> top-10 descriptors

main_system
  -> reuse all_data top-10 descriptors
  -> reselect the best model for Wrec and eta separately
  -> predict and rank candidate compositions
```

## Scripts

```text
ml_v2_wrec_eta.py
wrec_eta_dual_feature_selection.py
```

`ml_v2_wrec_eta.py` is the complete pipeline. It produces the `all_data` and
`main_system` outputs.

`wrec_eta_dual_feature_selection.py` is a focused diagnostic script. It only
performs Wrec+eta top-10 descriptor selection and compares the result with the
older Wrec-only top-10 descriptor set.

## Method Boundary

This version should be described as a Wrec-primary, eta-assisted
feature-selection and screening workflow. It identifies promising candidates
within the collected data range; it is not a global composition optimization
model.
