# 02_wrec_eta_dual_workflow

This folder contains the complete updated workflow and its corresponding
outputs.

## Method

This version uses Wrec+eta combined feature selection. The default feature
selection weight is:

```text
0.70 * Wrec + 0.30 * eta
```

The workflow is:

```text
all_data descriptors
-> dual-target Pearson redundancy pruning
-> dual-target tree-model feature importance
-> top-10 descriptors
-> reuse top-10 descriptors in main_system
-> model comparison and prediction for Wrec and eta
```

Compared with the previous version, introducing eta into feature selection
changes the top-10 descriptor set. Therefore, downstream model-selection
results, prediction tables and figures are also regenerated in this folder.

## Contents

```text
ML-4-数据收集.xlsx
ml_v2_wrec_eta.py
wrec_eta_dual_feature_selection.py
output/
  all_data/
  main_system/
  control_experiments/
  wrec_eta_dual_feature_selection/
```

## Interpretation

Use this folder for the current Wrec-primary, eta-assisted dual-target
screening workflow.

This version should be described as a Wrec+eta feature-selection and screening
extension of the previous Wrec-centered workflow. It identifies promising
candidates within the collected data range, not a global optimum.
