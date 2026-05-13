# 01_wrec_only_workflow

This folder contains the complete previous workflow and its corresponding
outputs.

## Method

This version uses `Wrec` as the main target for feature selection:

```text
all_data descriptors
-> Pearson redundancy pruning ranked by Wrec correlation
-> tree-model feature importance ranked by Wrec
-> top-10 descriptors
-> model comparison and prediction for Wrec and eta
```

In this version, eta is used in the later model evaluation, Pareto screening
and combined-score ranking, but eta is not used to determine the top-10
descriptor set.

## Contents

```text
ML-4-数据收集.xlsx
ml_change_wrec_only.py
output/
  all_data/
  main_system/
  control_experiments/
```

## Interpretation

Use this folder to reproduce the earlier Wrec-centered analysis and to compare
it against the updated Wrec+eta feature-selection workflow.

This version should be described as a Wrec-centered descriptor-screening
baseline, not as a dual-target feature-selection model.
