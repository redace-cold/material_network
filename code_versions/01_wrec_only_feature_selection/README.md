# Previous Version: Wrec-Only Feature Selection

This folder contains the previous machine-learning workflow.

## Main Difference

In this version, the top-10 descriptor selection is based mainly on `Wrec`
as the primary regression target. The feature-selection path is:

```text
descriptor matrix -> Pearson redundancy pruning using Wrec -> tree-model
feature importance using Wrec -> top-10 descriptors
```

After the top-10 descriptors are selected, the workflow still trains and
evaluates models for both `Wrec` and eta. Therefore, eta is present in the
later prediction and dual-objective scoring stage, but it is not used to drive
the original top-10 feature ranking.

## When To Use

Use this script when you want to reproduce the earlier Wrec-centered analysis
or compare it with the updated Wrec+eta feature-selection workflow.

## Script

```text
ml_change_wrec_only.py
```

## Method Boundary

This version should be described as a Wrec-centered descriptor screening
workflow. It should not be described as a dual-target feature-selection model.
