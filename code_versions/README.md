# Code Versions

This directory separates the previous Wrec-only feature-selection workflow from
the updated Wrec+eta dual-target feature-selection workflow.

## 01_wrec_only_feature_selection

This folder stores the previous workflow. In this version, the top-10
descriptor selection is driven mainly by the primary recoverable energy density
target, `Wrec`. The eta model is used later for prediction and auxiliary
screening, but eta does not participate in the top-10 feature selection step.

Use this version when you need to reproduce the earlier analysis logic or
compare the effect of adding eta into feature selection.

Main script:

```text
01_wrec_only_feature_selection/ml_change_wrec_only.py
```

## 02_wrec_eta_dual_feature_selection

This folder stores the updated workflow. In this version, the top-10
descriptor selection combines the primary `Wrec` target and the auxiliary eta
target. The default feature-selection weighting is:

```text
0.70 * Wrec + 0.30 * eta
```

This keeps Wrec as the primary optimization target while allowing eta to
contribute to descriptor ranking and candidate screening.

Main scripts:

```text
02_wrec_eta_dual_feature_selection/ml_v2_wrec_eta.py
02_wrec_eta_dual_feature_selection/wrec_eta_dual_feature_selection.py
```

The second script is a focused diagnostic tool that only performs Wrec+eta
dual-target feature selection and compares the resulting top-10 descriptors
with the older Wrec-only top-10 descriptors.

## Recommended Interpretation

The two folders should be described as two method versions rather than two
unrelated projects:

- the first version establishes the Wrec-centered descriptor-driven screening
  baseline;
- the second version extends that baseline by adding eta into the feature
  selection and screening logic.

Neither version should be described as a global composition optimization
framework. Both identify promising candidates within the collected dataset and
the measured BNT-KNN-xSBCT composition range.
