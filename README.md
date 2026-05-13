# Material Network: BNT-KNN-xSBCT Dielectric Ceramic Screening

This repository contains descriptor-driven machine-learning workflows for
small-sample screening of dielectric energy-storage ceramic compositions.

The repository is organized around two complete workflow versions. Each version
contains its own script, dataset copy, and output results, so the figures and
tables can be compared directly without mixing old and new outputs.

> Scope note: this project is a small-sample, descriptor-driven screening tool.
> It identifies promising candidates within the collected dataset and measured
> BNT-KNN-xSBCT composition range. It is not a universal ceramic predictor or a
> global composition optimization framework.

## Repository Structure

```text
.
├── 01_wrec_only_workflow/
│   ├── ML-4-数据收集.xlsx
│   ├── ml_change_wrec_only.py
│   ├── README.md
│   └── output/
│       ├── all_data/
│       ├── main_system/
│       └── control_experiments/
├── 02_wrec_eta_dual_workflow/
│   ├── ML-4-数据收集.xlsx
│   ├── ml_v2_wrec_eta.py
│   ├── wrec_eta_dual_feature_selection.py
│   ├── README.md
│   └── output/
│       ├── all_data/
│       ├── main_system/
│       ├── control_experiments/
│       └── wrec_eta_dual_feature_selection/
├── docs/
│   └── images/
├── requirements.txt
└── README.md
```

## Version 1: Wrec-Only Feature Selection

Folder:

```text
01_wrec_only_workflow/
```

This is the previous workflow. The top-10 descriptor selection is driven mainly
by recoverable energy density, `Wrec`.

Workflow logic:

```text
all_data descriptors
-> Pearson redundancy pruning ranked by Wrec correlation
-> tree-model feature importance ranked by Wrec
-> top-10 descriptors
-> model comparison and prediction for Wrec and eta
```

In this version, eta is still used later for prediction, Pareto screening, and
combined-score candidate ranking, but eta does not participate in selecting the
top-10 descriptors.

Use this folder when reproducing the earlier Wrec-centered method or comparing
it against the updated dual-target feature-selection workflow.

## Version 2: Wrec+eta Dual-Target Feature Selection

Folder:

```text
02_wrec_eta_dual_workflow/
```

This is the updated workflow. The top-10 descriptor selection combines Wrec and
eta. The default feature-selection weighting is:

```text
0.70 * Wrec + 0.30 * eta
```

Workflow logic:

```text
all_data descriptors
-> dual-target Pearson redundancy pruning
-> dual-target tree-model feature importance
-> top-10 descriptors
-> reuse top-10 descriptors in main_system
-> model comparison and prediction for Wrec and eta
```

Because eta participates in feature selection, the selected top-10 descriptors
change relative to the Wrec-only version. Therefore, model-selection results,
prediction tables, candidate rankings, and figures are regenerated and stored
inside this version folder.

Use this folder for the current Wrec-primary, eta-assisted screening workflow.

## all_data and main_system

Both versions retain the same two-scope analysis logic.

`all_data` contains all valid descriptor rows. It is used for descriptor
screening, redundancy pruning, feature-importance ranking, model comparison,
and construction of the screening strategy.

`main_system` contains the BNT-KNN-xSBCT target composition series. It reuses
the descriptor set selected from `all_data` and then performs model comparison,
prediction, Pareto screening, and candidate ranking within the target system.

This avoids treating a one-dimensional composition series as an independent
feature-selection space, where many descriptors vary monotonically with `x`.

## How To Run

Install the required packages:

```powershell
pip install -r requirements.txt
```

Run the previous Wrec-centered workflow:

```powershell
python 01_wrec_only_workflow/ml_change_wrec_only.py
```

Run the updated Wrec+eta workflow:

```powershell
python 02_wrec_eta_dual_workflow/ml_v2_wrec_eta.py
```

Run only the dual-target feature-selection diagnostic:

```powershell
python 02_wrec_eta_dual_workflow/wrec_eta_dual_feature_selection.py
```

## Interpretation Boundary

The machine-learning results should be described as descriptor-assisted
screening within the available dataset and measured composition range. The
predicted best candidate should be written as the most promising candidate
within the collected sample range, not as a global optimum.
