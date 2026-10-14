# Figure mapping and items requiring author reconciliation

The code preserves scientific inputs and algorithms. It does not silently modify data to match a manuscript figure or claim that all current manuscript panels are reproduced exactly.

| Current manuscript content | Source output |
| --- | --- |
| Fig. 1d, descriptor importance | `A_descriptor_models_v2/*/Figure1e_*` (legacy figure numbering) |
| Fig. 1e, model comparison | `A_descriptor_models_v2/*/Figure1f_*` (legacy figure numbering) |
| Fig. 1f, retrospective recovery | `B_retrospective_101pool/figure1f_recovery.*` |
| Pearson correlation matrix, now supplementary | `A_descriptor_models_v2/*/Figure1d_*` |

The material schematics and all experimental characterization panels are outside the ML code's scope. This package generates numerical analysis panels rather than the complete assembled Figure 1 or the Origin project.

The following discrepancies were found while comparing the saved code outputs and earlier manuscript/figure versions. They are not repaired by changing raw measurements:

- **148 versus 147:** 148 workbook rows are reduced to 147 by the code's efficiency threshold. Wording should distinguish raw compiled records from rows actually fitted.
- **Best model:** saved combined-scope results favor RF for both targets; saved target-system results favor GB for Wrec and ET for eta. “GB had the smallest errors for both” requires reconciliation with the actual scope and metric.
- **Model comparison panel:** the earlier figure's ET test-Wrec entry appears to repeat its eta entry. The archived target-scope CSV gives normalized MAE 0.020529625 with SD 0.006727209 for ET test-Wrec, rather than 0.037915010 with SD 0.016966788. The current manuscript image must be checked against the exported source data.
- **Feature labels:** old figure/Table S1 D-codes do not consistently correspond to the source script's locally numbered features. Match physical names before assigning final codes; do not relabel by position alone.
- **Efficiency values:** the workbook records eta=86 at x=0.16, while earlier manuscript text used 85.6. At x=0.10 the workbook records 70, whereas an earlier plotted value was around 81. The authors must reconcile measurement conditions, rounding, and original records.
- **Validation language:** feature selection precedes CV and the target-system data overlap the combined dataset. Avoid presenting the associated scores as nested CV or independent external validation. The retrospective GP search is a separate analysis, not a sequential application of the GB descriptor model.
- **Proposed off-line compositions:** the optional y-variable extension has no new experimental validation. It cannot support a claim that the model guided newly performed experiments.

These observations concern the versions available during the code audit. They do not assert that later author-corrected figures still contain the same discrepancies.
