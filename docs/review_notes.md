# Review decisions

The author requested consultation with Claude about code sharing and the repository update. Claude reviewed selected source files and the packaging plan; its advice was checked against the code and official journal/platform documentation before adoption. This was an implementation review, not independent scientific validation.

Implemented recommendations:

- Distinguish descriptor regression, retrospective sequential search, and untested off-line proposals in entrypoints, output folders, and documentation.
- Remove personal machine paths, provide a reproducible run command, pin dependencies, and preserve input hashes.
- Correct Extra Trees labels and distinguish censored all-run medians from successful-run-only medians.
- Fail explicitly on non-finite descriptor inputs and verify that unseen outcomes do not influence acquisition.
- Preserve the archived scientific algorithms, seeds, and measurements, then compare fresh output against the saved histories.
- Document dataset overlap, full-data feature selection, figure mapping, and numerical discrepancies requiring author review.

Advice was not accepted automatically. For example, the review inferred that the original shell entrypoint was absent, but it was present in the source snapshot and simply was not included in the selected review packet. The package adds a new portable entrypoint rather than claiming no entrypoint existed. A median is reported as not reached whenever the empirical median cannot be determined within the budget; this also covers the exactly-half-successful case for an even-sized cohort.

No recommendation was used to alter source measurements, introduce new experimental claims, invent a repository license, or assert that a Code Ocean capsule/DOI already exists.
