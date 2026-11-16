# v7b review record

This is a resource-driven exploratory revision, not an independent efficacy claim. The original full v7 training completed eight epochs and preserved the internally selected epoch6. The original pair+TTA failed 108.904556 ms; the separate Conv/BN fusion trial failed 103.341041 ms despite passing numerical equivalence. Neither was promoted or used to generate test predictions.

Before v7b official-valid evaluation, OPTIMIZATION-v7b.md fixed the original internal second-ranked pair_plain, with the same completed weights, equal probabilities, no TTA and no BN fusion. It passed the unchanged local100+5/three-round/batch1 gate at83.95543466666659 ms. Official T4 remains untested.

Read-only independent review found no critical or important defect in ranking restriction, original recipe/evidence checks, configuration, selection or packaging declarations. Its minor evidence-completeness finding was resolved by packaging four uniquely named original failure/configuration files plus a source/SHA/alias map. A second review confirmed the alias keys, non-overwrite protection and lack of filename collision.

New regression tests first failed, then the full verification suite passed48 tests, syntax and merge-marker checks (reports/v7b_verification_after_review.log, also copied to the canonical verification log while preserving the initial46-test log). Review did not use GPU or images. At review time full-valid comparison was still running; promotion, final packaged timing/limits/CRC/SHA/binding and workbook visual inspection remained pending. Those results must be established separately before upload guidance.

Repeated validation use and unknown patient clustering remain limitations. Platform v7b score is unknown; the user's99.3 target has not been established.
