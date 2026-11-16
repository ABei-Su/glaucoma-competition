# v10 preflight review and execution checks

2026-10-27, before supervised v10 fitting. Independent read-only v10_review inspected the fixed plan, Age transformation/model, training, schema2 inference, capacity/timing and synthetic tests. No images/GPU were accessed by the reviewer and no model quality outcomes existed. The work is exploratory and does not establish clinical generalization or99.3.

Two Important gaps were reproduced and fixed with regression tests before fitting:
- An empty --split-manifest could route non-full training to officialvalid and best-epoch selection. Non-full now requires a nonempty internal manifest; --full requires a fixed selected epoch.
- A pre-invocation changed Age table/checkpoint could previously be accepted by a newly captured fingerprint. Schema2 now requires and verifies exact expected metadata and member SHA256 before loading; changed-file rejection tests cover both.

Two Minor reproducibility suggestions were applied: imported predict.py is included in runtime fingerprints; clinical Predictor pins deterministic cuDNN/benchmark flags to match measured inference and records them in the prediction manifest.

The initial unpinned-contract resource campaign passed (peak3375903232bytes,88.533036ms), and was archived with exact runtime-source aliases in reports/v10_resource_before_hash_pins.json and reports/v10_resource_initial_source_aliases.json. It is not the promotion resource evidence. After the source-contract fixes, a separate fresh final campaign in output/v10/resource_final passed at the same peak3375903232bytes and80.632204333ms, with repeat probability delta0. The rerun is justified by changed pinned input contracts, not by searching timing until a pass; both campaigns passed. Canonical final report is reports/v10_resource_probe.json, whose source/config/model/metadata hashes must match before fitting.

Synthetic tests validate zero-age-head equality, age signal learning, row joins/missing handling, schema2 inference/TTA, exact interrupted-vs-uninterrupted model parameters, transform tampering rejection, completed-run recipe/epoch checks and coordinator promotion guards. Future comparison/full training/final outputs remain subject to their own completion/evidence audits. A status file alone is not proof of live training.

Follow-up review found that aggregate control metrics alone could conceal rank-preserving per-image probability drift. Fixed before any supervised fitting by requiring aligned prior control probabilities to reproduce within absolute1e-7 (the existing prediction/resource numerical tolerance), in addition to aggregate metrics. A synthetic regression rejects rank-preserving probability shifts, misalignment and nonfinite values. The tolerance is frozen with this source before new outcomes; it is not adjusted after comparison.
