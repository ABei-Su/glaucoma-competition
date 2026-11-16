# v9 result review

2026-10-27. Read-only skeptical review by v9_red_team under requesting-red-team-review. The reviewer independently read the terminal comparison, source arrays/checker, status and test log, verified all44 evidence entries and the original v6CSV hash, and confirmed no experiment lock, v9 full training or v9 testCSV. No GPU or image inspection was performed by the reviewer.

No remaining Critical/Important issue was identified. CleanAUROC gain .0000427928 and worst-condition gain .0004666838 are both below the fixed .0005 gate. Positive cleanAP gain .0020276005 does not compensate for those failed conjunctive requirements. Independent pair-comparison AUROC and grouped-tie AP agree with the primary calculations; test log records63passes.

The reviewer accepted rejection and the terminal not_promoted state, with limitations: exploratory reused holdout, simulated stresses, unavailable historical initial-head tensors, and untested officialT4/unknown platform score. No clinical, external-generalization or99.3 claim is supported. Review is about this bounded experiment, not proof that all weight averaging is ineffective.
