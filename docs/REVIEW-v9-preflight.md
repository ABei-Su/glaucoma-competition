# v9 preflight review

2026-10-27, before any v9 candidate evaluation. Independent reviewer v9_red_team performed read-only source inspection under requesting-red-team-review, without GPU or image access. No Git HEAD exists; no commit IDs are claimed.

Initial Important findings:
1. Startup could overwrite a terminal failed status. Fixed with a fresh-experiment guard before lock/state initialization, rejecting every existing status or partial output and preserving bytes. Regression tests exercise failed/internal-passed statuses and partial artifacts. Exclusive lock handles concurrent fresh launches.
2. Post-evaluation integrity checked metadata instead of image bytes. Fixed by repeating the existing complete raw/cache content audit before promotion, comparing the audit record and freezing its actual manifest. This verifies integrity without fitting validation data; metadata distinguishes validation-integrity access from validation-metric evaluation.

Reviewer re-inspected both fixes and reported no remaining actionable Critical/Important issues. FP64 arithmetic, FP32 output, no mutation, fixed AP/AUROC gates and reused-holdout limitations were acceptable. Historical binary-head initial bytes were not saved; matched seed/public hash/current construction is reconstruction evidence, not a historical byte-level guarantee. No claim of successful averaging, external generalization or 99.3 follows from preflight review. Full suite must pass after final edits before freezing and execution.
