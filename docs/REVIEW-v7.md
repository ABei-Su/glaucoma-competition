# v7 independent review and response

Read-only skeptical review found no critical defect and no leakage in the running first internal training job. It did not establish model efficacy or a leaderboard gain.

Two important findings were addressed before internal training completed:

- Completed-run reuse previously trusted a complete status alone. It now verifies every requested recipe field, cache/split fingerprint, checkpoint metadata, epoch and agreement among checkpoint, status and metric reports. Full training takes its batch from the frozen internal recipe.
- Final packaging necessarily consumes an already generated staging CSV, conflicting with the original plan wording. The plan now states that dependency explicitly. A later failure quarantines the CSV and binding manifest; only complete finalization permits staging into the participant delivery directory.

The selection record also freezes capacity/latency/provenance files and the deployed ConvNeXt and fixed v6 reference fingerprints. Final validation rejects changed evidence.

Targeted follow-up confirmed both important findings resolved under the clarified staging/delivery contract, with no remaining critical or important defect. Its minor suggestion to bind the split manifest through the end of full training was also applied: the manifest itself is now included in the frozen evidence checked before final selection. This is static review, not an independent performance replication.

Regression tests reproduced recipe-reuse and failed-package delivery defects before fixes. The complete tools/verify.py run after fixes passed 41 tests, syntax and merge-marker checks (reports/v7_verification.log). No official T4 test or v7 platform score is implied. Original validation reuse and unknown patient clustering remain limitations.
