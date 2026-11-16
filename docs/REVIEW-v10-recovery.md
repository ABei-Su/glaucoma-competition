# Recovery and finalization review

2026-10-28. Independent read-only v10_recovery_review inspected the recovery and finalization contracts, source boundaries and synthetic tests. It did not read images, run GPU work or select on quality outcomes. No Git HEAD exists; exact sources are identified by SHA256 in recovery/finalization evidence rather than invented commit IDs.

Three confirmed Important failure paths were fixed before launch:
- A partial finalizer copy could expose submission.csv in the official delivery directory. The new stage_delivery builds and checks files/sidecars inside workspace staging and publishes by a final same-volume rename. A second-copy OSError regression preserves only non-delivered staging, plus a successful-publication/overwrite-rejection test.
- Recovery originally rechecked only old file hashes, missing new last.pt/unknown files/directories added after preflight. Archive now reruns the complete no-checkpoint/allowlist/owner checks, requires equal complete snapshots, and checks the archived file set and bytes. Synthetic tests cover all three newly added content types and preserve source paths on rejection.
- A missing/invalid coordinator status parsed outside try could leak a new recovery guard without recording failure. Parsing now runs inside the guarded exception/finally path; both failure fixtures verify failed status and release of the process's own guard.

Final independent follow-up found no unresolved Critical/Important issues and freshly ran the seven recovery plus three finalization tests (10/10 passed in0.137s), with temporary fixtures explicitly on E:. No competition images/GPU/quality metrics were accessed for that review.

The unchanged original train/compare/full pipeline retains its fixed seeds, original train-only split and Age transforms, exact completed-recipe guards and conditional full-training gate. Recovery never resumes an unsaved mid-epoch state: it restarts the incomplete first epoch from the same public initialization. All interrupted files are archived; no stale lock is silently deleted. Further interruption is not automatically retried.

Verification: tools/verify.py freshly ran 89 tests in 7.599s, all passed, including syntax/merge-marker checks and inference/training integration fixtures; reports/v10_recovery_verification.log. git diff --check returned0 (repository is uncommitted/untracked, so that check alone does not review new-file contents). The original 78-test frozen log remains untouched. The same fresh full verification log is copied to the separately named finalizer verification record before either stage freezes it. Windows host process checks confirmed no Python owner before recovery. Official T4 environment and final quality improvement remain unverified.
