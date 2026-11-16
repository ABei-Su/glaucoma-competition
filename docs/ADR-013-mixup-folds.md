# ADR-013: isolated three-fold Mixup experiment

Accepted 2026-10-29 under the user's instruction to plan and execute autonomously.
The plan is docs/science-superpowers/plans/2026-10-29-breakthrough.md.
No legacy source, model, submission or frozen evidence is changed.

New v13_core owns schema2 fold validation and independent Mixup RNG. Schema2 is
a training partition contract, not an inference checkpoint format. Checkpoints
remain image-only schema1 with additional training provenance. The new trainer
starts fresh from the pinned public weights for each paired fold, fixes epoch8,
and persists all RNG states, configuration, source/data/public hashes on resume.
Mixup is applied only during training, using independent Beta/permutation RNGs;
control and candidate share data-loader and augmentation streams. Validation
uses fixed horizontal-flip probability averaging. No patient identity is inferred.

The coordinator has an exclusive operation lock and centralized transitions:
implementation -> preflight -> fold_training -> comparison -> full_training ->
finalization -> awaiting_review -> complete. failed, not_promoted and budget_exceeded are terminal.
awaiting_review is a stable handoff: the coordinator exits and releases its lock.
Only a separately written independent CPU/reviewer evidence record can be accepted
by the explicit accept_delivery command under a new exclusive lock. Staged CSV
presence alone never means complete or independently approved. Review rejection
leaves all evidence intact and does not trigger new fitting or prediction.
Each transition records an increasing revision and timestamp. Illegal/replayed
commands are rejected; a second invocation refuses existing output/status.
There is no automatic retry, stale-lock removal or terminal overwrite. The
12-hour deadline includes preflight and all six fits; each subprocess has at most
3 hours. Only a complete promoted comparison permits the separate 4-hour full
training/delivery budget. Timeout terminates the owned subprocess tree and
preserves its logs/checkpoints, never synthesizing a successful result.

Guards: frozen source/data hashes, exactly three paired completed epoch8 fits,
complete mutually exclusive OOF names and labels, all fixed quality gates and
actual resource limits. Test images are only read after final model/config freeze.
Recovery requires confirmed dead owners, intact evidence and exact configuration;
it is a distinct reviewed action, not an automatic new experiment.

Rollback is the retained 98.60 submission. No API/dependency change, external
upload or official T4 verification is implied. CPU tests cover schema rejection,
random-stream independence, loss and resume equivalence; actual resource testing
is required before accepting the training campaign.
