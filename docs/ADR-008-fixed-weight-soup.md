# ADR-008: derived equal-weight ConvNeXt checkpoint

Accepted 2026-10-27 for the user-approved v9 experiment.

Use an isolated utility to average exactly two compatible ConvNeXt state dictionaries with fixed coefficients .5/.5. Require matching model/data/split/training/initialization contract and finite FP32 tensors with identical keys/shapes; reject running-statistic buffers. Compute in float64 and return FP32 without mutating inputs. Store the derived checkpoint separately, with both source paths/SHA256/epochs/coefficients and shared public-initialization evidence. Existing Predictor can load its normal architecture/model/size fields; no reader or schema default changes.

A derived soup is not a trained epoch and must not acquire a fabricated completed-training status/history. Use a composition manifest and explicit source epochs. The source training locks, optimizer states and RNG remain owned by the original training jobs. Never resume training from a composition artifact. No BN calibration, target-data fitting, new runtime dependency or inference model is introduced.

Experiment state: ready -> verifying -> composing -> evaluating -> internal_passed | not_promoted | failed. Only internal_passed permits the separately guarded full_training phase. Once full gates are supported, use existing validation -> prediction -> declaration -> packaging phases; failures never produce a deliverable. Exclusive experiment lock prevents concurrent evaluation; hard termination requires manual PID verification before archival. No automatic retry or lock deletion. Frozen plan and source/input hashes are checked before/after evaluation; interrupted work may only be restarted under the same frozen inputs, with old evidence preserved. A passed comparison alone is not a submission.

Rollback consists of ceasing v9 and retaining v6; all new files are additive. Tests cover arithmetic, no mutation, incompatible initialization/data/state, nonfinite values, known signal after model reload and both outcomes of the promotion gate.
