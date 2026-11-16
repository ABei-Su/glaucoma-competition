# ADR-007: Versioned training-only appearance profile

Accepted for user-authorized v8,2026-10-26. Add optional training CLI/config/checkpoint `augmentation` in {mild,color_robust}, default mild. Data transform accepts the same explicit profile. New augmentation is active only when training=True; evaluation always retains original ToTensor/ImageNet normalization. Reject unknown profiles. Legacy checkpoints/configs lacking the field mean mild; resume rejects a changed profile. Preserve old seeds/data split/architecture defaults. No new dependency or preprocessing cache version is needed because cached deterministic images remain unchanged.

Add deterministic, label-independent clean/dim/desaturated stress transforms for internal evaluation only and a pure robust-selection function implementing the predeclared dual.0005 gate. Record all three conditions; no test-time adaptation or cross-image statistics. Tests must prove evaluation equality, old mild-seeded behavior, new-profile reproducibility, invalid-profile rejection and resume mismatch before deployment.

The fixed experiment and resource/promotion gates are OPTIMIZATION-v8.md. Its matched existing control is reused evidence, not a fresh independent replication. The stricter final interval filter remains descriptive. Roll back by choosing default mild and preserved v6; previous models/artifacts are not rewritten.
