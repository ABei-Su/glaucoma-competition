# ADR-004: Training-only internal validation and a preselected final checkpoint

Status: accepted for the user-authorized v5 experiment, 2026-10-25.

The official validation split has been reused in earlier exploratory comparisons. V5 adds an immutable stratified partition of official TRAIN, with a CSV and manifest hash binding. Both internal candidates begin from public ImageNet weights; supervised initialization is rejected when an internal split is requested. Existing competition models cannot act as unexposed internal validation controls.

The training entrypoint accepts optional --split-manifest and --checkpoint-epoch. Defaults preserve earlier behavior. An internal dataset reads images from the official train directory for both partitions and applies augmentation only to the training rows. Resume binds the complete split content through the data fingerprint and checks both new options. A fixed positive checkpoint epoch saves that epoch regardless of official validation rankings; best is absent until the chosen epoch is reached. The cosine schedule still runs for the declared eight epochs.

ConvNeXt-Tiny is an explicit checkpoint architecture. The trusted public 1000-class checkpoint is loaded before replacing the final classifier layer with a binary linear head. Frozen inference dispatches through the existing architecture field; old ResNet checkpoints remain compatible. No new production dependency is added.

The holdout is image-stratified, not patient-stratified, because patient identifiers are unavailable. It reduces reuse of official valid for selecting this new recipe; it does not make the previously inspected official validation independent or establish clinical generalization.

Raw data and older outputs are immutable. V5 writes runs/v5_* and output/v5 only. Rollback is to use the preserved v3 inference bundle and participant delivery. No migration of old weights or regeneration of past submissions is required.
