# ADR-005: Add EfficientNetV2-S as an explicit binary checkpoint architecture

Status: accepted for the user-authorized v7 experiment, 2026-10-26.

The existing builder dispatch and checkpoint architecture field gain an additive efficientnet_v2_s value. Official public ImageNet weights are loaded into the original 1000-class model before replacing the final classifier Linear layer with one binary logit. Its existing dropout and BatchNorm obey train/eval and frozen inference through Predictor. ImageNet normalization and 384 image cache remain unchanged.

The only new CLI value is explicit; old defaults and ResNet/ConvNeXt weights remain compatible. Unknown architecture values fail. Internal validation, immutable split binding, preselected checkpoint epoch and resume checks reuse the v5 mechanisms. No new production dependency, new data boundary or change to the submission CSV schema is introduced.

Hardware capacity and actual raw-image latency are measured before fitting. Only predeclared, speed-qualified single/equal-pair configurations may enter internal selection. Final comparison is against the fixed v6 inference bundle using official valid, without leaderboard fitting. Both heldout reuse and unknown patient clustering remain limitations.

Rollback is to the preserved v6 bundle; all v7 models and outputs occupy new directories. Existing weights and delivery files are not rewritten.

Review hardening, before the first v7 test prediction: completed-run reuse validates the requested recipe, cache/split fingerprint, checkpoint metadata and epoch metrics. Internal selection freezes resource/provenance evidence and final reference weights; full training uses the frozen internal batch. Packaging consumes a staging CSV after the validation promotion gate. Only complete finalization permits delivery; later failures quarantine the CSV and its manifest as unverified. This clarifies the original plan's impossible package-before-CSV ordering without changing model selection or data boundaries.
