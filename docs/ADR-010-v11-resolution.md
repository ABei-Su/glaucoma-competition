# ADR-010: fixed higher-resolution trial without changing frozen modules

Status: accepted as a routine implementation choice under the user's autonomous optimization instruction,2026-10-28. It is not a new public interface or permission to modify existing frozen evidence.

Reuse unchanged train.py/model.py/data.py/predict.py: current schema1 checkpoints already persist input size, architecture, memory layout, split fingerprint and fixed checkpoint epoch; Predictor pads each image once per(size,layout), so mixed384/448members share cropping without assuming equal shapes. Add only the v11 orchestration/guards/reports/tests. No public API/default/schema/dependency changes, old weight migration, training-data mutation or old CSV overwrite.

Internal matched448candidate and conditional full448candidate use the same existing448cache with content audits and immutable old train split. Source evidence guards actual public initialization bytes across the run even though legacy schema1 checkpoint does not itself embed the public hash. Do not retrofit frozen trainer/checkpoint fields. Snapshot all runtime and orchestration source hashes; reject source or recipe changes rather than silently interpreting old artifacts under new code.

Keep quality gates and resource gates explicit, separate, finite and centralized. Only fixed input resolution changes relative to matched image control; no guarantee that a training difference is clinically causal or will transfer to platform performance. Fresh fixed-seed full retraining does not select on officialvalid epochs. A final trialCSV may preserve a disclosed regression under its predeclared policy, not be labeled an improvement.

Failure recovery requires verified-dead recorded PIDs, preserved files and matched RNG/config/data before any explicit continuation. Never clear locks or overwrite terminal records automatically. CSV publication uses reviewed staging then same-volume atomic rename; preserve partial non-delivered evidence on failure. Rollback is simply using the untouched earlier98.60 submission. User alone performs platform upload.
