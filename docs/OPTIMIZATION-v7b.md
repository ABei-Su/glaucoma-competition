# v7b: fixed resource fallback after v7 latency failure

2026-10-26, before evaluating the fallback on full official valid or generating test predictions. This is an exploratory deployment revision to the original v7 plan, which had no final-stage fallback. Preserve v7's 108.904556 ms failure, bounded ~106 ms diagnosis and the rejected 103.341041 ms Conv/BN fusion trial. The fusion trial passed numerical checks but failed resources; do not deploy its fused weights/path.

Use exactly the existing internal ranking's second candidate `pair_plain`, after `pair_tta`. Its recorded internal AUROC .991723494971711 exceeds the fixed internal reference .9903250584431278. It was one of the original four declared, speed-qualified candidates. The fallback is chosen using that preserved internal ranking and the deployment failure, not full-valid accuracy or leaderboard feedback. No further shapes/candidates are appended in this revision.

Reuse the completed original EfficientNet full run's fixed epoch6 checkpoint and fixed v5_full ConvNeXt. Both remain original FP32 parameters, FP16 autocast forward, FP32 sigmoid, probability mean1:1, 384 input, **no TTA and no Conv/BN fusion**. No retraining, new seed, data, threshold or fusion-weight fitting. Keep official valid out of training and use no test images/statistics for selection.

Write only output/v7b and reports/v7b_*; keep output/v7 and all older deliveries intact. Bind the frozen internal evidence, original complete full-run recipe and new candidate to SHA hashes. Require one standard 100+5/three-round/batch1 timing <=100 ms before fresh complete original-image official-valid comparison against fixed v6. Strict AUROC improvement, AP tie-break only; otherwise stop without test CSV. If validation passes, proceed through uncertainty report, test prediction, declarations, final packaged timing, limits, CRC/SHA, binding and participant-directory checks. Official T4 and official samples remain unverified. Validation reuse/patient grouping limitations remain unchanged. User performs upload.

Rollback is the intact v6 98.60 delivery. No 99.3 claim is supported until an actual platform score is returned.
