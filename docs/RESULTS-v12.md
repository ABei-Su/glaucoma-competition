# v12 fixed Focal trial delivery

Completed and independently checked 2026-10-28. The fixed candidate is ResNet50/384/horizontal-flip TTA plus ConvNeXt-Tiny/384/Focal/horizontal-flip TTA, equal probability mean, fixed threshold0.5. There was no final official-valid model, ratio, threshold or checkpoint selection. Internal epoch selection and qualification did occur, as specified before fitting; `model_selection_performed:false` in the final report refers only to the final comparison stage.

## Execution and validation

- Internal43686/7710: new AUROC0.9908267300538436 and AP0.8379646395590242; original384BCE_TTA control AUROC0.9903250584431278 and AP0.8266185215952226; original control reference probability delta0. Fixed gate passed and locked epoch8.
- Fresh full officialtrain51396, seed20261032, 8-epoch schedule, checkpoint8. Officialvalid did not enter training. Actual checkpoint carries sigmoidFocal gamma2/alpha-1/mean/FP32 loss metadata and public initialization SHA983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d.
- Final fixed pair, all5849officialvalid raw images, batch1 actual inference: AUROC0.9898434131165804, AP0.8629619405118919, trapezoidPR0.8626745386578814.
- Fresh fixed-v6 control: AUROC0.9887089401283269, AP0.8664785841273269, trapezoidPR0.8662225240815388.
- AUROC difference+0.001134472988253532 (+0.1134472988253532percentage points). AP difference-0.0035166436154350222 (-0.3516643615435022percentage points). AUROC improved as a local point estimate; AP decreased. Do not claim all metrics improved.
- Saved image-level paired1000-bootstrap descriptive AUROC difference95% interval[-0.0009217593029560267,0.0036101873083378196]; AP[-0.012055992840323198,0.0053132098714528874]. Both cross0. Reused validation and post-selection, no selection correction or patient clustering. Not confirmatory superiority, clinical validation or official test performance.

## Resources and deliverable

- One actual trained-pair local timing campaign:100images+5warmup,batch1,3rounds,mean87.94682266669649ms,repeatdelta0. CPU recomputation from all300saved timing records matched. No monitoring inference rerun.
- Conservative resource record205706779B includes runtime source code, both weights and config; weights+config alone205679794B. Both below500MB. OfficialT4/Python3.12/PyTorch2.10 environment and organizer-provided timing images remain unverified.
- User-uploadable CSV: `E:/bisai/第十二版试交_v12/submission.csv`,189588B,5855rows,SHA256 `ae5be4bc90f8d4b19fda32142ea61abe3ba8ee4528f8436bb6c5b465d978ebf2`.
- UTF8, exact3headers, unique names and exact officialtest.csv case/order, finite probabilities[0,1], labels from>=0.5 all checked. Windows disk images have uppercase.JPG; mapping is case-insensitive while the CSV preserves exact official lowercase.jpg identifiers.
- Canonical, copiedCSV, sidecar config/manifest/validation/timing/bootstrap records hash-match. Model/runtime/source evidence and metadata verified;190unique frozen files matched. Oldv6/v8/v9/v10CSV hashes unchanged.
- Full repository verification:108tests passed9.768s, syntax and merge-marker scans passed, log`reports/v12_terminal_verification.log`. Frozen launch verification log was preserved.
- Independent CPU auditor`tmp/check_v12_delivery_cpu.py` and result`reports/v12_independent_delivery_check.json`. Two auditor assumptions corrected after source inspection: disk.JPG vs officialCSV.jpg case mapping, and conservative resource report including runtime source files. No delivered artifacts, production code, recipe or frozen evidence changed to address these auditor assumptions.
- Independent skeptical reviewer found noCritical blocking issue. Important reporting limitations are stated above; minor ambiguity in final-stage selection metadata is clarified without changing frozen reports.

This is the one predeclared trialCSV, not a model核验ZIP. User alone uploads. Platform score is unknown; no statement of exceeding98.60 or reaching99.3. The completed finite experiment does not authorize unattended candidate searches, scientific retries or any modification of the historical delivery.

## Subsequent platform report

User subsequently supplied screenshot of submission23686,2026-10-28 21:57:54 Asia/Shanghai,status已正确评分,displayed score98.50. See `reports/twelfth_submission_result.json` and archived `reports/v12_platform_9850.png`. Context associates this submission with the deliveredv12; localCSV SHA reverified unchanged, but uploaded bytes cannot be independently verified from a screenshot. Compared with the reported historical best98.60, the displayed result is0.10points lower. This trial did not improve the displayed platform result. Earlier statements of platform score unknown refer to the pre-upload delivery time.

The verified local validation point gain did not transfer to the reported platform result. Root cause is undetermined; this observation alone cannot identify a code defect, distribution difference, reused-validation selection effects, or a loss-specific effect. No test labels are available for deterministic reproduction or subgroup diagnosis. Do not fit subsequent recipes to this score, change existing frozen artifacts, or claim all Focal methods fail. History/evidence archived; no new training or resource retest was started from this feedback. Original98.60CSV preserved and hash rechecked.
