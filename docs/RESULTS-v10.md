# v10 fixed image/Age trial result

2026-10-28. Exploratory competition experiment under the unchanged OPTIMIZATION-v10 and FINALIZATION-v10 contracts. Original interrupted epoch1 evidence is preserved in output/v10/recovery_001. The explicit recovery repeated the incomplete first epoch with the same seed/recipe; it did not claim an unsaved mid-step resume.

Internal training completed8epochs. Fixed TTA actual-Age AUROC0.9910269680787215/AP0.8358892815299545 versus original image-only control AUROC0.9903250584431278/AP0.8266185215952227. All-missing Age AUROC0.9910135112222109. The registered internal gate passed and locked epoch8 before fresh full officialtrain training with seed20261041. These reused-holdout results do not establish that Age alone caused improvement. Full training used51396officialtrain rows and retained5849officialvalid rows separately; all Age transforms were fitted only on actual training rows.

One fixed equal-probability R50_TTA + full AgeCN_TTA pair was evaluated on the full raw officialvalid split, batch1. Fresh independent CPU recomputation from saved per-image predictions and immutable raw labels matched all reported metrics:

| Fixed predictor | AUROC | Average precision | Trapezoid AUPRC |
| --- | --- | --- | --- |
| v6 reference | 0.9887089401283269 | 0.8664785841273269 | 0.8662225240815388 |
| v10 actual Age | 0.9885284978504235 | 0.8622671366412282 | 0.8619914129970626 |
| v10 all Age missing (diagnostic only) | 0.9884119043785475 | 0.8626508305805163 | 0.8623788663433657 |

v10 point AUROC difference is-0.00018044227790336986 (about-0.01804percentage points), AP difference-0.004211447486098696. The saved1000-replicate paired image-level descriptive AUROC interval is[-0.0009267561660365964,0.0006251630920588553]. It includes zero and is not selection-adjusted or patient-clustered; neither benefit nor equivalence is established. Repeated holdout use, unavailable patient grouping and Age/source confounding remain limitations. No platform score for v10 exists; old platform98.60 cannot be substituted by local validation, and99.3 is not achieved or guaranteed.

Per the plan fixed before outcomes, a slight validation regression was disclosed and did not silently suppress the unique trial CSV. No alternate ratio, epoch, threshold, metadata transform or model was chosen after validation/test/platform feedback. TestAge was only each image's own input under fixed train-only transforms; no test-statistics fitting or test-based selection occurred.

Final local100+5warmup/3rounds/batch1 timing mean80.0444783333478ms, repeat prediction drift0. Necessary trial models/config/metadata/runtime source footprint205839222bytes (the recorded validation-based conservative footprint205844948bytes), under500MB. OfficialT4/Python3.12/torch2.10 environment remains unverified. The delivered directory is a CSV trial with sidecar evidence, not a model核验ZIP.

Delivery: E:\bisai\第十版试交_v10\submission.csv,195967bytes, UTF8,5855unique expected filenames in source order, exact three submission fields, finite[0,1]probabilities and0.5threshold labels. SHA256 e4314595284329f287e46fa25a45eb444ed400a68ec6171124e82eea017ab221. Source/copy/model/metadata/checkpoint/frozen-code bindings and prior v6/v8/v9 CSV hashes independently checked in reports/v10_independent_delivery_check.json. Full tools/verify.py freshly passed89tests in7.597s at delivery; no frozen old verification log was rewritten.

Independent read-only v10_delivery_review checked22frozen/inference evidence hashes plus delivery/config/metadata copies and fixed training scope; no reproducible Critical/Important delivery defect found. It did not read images or run GPU work. Files are available for user-controlled trial upload; preservation/validity is established, improved leaderboard performance is not. The finite training/finalization jobs are complete; periodic supervision ends after this delivery check.
