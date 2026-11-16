# v11b internal gate rejected the budgeted448 candidate

2026-10-28. The separately declared R50_384_TTA+CN448_plain deployment passed its one preflight capacity/timing campaign:4396387328peakallocated bytes,95.15571100000007ms mean, repeatdelta0. The rejected v11 double-TTA102.63348266666662ms campaign and all original sources remain preserved; it was not repeated until favorable.

The new448image-only model completed the frozen8epoch officialtrain internal recipe:43686training images,7710held-out images with249positive cases, seed20261031, batch16, mild augmentation and original public initialization. Best plain epoch8 was used. Original384TTA control probabilities reproduced exactly against their archived per-image reference (maxabsdelta0). Post-comparison source/cache content audits completed before selection was saved.

| Internal fixed predictor | AUROC | Average precision | Trapezoid AUPRC |
| --- | --- | --- | --- |
| Original384ConvNeXt_TTA | 0.9903250584431278 | 0.8266185215952226 | 0.8263327028828563 |
| New448ConvNeXt_plain | 0.9894371750505575 | 0.8284391613975506 | 0.8281715101886296 |

Primary AUROC difference-0.0008878833925702878 (about-0.08879percentage points); AP difference+0.001820639802327939. The exact predeclared rule required strictly better AUROC AND nondecreasing AP. It therefore failed despite the small AP increase. Neither gate, model, epoch, TTA, ratio, threshold nor seed was changed after outcomes. Coordinator and finite finalizer both ended not_promoted without errors, and their own locks were released normally.

No full officialtrain run was started and no v11btest prediction/CSV/delivery was generated. Host process query confirmed related jobs had exited; no stale lock was cleared or new job started. The existing platform98.60 results and prior delivery files remain untouched. This is a local reused-holdout result, not a competition score or evidence that all higher-resolution models fail. It compares a combined resolution/flip-policy change; a pure resolution or clinical/causal claim is unsupported. No claim of statistical significance or equivalence is made.

Independent fresh CPU audit reports/v11b_independent_terminal_check.json recomputed AUROC/AP from saved aligned NPZ predictions, verified row/positive counts and control reproduction, exact completed recipe/history/checkpoint, all frozen source/result evidence and priorv6/v10CSV hashes. It confirmed the unchanged gate and absence of a full run/CSV. No images or model inference were rerun. Full python tools/verify.py freshly passed98tests in7.742s in reports/v11b_terminal_verification.log; original frozen logs were not overwritten.

Independent terminal review by v11_preflight_review found no Critical/Important defect or decision-rule violation. It separately checked the8epoch history/plain best-epoch choice, fixed training scope, source/plan/NPZ hashes, both terminal states and absence of a full run/CSV, and read the successful CPU audit. It did not read images or run GPU/new inference. It confirmed that the point decline cannot be treated as a pure resolution, significance or leaderboard conclusion.

The current experiment has reached its checked terminal state; target99.3 remains unmet and officialT4 unverified. Its periodic supervision was deleted after the terminal audit. Any future method needs a separately fixed question/plan rather than appending hidden candidates to this one.
