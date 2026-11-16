# v16 result: retain v15 as the preferred local candidate

The fixed four-epoch low-rate stage from the locked v15 R50 completed. Trainer and finalizer exited0; finalizer reached awaiting_review. Same rawvalid Predictor path, all5849images, no model or ratio search. Independent CPU delivery review is recorded separately in reports/v16_delivery_review.json when approved.

| Shape | v15 AUROC | v16 AUROC | v15 AP | v16 AP |
| --- | ---: | ---: | ---: | ---: |
| R50 horizontalTTA | .9899331715830246 | .9875365279944627 | .8622196739018287 | .855847623973374 |
| R50 + CN8 equal horizontalTTA | .9902773999285633 | .9889550819022872 | .8669222708320721 | .8652156790872294 |

Both shapes fail the predefined local joint-improvement criterion: both AUROC and AP declined. The result does not identify a causal mechanism; no further seeds, epochs, weights or thresholds were tried. Official validation has been repeatedly reused, so this is exploratory evidence, not independent confirmation or a platform score.

The one-shot local resource checks pass: single80.9081393333ms/94340981B; pair87.0755800000ms/205679566B; all-round repeat delta0 as recorded. Official T4 not verified. v16 trials were still exported under the predeclared user-authorized policy, separately in E:/bisai/第十六版试交_v16. Saved predictions and delivery/model/source bindings are verified on CPU without new inference. The preserved v15 originals and tomorrow backups remain the preferred local candidates. User performs scoring upload; platform outcomes remain unknown. v15 material drafts cannot stand in for the different v16 8+4 training chain.
