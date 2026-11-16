# v7b validation result

The fixed no-TTA equal EfficientNetV2-S/ConvNeXt pair passed its declared validation promotion gate. It reuses original full-trained epoch6 EfficientNet and fixed v5_full ConvNeXt, with no BN fusion or additional training. The original TTA and fusion latency failures remain preserved separately.

Complete original-image batch1 official validation, 5849 images/191 positives:

| Metric | Fixed v6 | v7b |
| --- | ---: | ---: |
| AUROC | 0.9887089401283269 | 0.9900543917799751 |
| Average precision | 0.8664785841273269 | 0.8734908563241087 |
| Trapezoid AUPRC | 0.8662225240815388 | 0.8732397434426737 |

The standard local100+5/three-round/batch1 initial gate measured83.95543466666659ms. This is not official T4 verification. An independent CSV/name/label/threshold check reproduced both stored metrics and the prior v6 AUROC exactly (output/v7b/independent_validation_check.json).

Paired stratified1000-bootstrap descriptive95% intervals: AUROC gain .0013454516516482329, interval[-.0012689718861676996,.004742971541939496]; AP gain .007012272196781799, interval[-.007204554951206671,.02176624746189839]. These intervals include zero, do not correct model-selection bias or unknown patient clustering, and do not establish a reliable test-set gain. Official valid was reused for exploration. No clinical efficacy claim or platform99.3 claim follows.

At the time of this result record, test prediction and final packaging were still running. Only complete finalization, packaged timing/limits/CRC/SHA/binding checks, independent delivery verification and declaration-preview review allow upload guidance. The platform score remains unknown; v6 98.60 and previous deliveries remain intact.
