"""Deterministic internal color stresses and predeclared robust ranking."""
import math
from torchvision.transforms import functional as F
from data import tensor_transform

CONDITIONS=('clean','dim','desaturated')


def conservative_upgrade(auroc_gain,descriptive_lower_bound):
    if not all(math.isfinite(value) for value in (auroc_gain,descriptive_lower_bound)):
        raise ValueError('Nonfinite upgrade statistic')
    return auroc_gain>=.0005-1e-12 and descriptive_lower_bound>0


class StressTransform:
    def __init__(self,condition):
        if condition not in CONDITIONS:raise ValueError('Unknown fixed stress condition')
        self.condition=condition;self.transform=tensor_transform(False)

    def __call__(self,image):
        if self.condition=='dim':image=F.adjust_brightness(image,.70)
        elif self.condition=='desaturated':image=F.adjust_saturation(image,.50)
        return self.transform(image)


def choose_robust_candidate(reference,candidates,eligible,epoch):
    if type(epoch) is not int or not 1<=epoch<=8:raise ValueError('Invalid selected epoch')
    if set(candidates)!={'plain','tta'} or set(eligible)!={'plain','tta'} or any(type(v) is not bool for v in eligible.values()):
        raise ValueError('Exactly two boolean resource decisions required')
    for metrics in [reference]+list(candidates.values()):
        if set(metrics)!=set(CONDITIONS):raise ValueError('Missing fixed stress condition')
        for item in metrics.values():
            if any(not math.isfinite(item[key]) or not 0<=item[key]<=1 for key in ('auroc','average_precision')):
                raise ValueError('Invalid robustness metric')
    reference_worst=min(item['auroc'] for item in reference.values());ranked=[]
    for mode,values in candidates.items():
        if not eligible[mode]:continue
        worst=min(item['auroc'] for item in values.values())
        clean_gain=values['clean']['auroc']-reference['clean']['auroc'];worst_gain=worst-reference_worst
        ranked.append(dict(mode=mode,tta=mode=='tta',epoch=epoch,metrics=values,worst_auroc=worst,
                           clean_gain=clean_gain,worst_gain=worst_gain,qualified=min(clean_gain,worst_gain)>=.0005-1e-12))
    ranked.sort(key=lambda item:(item['worst_auroc'],item['metrics']['clean']['auroc'],item['metrics']['clean']['average_precision']),reverse=True)
    selected=next((item for item in ranked if item['qualified']),None)
    return dict(reference=reference,reference_worst_auroc=reference_worst,candidates=ranked,selected=selected,
                minimum_clean_and_worst_gain=.0005,source='reused train holdout; synthetic stress is not external validation')
