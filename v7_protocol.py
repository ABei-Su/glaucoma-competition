"""Pure, predeclared equal-weight candidate ranking on internal validation only."""
import numpy as np
from utils import metrics

MODES=('single_plain','single_tta','pair_plain','pair_tta')


def resource_fallback(report):
    candidates=report['candidates']
    if (not report['selected'] or report['selected']['name']!='pair_tta' or len(candidates)<2
            or candidates[0]['name']!='pair_tta' or candidates[1]['name']!='pair_plain'):
        raise ValueError('Fallback requires the unchanged pair_tta then pair_plain internal ranking')
    fallback=candidates[1];reference=report['reference']
    if (fallback['auroc'],fallback['average_precision'])<=(reference['auroc'],reference['average_precision']):
        raise ValueError('Fallback does not improve the fixed internal reference')
    epoch=report['selected']['efficientnet_epoch']
    if type(epoch) is not int or not 1<=epoch<=8:raise ValueError('Invalid frozen checkpoint epoch')
    return dict(fallback,efficientnet_epoch=epoch)


def rank_candidates(labels,new_plain,new_flipped,anchor_plain,anchor_flipped,eligible):
    if set(eligible)!=set(MODES) or any(type(value) is not bool for value in eligible.values()):
        raise ValueError('Exactly four boolean speed-eligibility decisions are required')
    arrays=[np.asarray(value,dtype=np.float64) for value in [new_plain,new_flipped,anchor_plain,anchor_flipped]]
    if any(array.ndim!=1 or len(array)!=len(labels) for array in arrays):raise ValueError('Unaligned internal predictions')
    for array in arrays:metrics(labels,array)
    new,flipped,anchor,anchor_flip=arrays
    new_tta=((new.astype(np.float32)+flipped.astype(np.float32))/2).astype(np.float64)
    anchor_tta=((anchor.astype(np.float32)+anchor_flip.astype(np.float32))/2).astype(np.float64)
    reference=metrics(labels,anchor_tta)
    probabilities=dict(single_plain=new,single_tta=new_tta,pair_plain=(new+anchor)/2,pair_tta=(new_tta+anchor_tta)/2)
    candidates=[dict(name=name,pair=name.startswith('pair_'),tta=name.endswith('_tta'),**metrics(labels,probabilities[name]))
                for name in MODES if eligible[name]]
    candidates.sort(key=lambda item:(item['auroc'],item['average_precision']),reverse=True)
    selected=candidates[0] if candidates and (candidates[0]['auroc'],candidates[0]['average_precision'])>(reference['auroc'],reference['average_precision']) else None
    return dict(candidates=candidates,reference=reference,selected=selected)
