"""Training-partition age transforms and an isolated additive image/Age model."""
import hashlib
import json
import math
import numpy as np
import torch
from torch import nn
from data import FundusDataset
from model import build_model


def parse_age(value):
    try:
        age=float(value)
    except (TypeError, ValueError):
        return None
    return age if math.isfinite(age) and 0<=age<=120 else None


def fit_age_stats(rows):
    values=[age for row in rows if (age:=parse_age(row.get('Age'))) is not None]
    if not rows or len(values)<.8*len(rows) or len(set(values))<2:
        raise ValueError('Training Age column has insufficient valid variation/coverage')
    pairs=[[row.get('Image_name',''),row.get('Age')] for row in rows]
    result=dict(schema_version=1,mean=float(np.mean(values,dtype=np.float64)),
                scale=max(1.,float(np.std(values,dtype=np.float64))),
                observed=len(values),missing=len(rows)-len(values),training_rows=len(rows),
                valid_range=[0,120],clip_z=4,
                fitted_rows_sha256=hashlib.sha256(json.dumps(pairs,ensure_ascii=False).encode('utf-8')).hexdigest())
    validate_age_stats(result)
    return result


def validate_age_stats(stats):
    if stats.get('schema_version')!=1 or stats.get('valid_range')!=[0,120] or stats.get('clip_z')!=4:
        raise ValueError('Unknown frozen Age transform schema/range')
    if not math.isfinite(stats['mean']) or not 0<=stats['mean']<=120 or not math.isfinite(stats['scale']) or stats['scale']<1:
        raise ValueError('Invalid frozen Age location/scale')
    for key in ('observed','missing','training_rows'):
        if type(stats[key]) is not int or stats[key]<0:
            raise ValueError('Invalid fitted Age counts')
    if stats['observed']<2 or stats['observed']+stats['missing']!=stats['training_rows']:
        raise ValueError('Misaligned fitted Age counts')
    fingerprint=stats.get('fitted_rows_sha256','')
    if len(fingerprint)!=64 or any(char not in '0123456789abcdef' for char in fingerprint):
        raise ValueError('Missing fitted Age row provenance')


def encode_age(value,stats):
    validate_age_stats(stats)
    age=parse_age(value)
    features=[0.,1.] if age is None else [max(-4.,min(4.,(age-stats['mean'])/stats['scale'])),0.]
    return torch.tensor(features,dtype=torch.float32)


class AgeFundusDataset(FundusDataset):
    def __init__(self,root,split,size,cache,rows,stats,training=False,force_missing=False):
        super().__init__(root,split,size,cache,training,augmentation='mild')
        validate_age_stats(stats)
        self.rows=rows
        self.age_stats=stats
        self.force_missing=force_missing

    def __getitem__(self,index):
        image,label,name=super().__getitem__(index)
        age=None if self.force_missing else self.rows[index].get('Age')
        return image,encode_age(age,self.age_stats),label,name


class AgeConvNeXt(nn.Module):
    def __init__(self,image_model):
        super().__init__()
        self.image=image_model
        self.age=nn.Linear(2,1,bias=False)
        nn.init.zeros_(self.age.weight)

    def forward(self,images,age_features):
        if age_features.ndim!=2 or age_features.shape!=(images.shape[0],2):
            raise ValueError('Age input must be aligned Bx2 features')
        image_logits=self.image(images).flatten().float()
        return image_logits+self.age(age_features.float()).flatten().float()


def build_age_model(pretrain=None):
    return AgeConvNeXt(build_model(pretrain,architecture='convnext_tiny'))


def age_gate(reference,clean,missing):
    values=[reference['auroc'],reference['average_precision'],clean['auroc'],clean['average_precision'],missing['auroc']]
    if any(not math.isfinite(value) or not 0<=value<=1 for value in values):
        raise ValueError('Invalid Age comparison metric')
    auroc_gain=clean['auroc']-reference['auroc']
    ap_gain=clean['average_precision']-reference['average_precision']
    missing_gain=missing['auroc']-reference['auroc']
    return dict(passed=auroc_gain>0 and ap_gain>=-1e-12 and missing_gain>=-.001-1e-12,
                clean_auroc_gain=auroc_gain,clean_ap_gain=ap_gain,missing_age_auroc_gain=missing_gain,
                missing_age_allowed_auroc_drop=.001,minimum_positive_clean_gain_only=True)
