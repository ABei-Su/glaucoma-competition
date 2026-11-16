"""Validated fixed-loss contract; all focal arithmetic receives FP32 logits."""
import math
import torch
from torch import nn
from torchvision.ops import sigmoid_focal_loss


def loss_spec(gamma=2.,alpha=-1.):
    if (type(gamma) not in (float,int) or not math.isfinite(gamma) or gamma<0 or
            type(alpha) not in (float,int) or not math.isfinite(alpha) or not (alpha==-1 or 0<=alpha<=1)):
        raise ValueError('Invalid focal gamma/alpha')
    return dict(name='sigmoid_focal',gamma=float(gamma),alpha=float(alpha),reduction='mean',input_precision='fp32')


class FocalBinaryLoss(nn.Module):
    def __init__(self,gamma=2.,alpha=-1.):
        super().__init__();self.spec=loss_spec(gamma,alpha)

    def forward(self,logits,labels):
        if logits.shape!=labels.shape or not logits.numel():raise ValueError('Focal logits/labels shape or empty batch')
        x,y=logits.float(),labels.float()
        if not torch.isfinite(x).all() or not torch.isfinite(y).all() or not ((y==0)|(y==1)).all():
            raise ValueError('Focal inputs must be finite and labels binary')
        return sigmoid_focal_loss(x,y,alpha=self.spec['alpha'],gamma=self.spec['gamma'],reduction='mean')
