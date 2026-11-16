"""Explicit torchvision ImageNet backbones with one binary logit."""
import copy
import torch
from torch import nn
from torchvision.models import resnet18,resnet50,convnext_tiny,efficientnet_v2_s


def fuse_inference_batchnorm(model):
    if any(module.training for module in model.modules()):
        raise ValueError('BatchNorm fusion requires eval mode')
    fused=copy.deepcopy(model)
    count=0
    def visit(module):
        nonlocal count
        for child in module.children():visit(child)
        if isinstance(module,nn.Sequential):
            children=list(module.named_children())
            for (conv_name,conv),(norm_name,norm) in zip(children,children[1:]):
                if isinstance(conv,nn.Conv2d) and isinstance(norm,nn.BatchNorm2d):
                    if norm.running_mean is None or norm.running_var is None:
                        raise ValueError('BatchNorm fusion requires frozen running statistics')
                    setattr(module,conv_name,torch.nn.utils.fusion.fuse_conv_bn_eval(conv,norm))
                    setattr(module,norm_name,nn.Identity().eval())
                    count+=1
    visit(fused)
    return fused,count


def probabilities_from_logits(logits):
    # Preserve ranking information near sigmoid saturation under mixed precision.
    return logits.float().sigmoid()


def build_model(pretrain=None,architecture='resnet18'):
    builders={'resnet18':resnet18,'resnet50':resnet50,'convnext_tiny':convnext_tiny,'efficientnet_v2_s':efficientnet_v2_s}
    if architecture not in builders:
        raise ValueError('Unsupported architecture: '+architecture)
    model = builders[architecture](weights=None)
    if pretrain:
        model.load_state_dict(torch.load(pretrain, map_location='cpu', weights_only=True))
    if architecture in ('convnext_tiny','efficientnet_v2_s'):
        model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,1)
    else:
        model.fc = nn.Linear(model.fc.in_features, 1)
    return model
