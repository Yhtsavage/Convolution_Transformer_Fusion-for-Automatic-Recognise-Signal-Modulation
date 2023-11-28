import torch
import torch.nn as nn
from Conform import Conformer
from timm.models.registry import register_model

@register_model
def Conformer_for_2016RMLa(pretrained=False, **kwargs):
    model = Conformer(data_length=128, in_chans=2,  channel_ratio=4, depth=9,
                      num_heads=4, mlp_ratio=4, qkv_bias=True, Device='cuda', **kwargs)
    if pretrained:
        raise NotImplementedError
    return model