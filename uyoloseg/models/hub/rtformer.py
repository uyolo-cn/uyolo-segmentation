# ==================================================================
# Copyright (c) 2023, uyolo.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the
# distribution.
# 3. All advertising materials mentioning features or use of this software
# must display the following acknowledgement:
# This product includes software developed by the uyolo Group. and
# its contributors.
# 4. Neither the name of the Group nor the names of its contributors may
# be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY UYOLO, GROUP AND CONTRIBUTORS
# ===================================================================

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from uyoloseg.models.modules import ConvBN, BNConv
from uyoloseg.models.hub import BasicBlock, DAPPM, DAPPMBlock, SegmentHead
from uyoloseg.utils.register import registers

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
 
 
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
 
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class FFN(nn.Module):
    def __init__(self, c1, c2, c3, drop_rate=0.0) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(c1, eps=1e-6)
        self.conv1 = nn.Conv2d(c1, c2, 3, 1, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(c2, c3, 3, 1, 1)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.drop(self.act(self.conv1(self.bn(x))))

        return self.drop(self.conv2(x))
    
class ExternalAttention(nn.Module):
    def __init__(self, c1, c2, c3, num_heads=8, use_cross_kv=False) -> None:
        super().__init__()
        assert c3 % num_heads == 0, "out_channels ({}) should be be a multiple of num_heads ({})".format(c3, num_heads)
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.num_heads = num_heads
        self.use_cross_kv = use_cross_kv
        self.bn = nn.BatchNorm2d(c1)
        if use_cross_kv:
            assert c1 == c3, "in_channels is not equal to out_channels when use_cross_kv is True"
        else:
            self.k = nn.Parameter(torch.normal(0, 0.001, size=(c2, c1, 1, 1)), requires_grad=True)
            self.v = nn.Parameter(torch.normal(0, 0.001, size=(c3, c2, 1, 1)), requires_grad=True)

    def _act_sn(self, x):
        _, _, h, w = x.shape
        x = x.reshape([-1, self.c2, h, w]) * (self.c2 ** (-0.5))
        x = F.softmax(x, dim=1)
        return x.reshape([1, -1, h, w])
    
    def _act_dn(self, x):
        n, _, h, w = x.shape
        x = x.reshape([n, self.num_heads, self.c2 // self.num_heads, -1])
        x = F.softmax(x, dim=3)
        x = x / (torch.sum(x, dim=2, keepdim=True) + 1e-06)
        return x.reshape([n, self.c2, h, w])

    def forward(self, x:Tensor, cross_k=None, cross_v=None):
        """
        Args:
            x (Tensor): The input tensor. 
            cross_k (Tensor, optional): The dims is (n*144, c_in, 1, 1)
            cross_v (Tensor, optional): The dims is (n*c_in, 144, 1, 1)
        """
        x = self.bn(x) # n,c1,h,w
        if not self.use_cross_kv:
            x = F.conv2d(x, self.k, bias=None, stride=(1 if self.c1 == self.c3 else 2)) # n,c2,h,w or n,c2,h/2,w/2
            x =self._act_dn(x)
            x = F.conv2d(x, self.v, bias=None, stride=1) # n,c3,h,w
        else:
            assert (cross_k is not None) and (cross_v is not None), "cross_k and cross_v should no be None when use_cross_kv"
            n, c, h, w = x.shape
            assert n > 0, "The first dim of x ({}) should be greater than 0.".format(n)
            x = x.reshape([1, -1, h, w])
            x = F.conv2d(x, cross_k, bias=None, groups=n)
            x = self._act_sn(x)
            x = F.conv2d(x, cross_v, bias=None, groups=n)
            x = x.reshape([n, -1, h, w])
        
        return x

class EABlock(nn.Module):
    def __init__(self, c1s, c2s, num_heads=8, 
                drop_rate=0.0, drop_path_rate=0.0,
                use_injection=True, use_cross_kv=True,
                cross_size=12) -> None:
        super().__init__()
        c1_h, c1_l = c1s
        c2_h, c2_l = c2s
        assert c1_h == c2_h, "in_channels_h is not equal to out_channels_h"
        self.c2_h = c2_h
        self.proj_flag = c1_l != c2_l
        self.use_injection = use_injection
        self.use_cross_kv = use_cross_kv
        self.cross_size = cross_size

        # low resolution
        if self.proj_flag:
            self.attn_shortcut_l = BNConv(c1_l, c2_l, 1, 2, act=False)
        
        self.attn_l = ExternalAttention(
            c1_l,
            c2_l,
            c2_l,
            num_heads=num_heads,
            use_cross_kv=False
        )

        self.ffn_l = FFN(c2_l, c2_l, c2_l, drop_rate)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.compression = BNConv(c2_l, c2_h, 1)

        # high resolution
        self.attn_h = ExternalAttention(
            c1_h,
            cross_size*cross_size,
            c1_h,
            num_heads=num_heads,
            use_cross_kv=use_cross_kv
        )

        self.ffn_h = FFN(c2_h, c2_h, c2_h, drop_rate)

        if use_cross_kv:
            self.cross_kv = nn.Sequential(
                nn.BatchNorm2d(c2_l),
                nn.AdaptiveMaxPool2d((self.cross_size,self.cross_size)),
                nn.Conv2d(c2_l, 2 * c2_h, 1, bias=False)
            )

        # injection
        if use_injection:
            self.down = nn.Sequential(
                BNConv(c2_h, c2_l // 2, 3, 2),
                BNConv(c2_l // 2, c2_l, 3, 2)
            )
    def forward(self, x):
        x_h, x_l = x

        # low resolution
        x_l_res = self.attn_shortcut_l(x_l) if self.proj_flag else x_l
        x_l = x_l_res + self.drop_path(self.attn_l(x_l))
        x_l = x_l + self.drop_path(self.ffn_l(x_l))  # n,out_chs_l,h,w 

        # compression
        x_h_shape = x_h.shape[2:]
        x_l_cp = self.compression(x_l)
        x_h += F.interpolate(x_l_cp, size=x_h_shape, mode='bilinear')

        # high resolution
        if not self.use_cross_kv:
            x_h = x_h + self.drop_path(self.attn_h(x_h))  # n,out_chs_h,h,w
        else:
            cross_kv = self.cross_kv(x_l)  # n,2*out_channels_h,12,12
            cross_k, cross_v = torch.split(cross_kv, cross_kv.shape[1] // 2, dim=1)
            cross_k = cross_k.permute([0, 2, 3, 1]).reshape(
                [-1, self.c2_h, 1, 1])  # n*144,out_channels_h,1,1
            cross_v = cross_v.reshape(
                [-1, self.cross_size * self.cross_size, 1,1])  # n*out_channels_h,144,1,1
            x_h = x_h + \
                self.drop_path(self.attn_h(x_h, cross_k, cross_v))  # n,out_chs_h,h,w

        x_h = x_h + self.drop_path(self.ffn_h(x_h))

        # injection
        if self.use_injection:
            x_l = x_l + self.down(x_h)

        return x_h, x_l
    
class RTFormer(nn.Module):
    def __init__(self, output_dim=19, layers=[2, 2, 2, 2], in_planes=3, planes=64, 
                 spp_planes=128, head_planes=128, num_heads=8, drop_rate=0., drop_path_rate=0.2, 
                 injection=[True, True], cross_size=12, augment=True) -> None:
        super().__init__()
        
        self.augment = augment

        self.conv1 = nn.Sequential(
            ConvBN(in_planes, planes, 3, 2, bias=True),
            ConvBN(planes, planes, 3, 2, bias=True)
        )

        self.relu = nn.ReLU(inplace=False)

        self.layer1 = self._make_layer(BasicBlock, planes, planes, layers[0])
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, planes * 2, planes * 4, layers[2], stride=2)
        self.layer3_ = self._make_layer(BasicBlock, planes * 2, planes * 2, 1)

        self.compression3 = BNConv(planes * 4, planes * 2, 1)

        self.layer4 = EABlock(
            c1s=[planes * 2, planes * 4],
            c2s=[planes * 2, planes * 8],
            num_heads=num_heads,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            use_injection=injection[0],
            use_cross_kv=True,
            cross_size=cross_size)
        self.layer5 = EABlock(
            c1s=[planes * 2, planes * 8],
            c2s=[planes * 2, planes * 8],
            num_heads=num_heads,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            use_injection=injection[1],
            use_cross_kv=True,
            cross_size=cross_size)
        
        self.spp = DAPPM(planes * 8, spp_planes, planes * 2)

        if self.augment:
            self.seghead_extra = SegmentHead(planes * 2, head_planes, output_dim)            

        self.seghead = SegmentHead(planes * 4, head_planes * 2, output_dim)

    def _make_layer(self, block_class, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block_class.expansion:
            downsample = ConvBN(inplanes, planes * block_class.expansion, 1, stride, act=False)

        layers = []
        layers.append(block_class(inplanes, planes, stride, downsample))
        inplanes = planes * block_class.expansion
        for i in range(1, blocks):
            if i == (blocks-1):
                layers.append(block_class(inplanes, planes, s=1, no_relu=True))
            else:
                layers.append(block_class(inplanes, planes, s=1, no_relu=False))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x1 = self.layer1(self.conv1(x))  # c, 1/4
        x2 = self.layer2(self.relu(x1))  # 2c, 1/8
        x3 = self.layer3(self.relu(x2))  # 4c, 1/16
        x3_ = x2 + F.interpolate(
            self.compression3(x3), size=x2.shape[2:], mode='bilinear')
        x3_ = self.layer3_(self.relu(x3_))  # 2c, 1/8

        x4_, x4 = self.layer4(
            [self.relu(x3_), self.relu(x3)])  # 2c, 1/8; 8c, 1/16
        x5_, x5 = self.layer5(
            [self.relu(x4_), self.relu(x4)])  # 2c, 1/8; 8c, 1/32

        x6 = self.spp(x5)
        x6 = F.interpolate(
            x6, size=x5_.shape[2:], mode='bilinear')  # 2c, 1/8
        x_out = self.seghead(torch.concat([x5_, x6], dim=1))  # 4c, 1/8
        logit_list = [x_out]

        if self.training and self.augment:
            x_out_extra = self.seghead_extra(x3_)
            logit_list.append(x_out_extra)

        logit_list = [
            F.interpolate(
                logit,
                x.shape[2:],
                mode='bilinear',
                align_corners=False) for logit in logit_list
        ]

        return logit_list
    
@registers.model_hub.register
def RTFormer_base(**kargs):
    return RTFormer(**kargs)

@registers.model_hub.register
def RTFormer_slim(**kargs):
    return RTFormer(planes=32, head_planes=64, **kargs)

        
if __name__ == '__main__':
    model = RTFormer()
    model.train()
    x = torch.zeros(2, 3, 224, 224)
    outs = model(x)
    for y in outs:
        print(y.shape)
