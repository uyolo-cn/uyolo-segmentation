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

from uyoloseg.models.modules import ConvBN
from uyoloseg.utils.register import registers

from .ddrnet import BasicBlock, DAPPM, DAPPMBlock, SegmentHead

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
    
class ExternalAttention:
    def __init__(self, c1, c2, c3, num_heads=8, use_cross_kv=False) -> None:
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
            self.k = nn.Parameter(torch.normal(std=0.001, size=(c2, c1, 1, 1)), requires_grad=True)
            self.v = nn.Parameter(torch.normal(std=0.001, size=(c3, c2, 1, 1)), requires_grad=True)

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