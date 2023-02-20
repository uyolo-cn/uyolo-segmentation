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

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from uyoloseg.models.modules import ConvBN
from uyoloseg.utils.register import registers

class UpSample(nn.Module):

    def __init__(self, n_chan, factor=2):
        super().__init__()
        out_chan = n_chan * factor * factor
        self.proj = nn.Conv2d(n_chan, out_chan, 1, 1, 0)
        self.up = nn.PixelShuffle(factor)
        self.init_weight()

    def forward(self, x):
        feat = self.proj(x)
        feat = self.up(feat)
        return feat

    def init_weight(self):
        nn.init.xavier_normal_(self.proj.weight, gain=1.)

class DetailBranch(nn.Module):

    def __init__(self):
        super(DetailBranch, self).__init__()
        self.S1 = nn.Sequential(
            ConvBN(3, 64, 3, s=2),
            ConvBN(64, 64, 3)
        )
        self.S2 = nn.Sequential(
            ConvBN(64, 64, 3, s=2),
            ConvBN(64, 64, 3),
            ConvBN(64, 64, 3)
        )
        self.S3 = nn.Sequential(
            ConvBN(64, 128, 3, s=2),
            ConvBN(128, 128, 3),
            ConvBN(128, 128, 3)
        )

    def forward(self, x):
        feat = self.S1(x)
        feat = self.S2(feat)
        feat = self.S3(feat)
        return feat

class StemBlock(nn.Module):

    def __init__(self):
        super(StemBlock, self).__init__()
        self.conv = ConvBN(3, 16, 3, s=2)
        self.left = nn.Sequential(
            ConvBN(16, 8, 1),
            ConvBN(8, 16, 3, s=2)
        )
        self.right = nn.MaxPool2d(3, 2, 1)
        self.fuse = ConvBN(32, 16, 3)

    def forward(self, x):
        feat = self.conv(x)
        feat_left = self.left(feat)
        feat_right = self.right(feat)
        feat = torch.cat([feat_left, feat_right], dim=1)
        feat = self.fuse(feat)
        return feat

class CEBlock(nn.Module):

    def __init__(self):
        super(CEBlock, self).__init__()
        self.bn = nn.BatchNorm2d(128)
        self.conv_gap = ConvBN(128, 128, 1)
        #TODO: in paper here is naive conv2d, no bn-relu
        self.conv_last = ConvBN(128, 128, 3)

    def forward(self, x):
        feat = torch.mean(x, dim=(2, 3), keepdim=True)
        feat = self.bn(feat)
        feat = self.conv_gap(feat)
        feat = feat + x
        feat = self.conv_last(feat)
        return feat