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

class DetailBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.S1 = nn.Sequential(
            ConvBN(3, 64, 3, 2, 1),
            ConvBN(64, 64, 3, 1, 1)
        )
        self.S2 = nn.Sequential(
            ConvBN(64, 64, 3, 2, 1),
            ConvBN(64, 64, 3, 1, 1),
            ConvBN(64, 64, 3, 1, 1)
        )
        self.S3 = nn.Sequential(
            ConvBN(64, 128, 3, 2, 1),
            ConvBN(128, 128, 3, 1, 1),
            ConvBN(128, 128, 3, 1, 1)
        )

    def forward(self, x):
        return self.S3(self.S2(self.S1(x)))


class StemBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_3x3 = ConvBN(3, 16, 3, 2, 1)
        self.left = nn.Sequential(
            ConvBN(16, 8, 1, 1, 0),            
            ConvBN(8, 16, 3, 2, 1)
        )
        self.right = nn.MaxPool2d(3, 2, 1, ceil_mode=False)
        self.fuse = ConvBN(32, 16, 3, 1, 1)

    def forward(self, x):
        x = self.conv_3x3(x)
        x_left = self.left(x)
        x_right = self.right(x)
        y = torch.cat([x_left, x_right], dim=1)
        return self.fuse(y)


class ContextEmbeddingBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inner = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.BatchNorm2d(128),
            ConvBN(128, 128, 1, 1, 0)           
        )
        self.conv = ConvBN(128, 128, 3, 1, 1)   

    def forward(self, x):
        y = self.inner(x)
        out = x + y
        return self.conv(out)


class GatherExpansionLayerv1(nn.Module):
    def __init__(self, in_ch, out_ch, e=6) -> None:
        super().__init__()
        self.inner = nn.Sequential(
            ConvBN(in_ch, in_ch, 3, 1, 1),
            ConvBN(in_ch, in_ch*e, 3, 1, 1, g=in_ch),  
            nn.Conv2d(in_ch*e, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.inner(x)
        out = x + y
        return self.relu(out)


class GatherExpansionLayerv2(nn.Module):
    def __init__(self, in_ch, out_ch, e=6) -> None:
        super().__init__()
        self.inner = nn.Sequential(
            ConvBN(in_ch, in_ch, 3, 1, 1),
            nn.Conv2d(in_ch, in_ch*e, 3, 2, 1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch*e),
            ConvBN(in_ch*e, in_ch*e, 3, 1, 1, g=in_ch*e),               
            nn.Conv2d(in_ch*e, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

        self.outer = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 2, 1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

        self.relu = nn.ReLU(True)

    def forward(self, x):
        x1 = self.inner(x)
        x2 = self.outer(x)
        out = x1 + x2
        return self.relu(out)


class SemanticBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.S1S2 = StemBlock()
        self.S3 = nn.Sequential(
            GatherExpansionLayerv2(16, 32),
            GatherExpansionLayerv1(32, 32)
        )
        self.S4 = nn.Sequential(
            GatherExpansionLayerv2(32, 64),
            GatherExpansionLayerv1(64, 64)
        )
        self.S5_1 = nn.Sequential(
            GatherExpansionLayerv2(64, 128),
            GatherExpansionLayerv1(128, 128),
            GatherExpansionLayerv1(128, 128),
            GatherExpansionLayerv1(128, 128)
        )
        self.S5_2 = ContextEmbeddingBlock()

    def forward(self, x):
        x2 = self.S1S2(x)
        x3 = self.S3(x2)
        x4 = self.S4(x3)
        x5_1 = self.S5_1(x4)
        x5_2 = self.S5_2(x5_1)
        return x2, x3, x4, x5_1, x5_2


class AggregationLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 1, 1, 0, bias=False)
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(3, 2, 1, ceil_mode=False)
        )

        self.right1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=4),
            nn.Sigmoid()
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self.up = nn.Upsample(scale_factor=4)           
        self.conv = ConvBN(128, 128, 3, 1, 1)       


    def forward(self, x_d, x_s):
        x1 = self.left1(x_d)
        x2 = self.left2(x_d)
        x3 = self.right1(x_s)
        x4 = self.right2(x_s)

        left = x1 * x3
        right = x2 * x4
        right = self.up(right)
        out = left + right

        return self.conv(out)


class SegHead(nn.Module):
    def __init__(self, in_ch, mid_ch, num_classes, upscale_factor=8, is_aux=True) -> None:
        super().__init__()
        out_ch = num_classes * upscale_factor * upscale_factor

        self.conv_3x3 = ConvBN(in_ch, mid_ch, 3, 1, 1)
        self.drop = nn.Dropout(0.1)

        if is_aux:
            self.conv_out = nn.Sequential(
                ConvBN(mid_ch, upscale_factor * upscale_factor, 3, 1, 1),
                nn.Conv2d(upscale_factor*upscale_factor, out_ch, 1, 1, 0),
                nn.PixelShuffle(upscale_factor)
            )
        else:
            self.conv_out = nn.Sequential(
                nn.Conv2d(mid_ch, out_ch, 1, 1, 0),
                nn.PixelShuffle(upscale_factor)
            )

    def forward(self, x):
        out = self.conv_3x3(x)
        out = self.drop(out)
        return self.conv_out(out)


class BiSeNetv2(nn.Module):
    def __init__(self, backbone: str = None, num_classes: int = 19) -> None:
        super().__init__()
        self.detail_branch = DetailBranch()
        self.semantic_branch = SemanticBranch()
        self.aggregation_layer = AggregationLayer()

        self.output_head = SegHead(128, 1024, num_classes, upscale_factor=8, is_aux=False)

        self.aux2_head = SegHead(16, 128, num_classes, upscale_factor=4)
        self.aux3_head = SegHead(32, 128, num_classes, upscale_factor=8)
        self.aux4_head = SegHead(64, 128, num_classes, upscale_factor=16)
        self.aux5_head = SegHead(128, 128, num_classes, upscale_factor=32)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_d = self.detail_branch(x)
        aux2, aux3, aux4, aux5, x_s = self.semantic_branch(x)

        output = self.aggregation_layer(x_d, x_s)
        output = self.output_head(output)

        if self.training:
            aux2 = self.aux2_head(aux2)
            aux3 = self.aux3_head(aux3)
            aux4 = self.aux4_head(aux4)
            aux5 = self.aux5_head(aux5)
            return output, aux2, aux3, aux4, aux5
        return output