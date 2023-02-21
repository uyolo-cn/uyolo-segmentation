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
from torch.nn import functional as F

from uyoloseg.models.modules import ConvBN, DwConvBN
# from uyoloseg.utils.register import registers

# This module or resize op?
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

    def __init__(self, in_channels=3, feature_channels=[64, 64, 128]):
        super().__init__()
        c1, c2, c3 = feature_channels
        self.stage = nn.Sequential(
            # stage 1
            ConvBN(in_channels, c1, 3, s=2),
            ConvBN(c1, c1, 3),
            # stage 2
            ConvBN(c1, c2, 3, s=2),
            ConvBN(c2, c2, 3),
            ConvBN(c2, c2, 3),
            # stage 3
            ConvBN(c2, c3, 3, s=2),
            ConvBN(c3, c3, 3),
            ConvBN(c3, c3, 3)
        )

    def forward(self, x):
        return self.stage(x)

class StemBlock(nn.Module):

    def __init__(self, in_dim=3, out_dim=16):
        super().__init__()
        self.conv = ConvBN(in_dim, out_dim, 3, s=2)
        self.left = nn.Sequential(
            ConvBN(out_dim, out_dim // 2, 1),
            ConvBN(out_dim // 2, out_dim, 3, s=2)
        )
        self.right = nn.MaxPool2d(3, 2, 1)
        self.fuse = ConvBN(out_dim * 2, out_dim, 3)

    def forward(self, x):
        feat = self.conv(x)
        feat_left = self.left(feat)
        feat_right = self.right(feat)
        feat = torch.cat([feat_left, feat_right], dim=1)
        feat = self.fuse(feat)
        return feat
    
class ContextEmbeddingBlock(nn.Module):
    def __init__(self, in_dim=128, out_dim=128):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = nn.BatchNorm2d(in_dim)
        self.conv_gap = ConvBN(in_dim, out_dim, 1)
        self.conv_last = nn.Conv2d(out_dim, out_dim, 3)

    def forward(self, x):
        feat = self.bn(self.gap(x))
        feat = self.conv_gap(feat) + x
        return self.conv_last(feat)
    
class GatherAndExpansionLayer1(nn.Layer):
    """Gather And Expansion Layer with stride 1"""
    def __init__(self, in_dim, out_dim, expand):
        super().__init__()
        expand_dim = expand * in_dim
        self.conv = nn.Sequential(
            ConvBN(in_dim, in_dim, 3),
            DwConvBN(in_dim, expand_dim, 3),
            ConvBN(expand_dim, out_dim, 1, act=False))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + x)
    
class GatherAndExpansionLayer2(nn.Layer):
    """Gather And Expansion Layer with stride 2"""

    def __init__(self, in_dim, out_dim, expand):
        super().__init__()

        expand_dim = expand * in_dim

        self.branch_1 = nn.Sequential(
            ConvBN(in_dim, in_dim, 3),
            DwConvBN(in_dim, expand_dim, 3, stride=2),
            DwConvBN(expand_dim, expand_dim, 3),
            ConvBN(expand_dim, out_dim, 1, act=False))

        self.branch_2 = nn.Sequential(
            DwConvBN(
                in_dim, in_dim, 3, stride=2),
            ConvBN(in_dim, out_dim, 1, act=False))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.branch_1(x) + self.branch_2(x))
    
class SemanticBranch(nn.Layer):
    """The semantic branch of BiSeNet, which has narrow channels but deep layers."""

    def __init__(self, in_channels, feature_channels):
        super().__init__()
        c1_2, c3, c4, c5 = feature_channels

        self.stage1_2 = StemBlock(in_channels, c1_2)

        self.stage3 = nn.Sequential(
            GatherAndExpansionLayer2(c1_2, c3, 6),
            GatherAndExpansionLayer1(c3, c3, 6))

        self.stage4 = nn.Sequential(
            GatherAndExpansionLayer2(c3, c4, 6),
            GatherAndExpansionLayer1(c4, c4, 6))

        self.stage5_4 = nn.Sequential(
            GatherAndExpansionLayer2(c4, c5, 6),
            GatherAndExpansionLayer1(c5, c5, 6),
            GatherAndExpansionLayer1(c5, c5, 6),
            GatherAndExpansionLayer1(c5, c5, 6))

        self.ce = ContextEmbeddingBlock(c5, c5)

    def forward(self, x):
        stage1_2 = self.stage1_2(x)
        stage3 = self.stage3(stage1_2)
        stage4 = self.stage4(stage3)
        stage5_4 = self.stage5_4(stage4)
        fm = self.ce(stage5_4)
        return stage1_2, stage3, stage4, stage5_4, fm
    
class BGA(nn.Layer):
    """The Bilateral Guided Aggregation Layer, used to fuse the semantic features and spatial features."""

    def __init__(self, out_dim, align_corners=False):
        super().__init__()

        self.align_corners = align_corners

        self.db_branch_keep = nn.Sequential(
            DwConvBN(out_dim, out_dim, 3),
            nn.Conv2d(out_dim, out_dim, 1))

        self.db_branch_down = nn.Sequential(
            ConvBN(out_dim, out_dim, 3, stride=2, act=False),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1))

        self.sb_branch_keep = nn.Sequential(
            DwConvBN(out_dim, out_dim, 3),
            nn.Conv2d(out_dim, out_dim, 1),
            nn.Sigmoid())

        self.sb_branch_up = ConvBN(out_dim, out_dim, 3, act=False)

        self.conv = ConvBN(out_dim, out_dim, 3, act=False)

    def forward(self, dfm, sfm):
        db_feat_keep = self.db_branch_keep(dfm)
        db_feat_down = self.db_branch_down(dfm)
        sb_feat_keep = self.sb_branch_keep(sfm)

        sb_feat_up = self.sb_branch_up(sfm)
        # UpSample ???
        sb_feat_up = F.interpolate(
            sb_feat_up,
            db_feat_keep.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        sb_feat_up = F.sigmoid(sb_feat_up)
        db_feat = db_feat_keep * sb_feat_up

        sb_feat = db_feat_down * sb_feat_keep
        # UpSample ???
        sb_feat = F.interpolate(
            sb_feat,
            db_feat.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        return self.conv(db_feat + sb_feat)

class SegmentHead(nn.Module):
    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBN(inplanes, interplanes, 3),
            nn.Dropout(0.1),
            nn.Conv2d(interplanes, outplanes, 1)
        )
        self.scale_factor = scale_factor

    def forward(self, x):
        out = self.conv(x)

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out, size=[height, width], mode='bilinear', align_corners=False)

        return out
    
class BiSeNetV2(nn.Module):
    """
    The BiSeNet V2 implementation based on PaddlePaddle.
    The original article refers to
    Yu, Changqian, et al. "BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation"
    (https://arxiv.org/abs/2004.02147)
    Args:
        num_classes (int): The unique number of target classes.
        lambd (float, optional): A factor for controlling the size of semantic branch channels. Default: 0.25.
        in_channels (int, optional): The channels of input image. Default: 3.
    """

    def __init__(self,
                 num_classes=19,
                 in_channels=3,
                 lambd=0.25,
                 align_corners=False):
        super().__init__()

        c1, c2, c3 = 64, 64, 128
        db_channels = (c1, c2, c3)
        c1, c3, c4, c5 = int(c1 * lambd), int(c3 * lambd), 64, 128
        sb_channels = (c1, c3, c4, c5)
        mid_channels = 128

        self.db = DetailBranch(in_channels, db_channels)
        self.sb = SemanticBranch(in_channels, sb_channels)

        self.bga = BGA(mid_channels, align_corners)
        self.aux_head1 = SegmentHead(c1, c1, num_classes)
        self.aux_head2 = SegmentHead(c3, c3, num_classes)
        self.aux_head3 = SegmentHead(c4, c4, num_classes)
        self.aux_head4 = SegmentHead(c5, c5, num_classes)
        self.head = SegmentHead(mid_channels, mid_channels, num_classes)

        self.align_corners = align_corners

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        dfm = self.db(x)
        feat1, feat2, feat3, feat4, sfm = self.sb(x)
        logit = self.head(self.bga(dfm, sfm))

        if not self.training:
            logit_list = [logit]
        else:
            logit1 = self.aux_head1(feat1)
            logit2 = self.aux_head2(feat2)
            logit3 = self.aux_head3(feat3)
            logit4 = self.aux_head4(feat4)
            logit_list = [logit, logit1, logit2, logit3, logit4]

        logit_list = [
            F.interpolate(
                logit,
                x.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners) for logit in logit_list
        ]

        return logit_list
    
if __name__ == '__main__':
    model = BiSeNetV2()
    x = torch.zeros(2, 3, 224, 224)
    outs = model(x)
    for y in outs:
        print(y.shape)
