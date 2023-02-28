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

from uyoloseg.models.modules import ConvBN
from uyoloseg.utils.register import registers

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, c1, c2, s=1, downsample=None, no_relu=False) -> None:
        super().__init__()
        self.conv1 = ConvBN(c1, c2, 3, s)
        self.conv2 = ConvBN(c2, c2, 3, act=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)

class BottleNeck(nn.Module):
    expansion = 2
    def __init__(self, c1, c2, s=1, downsample=None, no_relu=False) -> None:
        super().__init__()
        self.conv1 = ConvBN(c1, c2, 1)
        self.conv2 = ConvBN(c2, c2, 3, s)
        self.conv3 = ConvBN(c2, c2 * self.expansion, 1, act=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.no_relu = no_relu
    
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)

class DAPPMBlock(nn.Sequential):
    def __init__(self, c1, c2, k=1, s=1, p=0, d=1, g=1, with_pool=True, adaptive=False):
        super().__init__(
            nn.AdaptiveAvgPool2d(k) if adaptive else nn.AvgPool2d(k, s, p) if with_pool else nn.Identity(),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, 1, bias=False) if with_pool else nn.Conv2d(c1, c2, k, s, p, d, g, bias=False)
        )

class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes):
        super().__init__()
        self.scale0 = DAPPMBlock(inplanes, branch_planes, with_pool=False)
        self.scale1 = DAPPMBlock(inplanes, branch_planes, 5, 2, 2)
        self.scale2 = DAPPMBlock(inplanes, branch_planes, 9, 4, 4)
        self.scale3 = DAPPMBlock(inplanes, branch_planes, 17, 8, 8)
        self.scale4 = DAPPMBlock(inplanes, branch_planes, (1, 1), adaptive=True)

        self.process1 = DAPPMBlock(branch_planes, branch_planes, 3, p=1, with_pool=False)
        self.process2 = DAPPMBlock(branch_planes, branch_planes, 3, p=1, with_pool=False)
        self.process3 = DAPPMBlock(branch_planes, branch_planes, 3, p=1, with_pool=False)
        self.process4 = DAPPMBlock(branch_planes, branch_planes, 3, p=1, with_pool=False)

        self.compression = DAPPMBlock(branch_planes * 5, outplanes, 1, with_pool=False)

        self.shortcut = DAPPMBlock(inplanes, outplanes, 1, with_pool=False)

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]        
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                        size=[height, width],
                        mode='bilinear', align_corners=False) + x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                        size=[height, width],
                        mode='bilinear', align_corners=False) + x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                        size=[height, width],
                        mode='bilinear', align_corners=False) + x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                        size=[height, width],
                        mode='bilinear', align_corners=False) + x_list[3])))
       
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out 

class SegmentHead(nn.Module):
    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super().__init__()
        self.conv1 = ConvBN(inplanes, interplanes, 3)
        self.conv2 = ConvBN(interplanes, outplanes, 1, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv1(x)
        out = self.conv2(x)

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out, size=[height, width], mode='bilinear', align_corners=False)

        return out

class DualResNet(nn.Module):
    def __init__(self, output_dim=19, layers=[2, 2, 2, 2], in_planes=3, planes=64, spp_planes=128, head_planes=128, augment=False):
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
        self.layer4 = self._make_layer(BasicBlock, planes * 4, planes * 8, layers[3], stride=2)
        self.layer5 =  self._make_layer(BottleNeck, planes * 8, planes * 8, 1, stride=2)
        
        self.layer3_ = self._make_layer(BasicBlock, planes * 2, planes * 2, 2)
        self.layer4_ = self._make_layer(BasicBlock, planes * 2, planes * 2, 2)
        self.layer5_ = self._make_layer(BottleNeck, planes * 2, planes * 2, 1)
        

        self.compression3 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 2),
        )

        self.compression4 = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 2),
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(planes * 2, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes * 4, ),
        )

        self.down4 = nn.Sequential(
            nn.Conv2d(planes * 2, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes * 4, planes * 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes * 8),
        )

        self.spp = DAPPM(planes * 16, spp_planes, planes * 4)

        if self.augment:
            self.seghead_extra = SegmentHead(planes * 2, head_planes, output_dim)            

        self.final_layer = SegmentHead(planes * 4, head_planes, output_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def _make_layer(self, block_class, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block_class.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block_class.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block_class.expansion),
            )

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
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8
        layers = []

        x = self.conv1(x)

        x = self.layer1(x)
        layers.append(x)

        x = self.layer2(self.relu(x))
        layers.append(x)
  
        x = self.layer3(self.relu(x))
        layers.append(x)
        x_ = self.layer3_(self.relu(layers[1]))

        x = x + self.down3(self.relu(x_))
        x_ = x_ + F.interpolate(
                        self.compression3(self.relu(layers[2])),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=False)
        if self.augment:
            temp = x_

        x = self.layer4(self.relu(x))
        layers.append(x)
        x_ = self.layer4_(self.relu(x_))

        x = x + self.down4(self.relu(x_))
        x_ = x_ + F.interpolate(
                        self.compression4(self.relu(layers[3])),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=False)

        x_ = self.layer5_(self.relu(x_))
        x = F.interpolate(
                        self.spp(self.layer5(self.relu(x))),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=False)

        x_ = self.final_layer(x + x_)

        logit_list = [x_]

        if self.training or self.augment: 
            x_extra = self.seghead_extra(temp)
            logit_list = [x_extra, x_]

        return [
            F.interpolate(
                logit, [height_output * 8, width_output * 8], mode='bilinear', align_corners=False) for logit in logit_list
        ]

@registers.model_hub.register
def DDRNet23(**kargs):
    return DualResNet(**kargs)

@registers.model_hub.register
def DDRNet23_slim(**kargs):
    return DualResNet(planes=32, head_planes=64, **kargs)

        
if __name__ == '__main__':
    model = DualResNet()
    x = torch.zeros(2, 3, 224, 224)
    outs = model(x)
    for y in outs:
        print(y.shape)
