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
from torch import nn, Tensor
from torch.nn import functional as F

class DiceLoss(nn.Module):
    '''
    https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.7/paddleseg/models/losses/dice_loss.py
    
    The implements of the dice loss.
    Args:
        weight (list[float], optional): The weight for each class. Default: None.
        ignore_index (int64): ignore_index (int64, optional): Specifies a target value that
            is ignored and does not contribute to the input gradient. Default ``255``.
        smooth (float32): Laplace smoothing to smooth dice loss and accelerate convergence.
            Default: 1.0
    '''
    def __init__(self, weight: Tensor = None, ignore_index: int = 255, smooth: float = 1.0) -> None:
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.eps = 1e-8

    def forward(self, logit: Tensor, label: Tensor):
        num_classes =  logit.shape[1]

        mask = (label != self.ignore_index)

        labels_one_hot: Tensor = F.one_hot(label * mask, num_classes).permute((0, 3, 1, 2))

        prob = F.softmax(logit, dim=1) * mask.unsqueeze(dim=1)

        intersection = (prob * labels_one_hot).sum(dim=[2, 3])

        cardinality = (prob + labels_one_hot).sum(dim=[2, 3])

        dice_loss = 1 - (2 * intersection + self.smooth) / (cardinality + self.smooth + self.eps)

        if self.weight is not None:

            dice_loss = dice_loss.mean(dim=0) * self.weight

        return dice_loss.mean()

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(304)

    input = torch.randn(8, 5, 10, 10, requires_grad=True)

    print(input.dtype)

    target = torch.empty(8, 10, 10, dtype=torch.long).random_(5)

    print(target.dtype)

    loss = DiceLoss()

    out = loss(input, target)

    print(out, out.dtype)

    # >>> torch.float32
    # >>> torch.int64
    # >>> tensor(0.7779, grad_fn=<MeanBackward0>) torch.float32

    out.backward()
