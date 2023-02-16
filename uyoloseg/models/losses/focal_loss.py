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
import torch.nn.functional as F
from torch import Tensor

class FocalLoss(nn.Module):
    """
    The implement of focal loss for multi class.
    Args:
        alpha (float, list, optional): The alpha of focal loss. alpha is the weight
            of class 1, 1-alpha is the weight of class 0. Default: 0.25
        gamma (float, optional): The gamma of Focal Loss. Default: 2.0
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
    """

    def __init__(self, alpha=1.0, gamma=2.0, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.eps = 1e-10

    def forward(self, logit: Tensor, label: Tensor):
        """
        Forward computation.
        
        Args:
            logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, C, H, W), where C is number of classes.
            label (Tensor): Label tensor, the data type is int64. Shape is (N, H, W),
                where each value is 0 <= label[i] <= C-1.
        Returns:
            (Tensor): The average loss.
        """
        assert logit.ndim == 4, "The ndim of logit should be 4."
        assert label.ndim == 3, "The ndim of label should be 3."

        ce_loss = F.cross_entropy(
            logit, label, ignore_index=self.ignore_index, reduction='none')

        mask = (label != self.ignore_index).float()

        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt)**self.gamma) * ce_loss * mask

        avg_loss = focal_loss.mean() / (mask.mean() + self.eps)

        return avg_loss
    
if __name__ == '__main__':
    torch.manual_seed(304)

    input = torch.randn(8, 5, 10, 10, requires_grad=True)

    print(input.dtype)

    target = torch.empty(8, 10, 10, dtype=torch.long).random_(5)

    print(target.dtype)

    loss = FocalLoss()

    out = loss(input, target)

    print(out, out.dtype)

    loss1 = nn.CrossEntropyLoss()

    out1 = loss1(input, target)

    print(out1)

    out = out + out1

    out.backward()

    # >>> torch.float32
    # >>> torch.int64
    # >>> tensor(1.4726, grad_fn=<DivBackward0>) torch.float32