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

class OhemCrossEntropyLoss(nn.Module):
    '''
    https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.7/paddleseg/models/losses/ohem_cross_entropy_loss.py
    '''
    def __init__(self, thresh: float = 0.7, min_kept: int = 10000, ignore_label: int = 255) -> None:
        super(OhemCrossEntropyLoss, self).__init__()
        self.ignore_label = ignore_label
        self.min_kept = min_kept
        self.thresh = thresh
        self.eps = 1e-8

    def forward(self, logit, label):
        label = label.long().reshape((-1, ))
        mask = (label != self.ignore_label).long()
        num_valid = mask.sum()
        label = label * mask

        mask.required_grad = False
        label.required_grad = False

        prob = F.softmax(logit, dim=1).transpose(1, 0).reshape((logit.shape[1], -1))

        if self.min_kept < num_valid and num_valid > 0:
            prob = prob + (1 - mask)

            prob = (F.one_hot(label, logit.shape[1]).transpose(1, 0) * prob).sum(dim=0)

            threshold = self.thresh
            if self.min_kept > 0:
                index = prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                threshold_index = int(threshold_index)
                if prob[threshold_index] > self.thresh:
                    threshold = prob[threshold_index]
                kept_mask = (prob < threshold).long()
                label = label * kept_mask
                mask = mask * kept_mask

        label = (label + (1 - mask) * self.ignore_label).reshape((logit.shape[0], logit.shape[2], logit.shape[3]))

        mask = mask.reshape(label.shape).float()

        loss = F.cross_entropy(logit, label, ignore_index=self.ignore_label, reduction='none')
        loss = loss * mask
        avg_loss = loss.mean() / (mask.mean() + self.eps)

        return avg_loss

if __name__ == '__main__':
    torch.manual_seed(304)

    input = torch.randn(8, 5, 10, 10, requires_grad=True)

    print(input.dtype)

    target = torch.empty(8, 10, 10, dtype=torch.long).random_(5)

    print(target.dtype)

    loss = OhemCrossEntropyLoss(min_kept=1)

    out = loss(input, target)

    print(out, out.dtype)
    
    # >>> torch.float32
    # >>> torch.int64
    # >>> tensor(1.9575, grad_fn=<DivBackward0>) torch.float32