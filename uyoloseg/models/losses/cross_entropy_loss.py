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

class CrossEntropyLoss(nn.Module):
    '''
    https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.7/paddleseg/models/losses/cross_entropy_loss.py

    Implements the cross entropy loss function.
    Args:
        ignore_label (int64, optional): Specifies a target value that is ignored
                and does not contribute to the input gradient. Default ``255``.
        weight (tuple|list|ndarray|Tensor, optional): A manual rescaling weight
            given to each class. Its length must be equal to the number of classes.
            Default ``None``.
        top_k_percent_pixels (float, optional): the value lies in [0.0, 1.0].
            When its value < 1.0, only compute the loss for the top k percent pixels
            (e.g., the top 20% pixels). This is useful for hard pixel mining. Default ``1.0``.
        label_smoothing (float, optional): the value lies in [0.0, 1.0]. Default ``0.0``.
    '''
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, top_k_percent_pixels: float = 1.0, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.weight = weight
        self.ignore_label = ignore_label
        self.label_smoothing = label_smoothing
        self.top_k_percent_pixels = top_k_percent_pixels
        self.eps = 1e-8
        
    def forward(self, logit, label, semantic_weights=None):

        label = label.long()

        loss = F.cross_entropy(logit, label, 
                               weight=self.weight, 
                               ignore_index=self.ignore_label, 
                               reduction='none', 
                               label_smoothing=self.label_smoothing)
        
        return self.reduce_func(loss, logit, label, semantic_weights)

    def reduce_func(self, loss, logit, label, semantic_weights):
        mask = (label != self.ignore_label)
        mask.required_grad = False
        label.required_grad = False
        
        loss = loss * mask
        
        if semantic_weights is not None:
            loss = loss * semantic_weights
        
        if self.weight is not None:
            _one_hot = F.one_hot(label * mask, logit.shape[1])
            coef = (_one_hot * self.weight).sum(dim=-1)
        else:
            coef = torch.ones_like(label).float()
        
        if self.top_k_percent_pixels == 1.0:
            avg_loss = loss.mean() / (self.eps + (coef * mask).mean())
        else:
            loss = loss.reshape((-1, ))
            top_k_pixels = int(self.top_k_percent_pixels * loss.numel())
            loss, indices = torch.topk(loss, top_k_pixels)
            coef = coef.reshape((-1, ))
            coef = torch.gather(coef, dim=0, index=indices)
            coef.required_grad = False
            avg_loss = loss.mean() / (coef.mean() + self.eps)
        return avg_loss

if __name__ == '__main__':
    torch.manual_seed(304)

    input = torch.randn(8, 5, 10, 10, requires_grad=True)

    print(input.dtype)

    target = torch.empty(8, 10, 10, dtype=torch.long).random_(5)

    print(target.dtype)

    loss = CrossEntropyLoss()
    # loss = CrossEntropyLoss(weight=torch.randn(5), top_k_percent_pixels=0.8)

    out = loss(input, target)

    print(out, out.dtype)

    loss1 = nn.CrossEntropyLoss()

    print(loss1(input, target))

    # >>> torch.float32
    # >>> torch.int64
    # >>> tensor(1.9377, grad_fn=<DivBackward0>) torch.float32
    # >>> tensor(1.9377, grad_fn=<NllLoss2DBackward0>)

    # >>> torch.float32
    # >>> torch.int64
    # >>> tensor(1.1084, grad_fn=<DivBackward0>) torch.float32