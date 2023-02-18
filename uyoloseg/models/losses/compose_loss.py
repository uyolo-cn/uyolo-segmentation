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

from uyoloseg.utils.register import registers

@registers.losses.register
class ComposeLoss(nn.Module):
    """
    Weighted computations for multiple Loss.
    The advantage is that mixed loss training can be achieved without changing the networking code.
    Args:
        losses (list[nn.Layer]): A list consisting of multiple loss classes
        coef (list[float|int]): Weighting coefficient of multiple loss
    Returns:
        A callable object of MixedLoss.
    """

    def __init__(self, losses=None, coef=None, indexes=None, names=None):
        super().__init__()
        if not isinstance(losses, list):
            raise TypeError('`losses` must be a list!')
        if not isinstance(coef, list):
            raise TypeError('`coef` must be a list!')
        if not isinstance(indexes, list):
            raise TypeError('`indexes` must be a list!')
        
        len_losses = len(losses)
        len_coef = len(coef)
        len_indexes = len(indexes)

        if len_losses != len_coef:
            raise ValueError(
                'The length of `losses` should equal to `coef`, but they are {} and {}.'
                .format(len_losses, len_coef))
        
        if len_losses != len_indexes:
            raise ValueError(
                'The length of `losses` should equal to `indexes`, but they are {} and {}.'
                .format(len_losses, len_indexes))

        self.losses = losses
        self.coef = coef
        self.indexes = indexes
        self.names = names

    def forward(self, logits, labels):
        if isinstance(logits, Tensor):
            logits = [logits]
        if isinstance(labels, Tensor):
            labels = [labels]
        
        first_indexes, second_indexes = [idx[0] for idx in self.indexes], [idx[1] for idx in self.indexes]
        if max(first_indexes) >= len(logits) or min(first_indexes) < 0:
            raise ValueError(
                'The first number in `indexes` should in [0, {}).'
                .format(len(logits)))
        if max(second_indexes) >= len(labels) or min(second_indexes) < 0:
            raise ValueError(
                'The second number in `indexes` should in [0, {}).'
                .format(len(labels)))

        if self.names is None:
            names = [f'loss_{i}' for i in range(len(loss_list))]
        loss_list = [w * loss(logits[idx[0]], labels[idx[1]]) for idx, w, loss in zip(self.indexes, self.coef, self.losses)]
        return sum(loss_list), dict(zip(names, loss_list))