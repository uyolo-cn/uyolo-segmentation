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
        indexes(list[(int, int, int)]): corrsponding to (loss_idx, logit_idx, target_idx)
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
        
        loss_indexes = [idx[0] for idx in indexes]
        if max(loss_indexes) >= len(losses) or min(loss_indexes) < 0:
            raise ValueError(
                'The loss number in `indexes` should in [0, {}).'
                .format(len(losses)))

        len_coef = len(coef)
        len_indexes = len(indexes)

        if len_indexes != len_coef:
            raise ValueError(
                'The length of `indexes` should equal to `coef`, but they are {} and {}.'
                .format(len_indexes, len_coef))

        self.losses = losses
        self.coef = coef
        self.indexes = indexes
        self.names = names

    def forward(self, logits, labels):
        if isinstance(logits, Tensor):
            logits = [logits]
        if isinstance(labels, Tensor):
            labels = [labels]
        
        logit_indexes, label_indexes = [idx[1] for idx in self.indexes], [idx[2] for idx in self.indexes]
        if max(logit_indexes) >= len(logits) or min(logit_indexes) < 0:
            raise ValueError(
                'The logit number in `indexes` should in [0, {}).'
                .format(len(logits)))
        if max(label_indexes) >= len(labels) or min(label_indexes) < 0:
            raise ValueError(
                'The label number in `indexes` should in [0, {}).'
                .format(len(labels)))

        if self.names is None:
            names = [f'loss_{i}' for i in range(len(self.indexes))]

        loss_list = [w * self.losses[idx[0]](logits[idx[1]], labels[idx[2]]) for idx, w in zip(self.indexes, self.coef)]

        return sum(loss_list), dict(zip(names, loss_list))