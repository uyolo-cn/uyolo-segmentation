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

    def __init__(self, losses, coef):
        super().__init__()
        if not isinstance(losses, list):
            raise TypeError('`losses` must be a list!')
        if not isinstance(coef, list):
            raise TypeError('`coef` must be a list!')
        len_losses = len(losses)
        len_coef = len(coef)
        if len_losses != len_coef:
            raise ValueError(
                'The length of `losses` should equal to `coef`, but they are {} and {}.'
                .format(len_losses, len_coef))

        self.losses = losses
        self.coef = coef

    def forward(self, logits, labels):
        return [w * loss(logits, labels) for w, loss in zip(self.coef, self.losses)]