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

import copy

from .compose_loss import ComposeLoss
from .cross_entropy_loss import CrossEntropyLoss
from .ohem_cross_entropy_loss import OhemCrossEntropyLoss
from .dice_loss import DiceLoss
from .focal_loss import FocalLoss
from .lovasz_softmax_loss import LovaszSoftmaxLoss

from uyoloseg.utils.register import registers

def build_loss(cfg):
    loss_cfg = copy.deepcopy(cfg)
    name = loss_cfg.pop("name")
    if name != "ComposeLoss":
        return registers.losses[name](**loss_cfg)
    
    compose_losses = loss_cfg.pop("losses")
    losses = [build_loss(l_cfg) for l_cfg in compose_losses]
    return registers.losses[name](losses, **loss_cfg)