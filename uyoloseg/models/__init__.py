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
from torch import Tensor
import copy

from .hub import *
from .losses import *

from uyoloseg.utils.register import registers

class FullModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        if cfg.network.name in registers.model_hub:
            network_cfg = copy.deepcopy(cfg.network)
            name = network_cfg.pop("name")
            network_cfg.pop("backbone")
            network_cfg.pop("fpn")
            network_cfg.pop("head")
            self.model = registers.model_hub[name](**network_cfg)
        else:
            assert False, "TODO"

        import pdb; pdb.set_trace()

        if cfg.loss.name in registers.losses:
            self.loss_func = build_loss(cfg.loss)
        else:
            raise ValueError(f"{cfg.losses.name} not exists!!!")

    def forward(self, batch):
        return self.model(batch['img'])

    def forward_train(self, batch):
        logit = self.model(batch['img'])

        if isinstance(logit, list) and len(logit) == 1:
            logit = logit[0]

        if isinstance(logit, list) and len(logit) > 1 and not isinstance(self.loss_func, ComposeLoss):
            raise ValueError("Multi outputs should use ComposeLoss!!!")

        loss = self.loss_func(logit, batch['mask'].squeeze())

        loss_states = {"loss": loss}

        if isinstance(self.loss_func, ComposeLoss):
            loss, loss_states = loss
        
        return logit, loss, loss_states


def build_model(cfg):
    return FullModel(cfg)

def build_weight_averager(cfg):
    pass