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
            model_path = None
            if 'pretrained' in cfg.network:
                model_path = network_cfg.pop('pretrained')

            self.model = registers.model_hub[name](**network_cfg)

            if model_path:
                state_dict = torch.load(model_path)
                model_dict = self.model.state_dict()
                pretrained_state = {k[6:]: v for k, v in state_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
                model_dict.update(pretrained_state)
                self.model.load_state_dict(model_dict, strict=False)
        else:
            assert False, "TODO"

        self.loss_func = build_loss(cfg.loss)

    def forward(self, batch):
        return self.model(batch['img'])

    def forward_train(self, batch):
        logit = self.model(batch['img'])

        if not isinstance(logit, list):
            raise ValueError("Model outputs must be list type!!!")

        loss, loss_states = self.loss_func(logit, batch['mask'].squeeze(dim=1))
        
        return logit, loss, loss_states


def build_model(cfg):
    return FullModel(cfg)

def build_weight_averager(cfg):
    pass