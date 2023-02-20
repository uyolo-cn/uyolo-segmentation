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
import torch
from torch.nn import GroupNorm, LayerNorm
from torch.nn.modules.batchnorm import _BatchNorm

NORMS = (GroupNorm, LayerNorm, _BatchNorm)

def build_optimizer(model, config):
    config = copy.deepcopy(config)
    param_dict = {}

    optim_name = config.pop("name")

    base_lr = config.get("lr", None)
    base_wd = config.get("weight_decay", None)

    no_norm_decay = config.pop("no_norm_decay", False)
    no_bias_decay = config.pop("no_bias_decay", False)
    param_level_cfg = config.pop("param_level_cfg", {})

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        param_dict[p] = {"name": name}
        for key in param_level_cfg:
            if key in name:
                if "lr_mult" in param_level_cfg[key] and base_lr:
                    param_dict[p].update(
                        {"lr": base_lr * param_level_cfg[key]["lr_mult"]}
                    )
                if "decay_mult" in param_level_cfg[key] and base_wd:
                    param_dict[p].update(
                        {"weight_decay": base_wd * param_level_cfg[key]["decay_mult"]}
                    )
                break
    if no_norm_decay:
        # update norms decay
        for name, m in model.named_modules():
            if isinstance(m, NORMS):
                param_dict[m.bias].update({"weight_decay": 0})
                param_dict[m.weight].update({"weight_decay": 0})
    if no_bias_decay:
        # update bias decay
        for name, m in model.named_modules():
            if hasattr(m, "bias"):
                param_dict[m.bias].update({"weight_decay": 0})

    # convert param dict to optimizer's param groups
    param_groups = []
    for p, pconfig in param_dict.items():
        param_groups += [{"params": p, **pconfig}]

    # import pdb; pdb.set_trace()

    optimizer = getattr(torch.optim, optim_name)(param_groups, **config)

    return optimizer