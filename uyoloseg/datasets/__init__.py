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
import warnings

from custom import CustomDataset
from transforms import *
from collate import *

def build_dataset(cfg, mode):
    dataset_cfg = copy.deepcopy(cfg)
    name = dataset_cfg.pop("name")
    trans = dataset_cfg.pop("transforms")
    ops = build_transforms(trans)
    if name == 'CustomDataset':
        return CustomDataset(task=mode, transform=ops, **dataset_cfg)
    else:
        raise NotImplementedError("Unknown dataset type!")

def build_transforms(cfg):
    transforms_cfg = copy.deepcopy(cfg)
    op_list = []
    for op_cfg in transforms_cfg:
        op_name = op_cfg.pop('op')
        op_list.append(eval(op_name)(**op_cfg))
    ops = Compose(op_list)
    return ops