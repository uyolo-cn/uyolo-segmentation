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

from yacs.config import CfgNode

cfg = CfgNode(new_allowed=True)

cfg.device = CfgNode(new_allowed=True)
cfg.device.precision = 32

cfg.save_dir = "./"

# Model
cfg.model = CfgNode(new_allowed=True)

cfg.model.network = CfgNode(new_allowed=True)

cfg.model.network.backbone = CfgNode(new_allowed=True)
cfg.model.network.fpn = CfgNode(new_allowed=True)
cfg.model.network.head = CfgNode(new_allowed=True)

# Dataset
cfg.train_dataset = CfgNode(new_allowed=True)

cfg.val_dataset = CfgNode(new_allowed=True)

# Train Settings

cfg.schedule = CfgNode(new_allowed=True)

cfg.evaluator = CfgNode(new_allowed=True)

# logger
cfg.log = CfgNode()
cfg.log.interval = 50

def update_config(cfg, args):
    cfg.defrost()
    
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()

