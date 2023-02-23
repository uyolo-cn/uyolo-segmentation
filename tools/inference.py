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
import torchvision.io as io

from uyoloseg.models import build_model
from uyoloseg.datasets import build_transforms


class Predictor:
    def __init__(self, cfg, model_path, device="cuda:0") -> None:
        self.cfg = cfg
        self.device = device
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(ckpt["state_dict"])
        self.model = model.to(device).eval()
        self.transform = build_transforms(cfg.val_dataset.transforms)

    def inference(self, img):
        if isinstance(img ,str):
            img = io.read_image(img)
        
        mask = torch.zeros(img.shape)

        img, mask = self.transform(img, mask)

        meta = {
            'img': img.unsqueeze(0).to(self.device), 
            'mask': mask.unsqueeze(0).to(self.device)
        }

        with torch.no_grad():
            result = self.model(meta)
        
        return result