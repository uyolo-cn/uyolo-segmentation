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

import argparse

import torch
import pytorch_lightning as pl
import onnx
import onnxsim

from uyoloseg.models import build_model
from uyoloseg.utils import cfg, update_config, UYOLOLightningLogger, import_all_modules_for_register

"""
python tools/export.py --model /project/custom/weights/model_best_avg.pth --out_path /project/custom/weights/model_best.onnx --input_shape 576 1024
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default='configs/ddrnet.yaml', type=str, help="train config file path")
    parser.add_argument("--model", type=str, help="model file(.pth) path")
    parser.add_argument("--include", type=str, default='onnx', nargs='*', help="onnx | openvino | etc.")
    parser.add_argument(
        "--out_path", type=str, default="ddrnet23_slim.onnx", help="Onnx model output path."
    )
    parser.add_argument(
        "--input_shape", type=int, nargs=2, help="Model intput shape."
    )
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(args)
    return args

def main():
    args = parse_args()

    logger = UYOLOLightningLogger(cfg.save_dir.log)

    import_all_modules_for_register()

    device = torch.device('cpu' if cfg.device.gpus == -1 else 'cuda')
    
    logger.info("Creating model...")
    
    model = build_model(cfg.model)
    pth = torch.load(args.model, map_location=lambda storage, loc: storage)
    model.load_state_dict(pth["state_dict"])
    model.to(device).eval()

    dummy_input = torch.autograd.Variable(
        torch.randn(1, 3, args.input_shape[0], args.input_shape[1])
    ).to(device)

    if 'onnx' in args.include:
        torch.onnx.export(
            model,
            dummy_input,
            args.out_path,
            verbose=False,
            keep_initializers_as_inputs=True,
            opset_version=11,
            input_names=["data"],
            output_names=["output"],
        )
        logger.log("Finished exporting onnx ")

        logger.log("Start simplifying onnx ")
        input_data = {"data": dummy_input.detach().cpu().numpy()}
        model_sim, flag = onnxsim.simplify(args.out_path, input_data=input_data)
        if flag:
            onnx.save(model_sim, args.out_path)
            logger.log("Simplify onnx successfully")
        else:
            logger.log("Simplify onnx failed")
    else:
        assert False, "TODO"


if __name__ == '__main__':
    main()