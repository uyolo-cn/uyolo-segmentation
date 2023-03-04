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
from pytorch_lightning.callbacks import TQDMProgressBar

from uyoloseg.datasets import build_dataset, naive_collate
from uyoloseg.utils import cfg, update_config, UYOLOLightningLogger, import_all_modules_for_register
from uyoloseg.core import build_evaluator, TrainingTask

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default='configs/ddrnet.yaml', type=str, help="train config file path")
    parser.add_argument("--model", type=str, help="ckeckpoint file(.ckpt) path")
    parser.add_argument("--seed", type=int, default=304, help="random seed")
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

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')
    device = torch.device('cpu' if cfg.device.gpus == -1 else 'cuda')

    if args.seed is not None:
        pl.seed_everything(args.seed)

    logger.info("Setting up data...")
    val_dataset = build_dataset(cfg.val_dataset, "val", logger)

    evaluator = build_evaluator(cfg.evaluator, dataset=val_dataset, device=device)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=False,
        num_workers=cfg.device.workers_per_gpu,
        pin_memory=True,
        collate_fn=naive_collate,
        drop_last=False,
    )

    logger.info("Creating model...")
    task = TrainingTask(cfg, evaluator)

    ckpt = torch.load(args.model)
    task.load_state_dict(ckpt["state_dict"])

    if cfg.device.gpus == -1:
        logger.info("Using CPU training")
        accelerator, devices = "cpu", None
    else:
        accelerator, devices = "gpu", cfg.device.gpus

    trainer = pl.Trainer(
        default_root_dir=cfg.save_dir.model,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=cfg.log.interval,
        num_sanity_val_steps=0,
        logger=logger,
        callbacks=[TQDMProgressBar(refresh_rate=0)],  # disable tqdm bar
    )

    logger.info("Starting testing...")
    trainer.test(task, val_dataloader)


if __name__ == '__main__':
    main()