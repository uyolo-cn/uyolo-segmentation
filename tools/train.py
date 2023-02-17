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
import os
import warnings

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import TQDMProgressBar

from uyoloseg.datasets import build_dataset, naive_collate
from uyoloseg.utils import cfg, update_config, UYOLOLightningLogger, set_multi_processing, set_same_logger, registers, import_all_modules_for_register
from uyoloseg.core import build_evaluator, TrainingTask

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default='configs/custom.yaml', type=str, help="train config file path")
    parser.add_argument(
        "--local_rank", default=-1, type=int, help="node rank for distributed training"
    )
    parser.add_argument("--seed", type=int, default=304, help="random seed")
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(cfg, args)
    return args

def main():
    args = parse_args()

    logger = UYOLOLightningLogger(cfg.save_dir)
    import_all_modules_for_register()
    logger.dump_cfg(cfg)

    local_rank = int(args.local_rank)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cpu' if cfg.device.gpus == -1 else 'cuda')

    if args.seed is not None:
        pl.seed_everything(args.seed)

    logger.info("Setting up data...")
    train_dataset = build_dataset(cfg.train_dataset, "train", logger)
    val_dataset = build_dataset(cfg.val_dataset, "val", logger)

    evaluator = build_evaluator(cfg.evaluator, dataset=val_dataset, device=device)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=True,
        num_workers=cfg.device.workers_per_gpu,
        pin_memory=True,
        collate_fn=naive_collate,
        drop_last=True,
    )
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

    model_resume_path = (
        os.path.join(cfg.save_dir, "model_last.ckpt")
        if "resume" in cfg.schedule and cfg.schedule.resume
        else None
    )

    if cfg.device.gpus == -1:
        logger.info("Using CPU training")
        accelerator, devices, strategy, precision = (
            "cpu",
            None,
            None,
            cfg.device.precision,
        )
    else:
        accelerator, devices, strategy, precision = (
            "gpu",
            cfg.device.gpus,
            None,
            cfg.device.precision,
        )

    if devices and len(devices) > 1:
        strategy = "ddp"
        set_multi_processing(distributed=True)

    trainer = pl.Trainer(
        default_root_dir=cfg.save_dir,
        max_epochs=cfg.schedule.total_epochs,
        check_val_every_n_epoch=cfg.schedule.val_intervals,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=cfg.log.interval,
        num_sanity_val_steps=0,
        callbacks=[TQDMProgressBar(refresh_rate=0)],  # disable tqdm bar
        logger=logger,
        benchmark=cfg.get("cudnn_benchmark", True),
        gradient_clip_val=cfg.get("grad_clip", 0.0),
        strategy=strategy,
        precision=precision,
    )

    import pdb; pdb.set_trace()

    trainer.fit(task, train_dataloader, val_dataloader, ckpt_path=model_resume_path)

if __name__ == '__main__':
    main()