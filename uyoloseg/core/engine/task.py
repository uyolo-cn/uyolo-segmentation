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
import os
import copy
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only

from uyoloseg.models import build_model, build_weight_averager
from uyoloseg.core import build_optimizer

class TrainingTask(LightningModule):
    """
    Pytorch Lightning module of a general training task.
    Including training, evaluating and testing.
    Args:
        cfg: Training configurations
        evaluator: Evaluator for evaluating the model performance.
    """

    def __init__(self, cfg, evaluator=None) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = build_model(cfg.model)
        self.evaluator = evaluator
        self.best_metric = -1000
        self.weight_averager = None
        if "weight_averager" in cfg.model:
            self.weight_averager = build_weight_averager(
                cfg.model.weight_averager, device=self.device
            )
            self.avg_model = copy.deepcopy(self.model)
    
    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        preds, loss, loss_dict = self.model.forward_train(batch)

        if self.global_step % self.cfg.log.interval == 0:
            memory = (
                torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
            )
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            log_msg = "Train|Epoch{}/{}|Iter{}({}/{})| mem:{:.3g}G| lr:{:.2e}| ".format(
                self.current_epoch + 1,
                self.cfg.schedule.total_epochs,
                self.global_step,
                batch_idx + 1,
                self.trainer.num_training_batches,
                memory,
                lr,
            )
            self.scalar_summary("Train_loss/lr", "Train", lr, self.global_step)
            for loss_name in loss_dict:
                log_msg += "{}:{:.6f}| ".format(
                    loss_name, loss_dict[loss_name].mean().item()
                )
                self.scalar_summary(
                    "Train_loss/" + loss_name,
                    "Train",
                    loss_dict[loss_name].mean().item(),
                    self.global_step,
                )
            self.logger.info(log_msg)

        return loss


    def training_epoch_end(self, outputs: List[Any]) -> None:
        self.trainer.save_checkpoint(os.path.join(self.cfg.save_dir, 'weights', "model_last.ckpt"))

    def validation_step(self, batch, batch_idx):
        if self.weight_averager is not None:
            preds, loss, loss_states = self.avg_model.forward_train(batch)
        else:
            preds, loss, loss_states = self.model.forward_train(batch)

        self.evaluator.update(preds, batch['mask'])

        if batch_idx % self.cfg.log.interval == 0:
            memory = (
                torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
            )
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            log_msg = "Val|Epoch{}/{}|Iter{}({}/{})| mem:{:.3g}G| lr:{:.2e}| ".format(
                self.current_epoch + 1,
                self.cfg.schedule.total_epochs,
                self.global_step,
                batch_idx + 1,
                sum(self.trainer.num_val_batches),
                memory,
                lr
            )
            for loss_name in loss_states:
                log_msg += "{}:{:.6f}| ".format(
                    loss_name, loss_states[loss_name].mean().item()
                )
            self.logger.info(log_msg)
        return loss

    def validation_epoch_end(self, validation_step_outputs) -> None:
        
        all_result = self.evaluator.evaluate() # list

        if all_result:

            metric_key = self.evaluator.evals[self.evaluator.metric_index].metric_key

            metric = all_result[self.evaluator.metric_index][metric_key]

            if metric > self.best_metric:
                self.best_metric = metric
                self.trainer.save_checkpoint(
                    os.path.join(self.cfg.save_dir, 'weights', "model_best.ckpt")
                )
                self.save_model_state(
                    os.path.join(self.cfg.save_dir, 'weights', "model_best_avg.pth")
                )
        
        self.evaluator.reset()

        self.logger.log_metrics(all_result, self.current_epoch + 1)

    def configure_optimizers(self):
        """
        Prepare optimizer and learning-rate scheduler
        to use in optimization.
        Returns:
            optimizer
        """
        optimizer_cfg = copy.deepcopy(self.cfg.schedule.optimizer)
        optimizer = build_optimizer(self.model, optimizer_cfg)

        schedule_cfg = copy.deepcopy(self.cfg.schedule.lr_schedule)
        name = schedule_cfg.pop("name")
        build_scheduler = getattr(torch.optim.lr_scheduler, name)
        scheduler = {
            "scheduler": build_scheduler(optimizer=optimizer, **schedule_cfg),
            "interval": "epoch",
            "frequency": 1,
        }
        return dict(optimizer=optimizer, lr_scheduler=scheduler)


    def optimizer_step(
        self,
        epoch=None,
        batch_idx=None,
        optimizer=None,
        optimizer_idx=None,
        optimizer_closure=None,
        on_tpu=None,
        using_lbfgs=None,
    ):
        """
        Performs a single optimization step (parameter update).
        Args:
            epoch: Current epoch
            batch_idx: Index of current batch
            optimizer: A PyTorch optimizer
            optimizer_idx: If you used multiple optimizers this indexes into that list.
            optimizer_closure: closure for all optimizers
            on_tpu: true if TPU backward is required
            using_lbfgs: True if the matching optimizer is lbfgs
        """
        # warm up lr
        if self.trainer.global_step <= self.cfg.schedule.warmup.steps:
            if self.cfg.schedule.warmup.name == "constant":
                k = self.cfg.schedule.warmup.ratio
            elif self.cfg.schedule.warmup.name == "linear":
                k = 1 - (
                    1 - self.trainer.global_step / self.cfg.schedule.warmup.steps
                ) * (1 - self.cfg.schedule.warmup.ratio)
            elif self.cfg.schedule.warmup.name == "exp":
                k = self.cfg.schedule.warmup.ratio ** (
                    1 - self.trainer.global_step / self.cfg.schedule.warmup.steps
                )
            else:
                raise Exception("Unsupported warm up type!")
            for pg in optimizer.param_groups:
                pg["lr"] = pg["initial_lr"] * k

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def scalar_summary(self, tag, phase, value, step):
        """
        Write Tensorboard scalar summary log.
        Args:
            tag: Name for the tag
            phase: 'Train' or 'Val'
            value: Value to record
            step: Step value to record
        """
        if self.local_rank < 1:
            self.logger.experiment.add_scalars(tag, {phase: value}, step)

    @rank_zero_only
    def save_model_state(self, path):
        self.logger.info("Saving model to {}".format(path))
        state_dict = (
            self.weight_averager.state_dict()
            if self.weight_averager
            else self.model.state_dict()
        )
        torch.save({"state_dict": state_dict}, path)


    def on_fit_start(self) -> None:
        if "weight_averager" in self.cfg.model:
            self.logger.info("Weight Averaging is enabled")
            self.weight_averager.load_from(self.model)

    # step weight_averager
    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        if self.weight_averager:
            self.weight_averager.update(self.model, self.global_step)

    def on_validation_epoch_start(self):
        if self.weight_averager:
            self.weight_averager.apply_to(self.avg_model)

    def on_load_checkpoint(self, checkpointed_state: Dict[str, Any]) -> None:
        def convert_avg_params(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
            """Converts average state dict to the format that can be loaded to a model.
            Args:
                checkpoint: model.
            Returns:
                Converted average state dict.
            """
            state_dict = checkpoint["state_dict"]
            avg_weights = {}
            for k, v in state_dict.items():
                if "avg_model" in k:
                    avg_weights[k[10:]] = v
            return avg_weights
        if self.weight_averager:
            avg_params = convert_avg_params(checkpointed_state)
            if len(avg_params) != len(self.model.state_dict()):
                self.logger.info(
                    "Weight averaging is enabled but average state does not"
                    "match the model"
                )
            else:
                self.weight_averager.load_state_dict(avg_params)
                self.logger.info("Loaded average state from checkpoint.")
  