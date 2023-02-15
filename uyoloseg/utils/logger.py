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
import logging
import os
import time

import lightning_lite
from lightning_lite.utilities.cloud_io import get_filesystem
from pytorch_lightning.loggers import Logger as LightningLoggerBase
from pytorch_lightning.loggers.logger import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only
from termcolor import colored

class UYOLOLightningLogger(LightningLoggerBase):
    def __init__(self, save_dir="./", **kwargs):
        super().__init__()
        self._name = "uyolo"
        self._version = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self._save_dir = os.path.join(save_dir, f"logs-{self._version}")

        self._fs = get_filesystem(save_dir)
        self._fs.makedirs(self._save_dir, exist_ok=True)
        self._init_logger()

        self._experiment = None
        self._kwargs = kwargs

    @property
    def name(self):
        return self._name

    @property
    @rank_zero_experiment
    def experiment(self):
        r"""
        Actual tensorboard object. To use TensorBoard features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.
        Example::
            self.logger.experiment.some_tensorboard_function()
        """
        if self._experiment is not None:
            return self._experiment

        assert rank_zero_only.rank == 0, "tried to init log dirs in non global_rank=0"

        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            raise ImportError(
                'Please run "pip install future tensorboard" to install '
                "the dependencies to use torch.utils.tensorboard "
                "(applicable to PyTorch 1.1 or higher)"
            ) from None

        self._experiment = SummaryWriter(log_dir=self._save_dir, **self._kwargs)
        return self._experiment

    @property
    def version(self):
        return self._version

    def _init_logger(self):
        self.logger = logging.getLogger(name=self.name)
        self.logger.setLevel(logging.INFO)

        # create file handler
        fh = logging.FileHandler(os.path.join(self._save_dir, "logs.txt"))
        fh.setLevel(logging.INFO)
        # set file formatter
        f_fmt = "[%(name)s][%(asctime)s]%(levelname)s: %(message)s"
        file_formatter = logging.Formatter(f_fmt, datefmt="%m-%d %H:%M:%S")
        fh.setFormatter(file_formatter)

        # create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # set console formatter
        c_fmt = (
            colored("[%(name)s]", "magenta", attrs=["bold"])
            + colored("[%(asctime)s]", "blue")
            + colored("%(levelname)s:", "green")
            + colored("%(message)s", "white")
        )
        console_formatter = logging.Formatter(c_fmt, datefmt="%m-%d %H:%M:%S")
        ch.setFormatter(console_formatter)

        # add the handlers to the logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    @rank_zero_only
    def info(self, string):
        self.logger.info(string)

    @rank_zero_only
    def log(self, string):
        self.logger.info(string)

    @rank_zero_only
    def dump_cfg(self, cfg_node):
        with open(os.path.join(self._save_dir, "train_cfg.yml"), "w") as f:
            cfg_node.dump(stream=f)

    @rank_zero_only
    def log_hyperparams(self, params):
        self.logger.info(f"hyperparams: {params}")

    @rank_zero_only
    def log_metrics(self, metrics, step):
        self.logger.info(f"Val_metrics: {metrics}")
        for k, v in metrics.items():
            self.experiment.add_scalars("Val_metrics/" + k, {"Val": v}, step)

    @rank_zero_only
    def save(self):
        super().save()

    @rank_zero_only
    def finalize(self, status):
        self.experiment.flush()
        self.experiment.close()
        self.save()

def set_same_logger(logger):
    lightning_lite.utilities.seed.log = logger
    lightning_lite.utilities.rank_zero.rank_zero_module.log = logger