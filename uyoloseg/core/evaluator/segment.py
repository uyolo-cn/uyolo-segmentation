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
from torch import Tensor
from typing import Tuple, Union
from tabulate import tabulate


from uyoloseg.utils import registers

class SegmentMetrics:
    def __init__(self, num_classes: int, ignore_label: int, device) -> None:
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.hist = torch.zeros(num_classes, num_classes).to(device)

    def update(self, pred: Tensor, target: Tensor) -> None:
        pred = pred.argmax(dim=1, keepdim=True)
        keep = target != self.ignore_label
        self.hist += torch.bincount(target[keep] * self.num_classes + pred[keep], minlength=self.num_classes**2).view(self.num_classes, self.num_classes)

    def compute_iou(self) -> Tuple[Tensor, Tensor]:
        ious = self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diag())
        miou = ious[~ious.isnan()].mean().item()
        ious *= 100
        miou *= 100
        return ious, round(miou, 2)

    def compute_f1(self) -> Tuple[Tensor, Tensor]:
        f1 = 2 * self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1))
        mf1 = f1[~f1.isnan()].mean().item()
        f1 *= 100
        mf1 *= 100
        return f1, round(mf1, 2)

    def compute_pixel_acc(self) -> Tuple[Tensor, Tensor]:
        acc = self.hist.diag() / self.hist.sum(1)
        macc = acc[~acc.isnan()].mean().item()
        acc *= 100
        macc *= 100
        return acc, round(macc, 2)
    
    def reset(self):
        self.hist.fill_(0)

@registers.evaluator.register
class SegmentEvaluator:
    def __init__(self, metric_key='miou', dataset=None, device=None) -> None:
        self.metric = SegmentMetrics(dataset.num_classes, dataset.ignore_index, device)
        self.metric_key = metric_key
   
    def update(self, pred, target):
        self.metric.update(pred, target)
   
    def evaluate(self):
        eval_results = {}
       
        ious, eval_results['miou'] = self.metric.compute_iou()
        accs, eval_results['macc'] = self.metric.compute_pixel_acc()
        f1s, eval_results['mf1'] = self.metric.compute_f1()

        table_header = [f'class_{i}' for i in range(ious.shape[0])] + ['mean']
        metric_dict = ['iou', 'acc', 'f1']
        row_data = [
            ious.tolist() + [eval_results['miou']],
            accs.tolist() + [eval_results['macc']],
            f1s.tolist() + [eval_results['mf1']],
        ]
        
        table = tabulate(
            row_data,
            headers=table_header,
            tablefmt='pipe',
            floatfmt=".2f",
            numalign="center",
            showindex=metric_dict
        )

        self.metric.reset()

        return eval_results, table