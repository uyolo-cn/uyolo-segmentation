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
import logging

from uyoloseg.utils import registers, UYOLOLightningLogger

logger = logging.getLogger(UYOLOLightningLogger._name)

@registers.evaluator.register
class ComposeEvaluator:
    def __init__(self, evals=[], indexes=[[0, 0, 0]], metric_index=0) -> None:
        self.evals = evals
        self.indexes = indexes
        self.metric_index = metric_index
    
    def update(self, preds, targets):
        if not isinstance(preds, list):
            preds = [preds]
        if not isinstance(targets, list):
            targets = [targets]

        for i, j, k in self.indexes:
            self.evals[i].update(preds[j], targets[k])
    
    def evaluate(self):
        res = [ ]
        msg = ''
        
        for idx in self.indexes:
            re, table = self.evals[idx[0]].evaluate()
            res.append(re)
            msg += '\n' + f'Output {idx[0]} evaluation result:' + '\n' + table
        
        logger.info(msg)
        return res