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

from .compose import ComposeEvaluator
from .segment import SegmentEvaluator

from uyoloseg.utils import registers

def build_evaluator(cfg, **kargs):
    compose_evals = get_evaluator(cfg, **kargs)
    if not isinstance(compose_evals, ComposeEvaluator):
        compose_evals = ComposeEvaluator(
            evals=[compose_evals],
            indexes=[[0, 0, 0]]
        )
    return compose_evals

def get_evaluator(cfg, **kargs):
    eval_cfg = copy.deepcopy(cfg)
    name = eval_cfg.pop("name")
    assert name in registers.evaluator, f"{name} not exists"

    if name != "ComposeEvaluator":
        return registers.evaluator[name](**eval_cfg, **kargs)

    compose_evals = eval_cfg.pop("evals")
    evals = [get_evaluator(e_cfg, **kargs) for e_cfg in compose_evals]
    return registers.evaluator[name](evals, **eval_cfg)