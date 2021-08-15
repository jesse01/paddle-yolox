#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import math
from copy import deepcopy

import paddle
import paddle.nn as nn


def is_parallel(model):
    """check if model is in parallel mode."""
    import apex
    parallel_type = (
        paddle.DataParallel,
        apex.parallel.distributed.DistributedDataParallel,
    )
    return isinstance(model, parallel_type)


class ModelEMA:
    def __init__(self, model, decay=0.9999, updates=0):
        """
        Args:
            model (nn.Layer): model to apply EMA.
            decay (float): ema decay reate.
            updates (int): counter of EMA updates.
        """
        # Create EMA(FP32)
        self.ema = deepcopy(model.module if is_parallel(model) else model)
        self.ema.eval()
        self.updates = updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            p.stop_gradient = True

    def update(self, model):
        # Update EMA parameters
        with paddle.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = (
                model.module.state_dict() if is_parallel(model) else model.state_dict()
            )  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype in [paddle.float16, paddle.float32, paddle.float64]:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()
