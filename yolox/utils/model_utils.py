#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from copy import deepcopy

import paddle
import paddle.nn as nn
# from paddle.fluid import profiler as profile
from .profile import profile

__all__ = [
    "fuse_conv_and_bn",
    "fuse_model",
    "get_model_info",
    "replace_module",
]


def get_model_info(model, tsize):

    stride = 64
    img = paddle.zeros((1, 3, stride, stride))
    flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)
    params /= 1e6
    flops /= 1e9
    flops *= tsize[0] * tsize[1] / stride / stride * 2  # Gflops
    info = "Params: {:.2f}M, Gflops: {:.2f}".format(params, flops)
    return info


def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = (
        nn.Conv2D(
            conv._in_channels,
            conv._out_channels,
            kernel_size=conv._kernel_size,
            stride=conv._stride,
            padding=conv._padding,
            groups=conv._groups,
            bias_attr=True,
        )
        # .requires_grad_(False)
        # .to(conv.weight.device)
    )

    # prepare filters
    w_conv = paddle.reshape(conv.weight.clone(), (conv._out_channels, -1))
    w_bn = paddle.diag(bn.weight.divide(paddle.sqrt(bn.eps + bn._variance)))
    fusedconv.weight.set_value(paddle.reshape(paddle.mm(w_bn, w_conv), fusedconv.weight.shape))
    # prepare spatial bias
    b_conv = (
        paddle.zeros([conv.weight.shape[0]])
        if conv.bias is None
        else conv.bias
    )
    b_bn = bn.bias - bn.weight.multiply(bn._mean).divide(
        paddle.sqrt(bn._variance + bn.eps)
    )
    fusedconv.bias.set_value(paddle.reshape(paddle.mm(w_bn, paddle.reshape(b_conv, [-1, 1])), [-1]) + b_bn)
    for p in fusedconv.parameters():
        p.stop_gradient = True

    return fusedconv


def fuse_model(model):
    from yolox.models.network_blocks import BaseConv

    for m in model.sublayers():
        if type(m) is BaseConv and hasattr(m, "bn"):
            m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
            delattr(m, "bn")  # remove batchnorm
            m.forward = m.fuseforward  # update forward
    return model


def replace_module(module, replaced_module_type, new_module_type, replace_func=None):
    """
    Replace given type in module to a new type. mostly used in deploy.

    Args:
        module (nn.Layer): model to apply replace operation.
        replaced_module_type (Type): module type to be replaced.
        new_module_type (Type)
        replace_func (function): python function to describe replace logic. Defalut value None.

    Returns:
        model (nn.Layer): module that already been replaced.
    """

    def default_replace_func(replaced_module_type, new_module_type):
        return new_module_type()

    if replace_func is None:
        replace_func = default_replace_func

    model = module
    if isinstance(module, replaced_module_type):
        model = replace_func(replaced_module_type, new_module_type)
    else:  # recurrsively replace
        for name, child in module.named_children():
            new_child = replace_module(child, replaced_module_type, new_module_type)
            if new_child is not child:  # child is already replaced
                model.add_module(name, new_child)

    return model
