#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# This file mainly comes from
# https://github.com/facebookresearch/detectron2/blob/master/detectron2/utils/comm.py
# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""

import functools
import logging
import pickle
import time

import numpy as np

import paddle
from paddle import distributed as dist

__all__ = [
    "is_main_process",
    "synchronize",
    "get_world_size",
    "get_rank",
    "get_local_rank",
    "get_local_size",
    "time_synchronized",
    "gather",
    "all_gather",
]

_LOCAL_PROCESS_GROUP = None


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when using distributed training
    """
    # if not dist.is_available():
    #     return
    # if not dist.is_initialized():
    #     return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def get_world_size() -> int:
    # if not dist.is_available():
    #     return 1
    # if not dist.is_initialized():
    #     return 1
    return dist.get_world_size()


def get_rank() -> int:
    # if not dist.is_available():
    #     return 0
    # if not dist.is_initialized():
    #     return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    # if not dist.is_available():
    #     return 0
    # if not dist.is_initialized():
    #     return 0
    assert _LOCAL_PROCESS_GROUP is not None
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)


def get_local_size() -> int:
    """
    Returns:
        The size of the per-machine process group, i.e. the number of processes per machine.
    """
    # if not dist.is_available():
    #     return 1
    # if not dist.is_initialized():
    #     return 1
    return dist.get_world_size(group=_LOCAL_PROCESS_GROUP)


def is_main_process() -> bool:
    return get_rank() == 0


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def _serialize_to_tensor(data, group):
    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    device = "cpu" if backend == "gloo" else "cuda"

    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        logger = logging.getLogger(__name__)
        logger.warning(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                get_rank(), len(buffer) / (1024 ** 3), device
            )
        )
    tensor = paddle.to_tensor(list(buffer), paddle.int8)
    return tensor


def _pad_to_largest_tensor(tensor, group):
    """
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    world_size = dist.get_world_size(group=group)
    assert (
        world_size >= 1
    ), "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = paddle.to_tensor([tensor.numel()], dtype=paddle.int64, device=tensor.device)
    size_list = [
        paddle.zeros([1], dtype=paddle.int64, device=tensor.device)
        for _ in range(world_size)
    ]
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # we pad the tensor because paddle all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = paddle.zeros(
            (max_size - local_size,), dtype=paddle.uint8, device=tensor.device
        )
        tensor = paddle.concat((tensor, padding), axis=0)
    return size_list, tensor


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        group: a paddle process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return [data]

    tensor = _serialize_to_tensor(data, group)

    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving Tensor from all ranks
    tensor_list = [
        paddle.empty((max_size,), dtype=paddle.uint8)
        for _ in size_list
    ]
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def gather(data, dst=0, group=None):
    """
    Run gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        dst (int): destination rank
        group: a paddle process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: on dst, a list of data gathered from each rank. Otherwise,
            an empty list.
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group=group) == 1:
        return [data]
    rank = dist.get_rank(group=group)

    tensor = _serialize_to_tensor(data, group)
    size_list, tensor = _pad_to_largest_tensor(tensor, group)

    # receiving Tensor from all ranks
    if rank == dst:
        max_size = max(size_list)
        tensor_list = [
            paddle.empty((max_size,), dtype=paddle.uint8, device=tensor.device)
            for _ in size_list
        ]
        dist.gather(tensor, tensor_list, dst=dst, group=group)

        data_list = []
        for size, tensor in zip(size_list, tensor_list):
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))
        return data_list
    else:
        dist.gather(tensor, [], dst=dst, group=group)
        return []


def shared_random_seed():
    """
    Returns:
        int: a random number that is the same across all workers.
            If workers need a shared RNG, they can use this shared seed to
            create one.
    All workers must call this function, otherwise it will deadlock.
    """
    ints = np.random.randint(2 ** 31)
    all_ints = all_gather(ints)
    return all_ints[0]


def time_synchronized():
    # if paddle.fluid.is_compiled_with_cuda():
    #     paddle.device.cuda.synchronize()
    return time.time()