# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import functools

import paddle
import paddle.distributed as dist


def get_dist_info():
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    return rank, world_size


def main_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper


def gather_from_gpu(gather_object: paddle.Tensor,
                    concat_axis=0) -> paddle.Tensor:
    """gather Tensor from all gpus into a list and concatenate them on `concat_axis`.

    Args:
        gather_object (paddle.Tensor): gather object Tensor
        concat_axis (int, optional): axis for concatenation. Defaults to 0.

    Returns:
        paddle.Tensor: gatherd & concatenated Tensor
    """
    gather_object_list = []
    dist.all_gather(gather_object_list, gather_object)
    return paddle.concat(gather_object_list, axis=concat_axis)
