# coding=utf-8
# 
# Modifications from original work
# 29-03-2021 (tuero@ualberta.ca) : Convert Tensorflow code to PyTorch
# 
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

# Lint as: python3
"""Implementation of a Fenchel-Young loss using perturbation techniques."""

import torch
import torch.nn as nn

import perturbations


class PerturbedFunc(torch.autograd.Function):
    """Implementation of a Fenchel Young loss."""
    @staticmethod
    def forward(ctx, input_tensor, y_true, perturbed, batched, maximize, *args):
        diff = perturbed(input_tensor, *args) - y_true.type(input_tensor.dtype)
        if not maximize:
            diff = -diff
        # Computes per-example loss for batched inputs.
        if batched:
            loss = torch.sum(torch.reshape(diff, [list(diff.shape)[0], -1]) ** 2, dim=-1)
        else:  # Computes loss for unbatched inputs.
            loss = torch.sum(diff ** 2)
        ctx.save_for_backward(diff)
        ctx.batched = batched
        return loss

    @staticmethod
    def backward(ctx, dy):
        diff,  = ctx.saved_tensors
        batched = ctx.batched
        if batched:  # dy has shape (batch_size,) in this case.
            dy = torch.reshape(dy, [list(dy.shape)[0]] + (diff.dim() - 1) * [1])
        return dy * diff, None, None, None, None


class FenchelYoungLoss(nn.Module):
    def __init__(self,
                 func = None,
                 num_samples = 1000,
                 sigma = 0.01,
                 noise = perturbations._NORMAL,
                 batched = True,
                 maximize = True,
                 device=None):
        """Initializes the Fenchel-Young loss.

        Args:
            func: the function whose argmax is to be differentiated by perturbation.
            num_samples: (int) the number of perturbed inputs.
            sigma: (float) the amount of noise to be considered
            noise: (str) the noise distribution to be used to sample perturbations.
            batched: whether inputs to the func will have a leading batch dimension
            (True) or consist of a single example (False). Defaults to True.
            maximize: (bool) whether to maximize or to minimize the input function.
            device: The device to create tensors on (cpu/gpu). If None given, it will
            default to gpu:0 if available, cpu otherwise.
        """
        super().__init__()
        self._batched = batched
        self._maximize = maximize
        self.func = func
        self.perturbed = perturbations.perturbed(func=func,
                                                num_samples=num_samples,
                                                sigma=sigma,
                                                noise=noise,
                                                batched=batched,
                                                device=device)

    def forward(self, input_tensor, y_true, *args):
        return PerturbedFunc.apply(input_tensor, y_true, self.perturbed, self._batched, self._maximize, *args)

