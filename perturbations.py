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
"""Introduces differentiation via perturbations.

Example of usage:

  @perturbed
  def sign_or(x, axis=-1):
    s = ((torch.sign(x) + 1) / 2.0).type(torch.bool)
    result = torch.any(s, dim=-1)
    return result.type(torch.float) * 2.0 - 1


Then sign_or is differentiable (unlike what it seems).

It is possible to specify the parameters of the perturbations using:
  @perturbed(num_samples=1000, sigma=0.1, noise='gumbel')
  ...

The decorator can also be used directly as a function, for example:
  soft_argsort = perturbed(torch.argsort, num_samples=200, sigma=0.01)
"""

import functools
from typing import Tuple
import torch
from torch.distributions.gumbel import Gumbel
from torch.distributions.normal import Normal

_GUMBEL = 'gumbel'
_NORMAL = 'normal'
SUPPORTED_NOISES = (_GUMBEL, _NORMAL)


def sample_noise_with_gradients(noise, shape):
    """Samples a noise tensor according to a distribution with its gradient.

    Args:
    noise: (str) a type of supported noise distribution.
    shape: torch.tensor<int>, the shape of the tensor to sample.

    Returns:
    A tuple Tensor<float>[shape], Tensor<float>[shape] that corresponds to the
    sampled noise and the gradient of log the underlying probability
    distribution function. For instance, for a gaussian noise (normal), the
    gradient is equal to the noise itself.

    Raises:
    ValueError in case the requested noise distribution is not supported.
    See perturbations.SUPPORTED_NOISES for the list of supported distributions.
    """
    if noise not in SUPPORTED_NOISES:
        raise ValueError('{} noise is not supported. Use one of [{}]'.format(
            noise, SUPPORTED_NOISES))

    if noise == _GUMBEL:
        sampler = Gumbel(0.0, 1.0)
        samples = sampler.sample(shape)
        gradients = 1 - torch.exp(-samples)
    elif noise == _NORMAL:
        sampler = Normal(0.0, 1.0)
        samples = sampler.sample(shape)
        gradients = samples

    return samples, gradients


def perturbed(func=None,
              num_samples = 1000,
              sigma = 0.05,
              noise = _NORMAL,
              batched = True,
              device=None):
    """Turns a function into a differentiable one via perturbations.

    The input function has to be the solution to a linear program for the trick
    to work. For instance the maximum function, the logical operators or the ranks
    can be expressed as solutions to some linear programs on some polytopes.
    If this condition is violated though, the result would not hold and there is
    no guarantee on the validity of the obtained gradients.

    This function can be used directly or as a decorator.

    Args:
    func: the function to be turned into a perturbed and differentiable one.
    Four I/O signatures for func are currently supported:
        If batched is True,
        (1) input [B, D1, ..., Dk], output [B, D1, ..., Dk], k >= 1
        (2) input [B, D1, ..., Dk], output [B], k >= 1
        If batched is False,
        (3) input [D1, ..., Dk], output [D1, ..., Dk], k >= 1
        (4) input [D1, ..., Dk], output [], k >= 1.
    num_samples: the number of samples to use for the expectation computation.
    sigma: the scale of the perturbation.
    noise: a string representing the noise distribution to be used to sample
    perturbations.
    batched: whether inputs to the perturbed function will have a leading batch
    dimension (True) or consist of a single example (False). Defaults to True.
    device: The device to create tensors on (cpu/gpu). If None given, it will
    default to gpu:0 if available, cpu otherwise.

    Returns:
    a function has the same signature as func but that can be back propagated.
    """
    # If device not supplied, auto detect
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # This is a trick to have the decorator work both with and without arguments.
    if func is None:
        return functools.partial(
            perturbed, num_samples=num_samples, sigma=sigma, noise=noise,
            batched=batched, device=device)

    @functools.wraps(func)
    def wrapper(input_tensor, *args):
        class PerturbedFunc(torch.autograd.Function):

            @staticmethod
            def forward(ctx, input_tensor, *args):
                original_input_shape = input_tensor.shape
                if batched:
                    if not input_tensor.dim() >= 2:
                        raise ValueError('Batched inputs must have at least rank two')
                else:  # Adds dummy batch dimension internally.
                    input_tensor = input_tensor.unsqueeze(0)
                input_shape = input_tensor.shape  # [B, D1, ... Dk], k >= 1
                perturbed_input_shape = [num_samples] + list(input_shape)

                noises = sample_noise_with_gradients(noise, perturbed_input_shape)
                additive_noise, noise_gradient = tuple(
                    [noise.type(input_tensor.dtype) for noise in noises])
                additive_noise = additive_noise.to(device)
                noise_gradient = noise_gradient.to(device)
                perturbed_input = input_tensor.unsqueeze(0) + sigma * additive_noise

                # [N, B, D1, ..., Dk] -> [NB, D1, ..., Dk].
                flat_batch_dim_shape = [-1] + list(input_shape)[1:]
                perturbed_input = torch.reshape(perturbed_input, flat_batch_dim_shape)
                # Calls user-defined function in a perturbation agnostic manner.
                perturbed_output = func(perturbed_input, *args)
                # [NB, D1, ..., Dk] ->  [N, B, D1, ..., Dk].
                perturbed_input = torch.reshape(perturbed_input, perturbed_input_shape)
                # Either
                #   (Default case): [NB, D1, ..., Dk] ->  [N, B, D1, ..., Dk]
                # or
                #   (Full-reduce case) [NB] -> [N, B]
                perturbed_output_shape = [num_samples, -1] + list(perturbed_output.shape)[1:]
                perturbed_output = torch.reshape(perturbed_output, perturbed_output_shape)

                forward_output = torch.mean(perturbed_output, dim=0)
                if not batched:  # Removes dummy batch dimension.
                    forward_output = forward_output[0]

                # Save context for backward pass
                ctx.save_for_backward(perturbed_input, perturbed_output, noise_gradient)
                ctx.original_input_shape = original_input_shape

                return forward_output

            @staticmethod
            def backward(ctx, dy):
                # Pull saved tensors
                original_input_shape = ctx.original_input_shape
                perturbed_input, perturbed_output, noise_gradient = ctx.saved_tensors
                output, noise_grad = perturbed_output, noise_gradient
                # Adds dummy feature/channel dimension internally.
                if perturbed_input.dim() > output.dim():
                    dy = dy.unsqueeze(-1)
                    output = output.unsqueeze(-1)
                # Adds dummy batch dimension internally.
                if not batched:
                    dy = dy.unsqueeze(0)
                # Flattens [D1, ..., Dk] to a single feat dim [D].
                flatten = lambda t: torch.reshape(t, (list(t.shape)[0], list(t.shape)[1], -1))
                dy = torch.reshape(dy, (list(dy.shape)[0], -1))  # (B, D)
                output = flatten(output)  # (N, B, D)
                noise_grad = flatten(noise_grad)  # (N, B, D)

                g = torch.einsum('nbd,nb->bd', noise_grad, torch.einsum('nbd,bd->nb', output, dy))
                g /= sigma * num_samples
                return torch.reshape(g, original_input_shape)

        return PerturbedFunc.apply(input_tensor, *args)

    return wrapper
