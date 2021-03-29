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
"""Tests for differentiable_programming.perturbations."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import numpy.testing as npt
import torch
import torch.nn.functional as F

import perturbations

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def reduce_sign_any(input_tensor, axis=-1):
    """A logical or of the signs of a tensor along an axis.

    Args:
    input_tensor: Tensor<float> of any shape.
    axis: the axis along which we want to compute a logical or of the signs of
        the values.

    Returns:
    A Tensor<float>, which as the same shape as the input tensor, but without the
        axis on which we reduced.
    """
    boolean_sign = torch.any(
        ((torch.sign(input_tensor) + 1) / 2.0).type(torch.bool), dim=axis)
    return boolean_sign.type(input_tensor.dtype) * 2.0 - 1.0


class PerturbationsTest(parameterized.TestCase):
    """Testing the perturbations module."""

    def setUp(self):
        super(PerturbationsTest, self).setUp()
        torch.manual_seed(0)

    @parameterized.parameters([perturbations._GUMBEL, perturbations._NORMAL])
    def test_sample_noise_with_gradients(self, noise):
        shape = (3, 2, 4)
        samples, gradients = perturbations.sample_noise_with_gradients(noise, shape)
        self.assertEqual(tuple(samples.shape), shape)
        self.assertEqual(tuple(gradients.shape), shape)

    def test_sample_noise_with_gradients_raise(self):
        with self.assertRaises(ValueError):
            _, _ = perturbations.sample_noise_with_gradients('unknown', (3, 2, 4))

    @parameterized.parameters([1e-3, 1e-2, 1e-1])
    def test_perturbed_reduce_sign_any(self, sigma):
        input_tensor = torch.tensor([[-0.3, -1.2, 1.6], [-0.4, -2.4, -1.0]], device=device)
        soft_reduce_any = perturbations.perturbed(reduce_sign_any, sigma=sigma)
        output_tensor = soft_reduce_any(input_tensor, -1)
        npt.assert_almost_equal(output_tensor.cpu().numpy(), np.array([1.0, -1.0]), decimal=2)

    def test_perturbed_reduce_sign_any_gradients(self):
        # We choose a point where the gradient should be above noise, that is
        # to say the distance to 0 along one direction is about sigma.
        sigma = 0.1
        input_tensor = torch.tensor([[-0.6, -1.2, 0.5 * sigma], [-2 * sigma, -2.4, -1.0]], requires_grad=True, device=device)
        soft_reduce_any = perturbations.perturbed(reduce_sign_any, sigma=sigma)

        output_tensor = soft_reduce_any(input_tensor)
        output_tensor.backward(torch.ones_like(output_tensor).to(device))
        gradient = input_tensor.grad

        # The two values that could change the soft logical or should be the one
        # with real positive impact on the final values.
        self.assertGreater(gradient[0, 2], 0.0)
        self.assertGreater(gradient[1, 0], 0.0)
        # The value that is more on the fence should bring more gradient than any
        # other one.
        self.assertTrue((gradient.cpu().numpy() <= gradient[0, 2].cpu().numpy()).all())

    def test_unbatched_rank_one_raise(self):
        with self.assertRaises(ValueError):
            input_tensor = torch.tensor([-0.6, -0.5, 0.5], device=device)
            dim = len(input_tensor)
            n = 10000000

            argmax = lambda t: F.one_hot(torch.argmax(t, 1), dim)
            soft_argmax = perturbations.perturbed(argmax, sigma=0.5, num_samples=n)
            _ = soft_argmax(input_tensor)

    def test_perturbed_argmax_gradients_without_minibatch(self):
        input_tensor = torch.tensor([-0.6, -0.5, 0.5], requires_grad=True, device=device)
        dim = input_tensor.shape[-1]
        eps = 1e-2
        n = 10000000

        argmax = lambda t: F.one_hot(torch.argmax(t, 1), dim).float()
        soft_argmax = perturbations.perturbed(
            argmax, sigma=0.5, num_samples=n, batched=False)
        norm_argmax = lambda t: torch.sum(torch.square(soft_argmax(t)))

        w = torch.randn(input_tensor.shape).to(device)
        w /= torch.linalg.norm(w)

        value = norm_argmax(input_tensor)
        value.backward(torch.ones_like(value))
        grad = torch.reshape(input_tensor.grad, list(input_tensor.shape))

        value_minus = norm_argmax(input_tensor - eps * w)
        value_plus = norm_argmax(input_tensor + eps * w)

        lhs = torch.sum(w * grad)
        rhs = (value_plus - value_minus) * 1./(2*eps)
        self.assertLess(torch.abs(lhs - rhs), 0.05)

    def test_perturbed_argmax_gradients_with_minibatch(self):
        input_tensor = torch.tensor([[-0.6, -0.7, 0.5], [0.9, -0.6, -0.5]], requires_grad=True, device=device)
        dim = input_tensor.shape[-1]
        eps = 1e-2
        n = 10000000

        argmax = lambda t: F.one_hot(torch.argmax(t, -1), dim).float()
        soft_argmax = perturbations.perturbed(argmax, sigma=2.5, num_samples=n)
        norm_argmax = lambda t: torch.sum(torch.square(soft_argmax(t)))

        w = torch.randn(input_tensor.shape).to(device)
        w /= torch.linalg.norm(w)
        
        value = norm_argmax(input_tensor)
        value.backward(torch.ones_like(value))
        grad = torch.reshape(input_tensor.grad, list(input_tensor.shape))

        value_minus = norm_argmax(input_tensor - eps * w)
        value_plus = norm_argmax(input_tensor + eps * w)

        lhs = torch.sum(w * grad)
        rhs = (value_plus - value_minus) * 1./(2*eps)
        self.assertLess(torch.abs(lhs - rhs), 0.05)

if __name__ == '__main__':
    absltest.main()
