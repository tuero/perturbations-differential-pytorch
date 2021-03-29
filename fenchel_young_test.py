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
"""Tests for the fenchel_young module."""

import torch
from absl.testing import absltest
import numpy as np
import numpy.testing as npt

import fenchel_young as fy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def ranks(inputs, axis=-1):
    """Returns the ranks of the input values among the given axis."""
    return 1 + torch.argsort(torch.argsort(inputs, axis=axis), axis=axis).type(inputs.dtype)


class FenchelYoungTest(absltest.TestCase):
    """Testing the gradients obtained by the FenchelYoungLoss class."""

    def test_gradients(self):
        loss_fn = fy.FenchelYoungLoss(ranks, num_samples=10000, sigma=0.1, batched=False)

        theta1 = torch.tensor([1, 20, 7.3, 7.35], requires_grad=True, device=device)
        theta2 = torch.tensor([1, 20, 7.3, 7.35], requires_grad=True, device=device)
        theta3 = torch.tensor([1, 20, 7.3, 7.35], requires_grad=True, device=device)
        y_true = torch.tensor([1, 4, 3, 2], dtype=theta1.dtype, device=device)
        y_hard_minimum = torch.tensor([1, 4, 2, 3], dtype=theta2.dtype, device=device)
        y_perturbed_minimum = loss_fn.perturbed(theta3)

        # Compute losses
        output_true = loss_fn(theta1, y_true)
        output_hard_minimum = loss_fn(theta2, y_hard_minimum)
        output_perturbed_minimum = loss_fn(theta3, y_perturbed_minimum)

        # Compute gradients
        output_true.backward(torch.ones_like(output_true))
        output_hard_minimum.backward(torch.ones_like(output_hard_minimum))
        output_perturbed_minimum.backward(torch.ones_like(output_perturbed_minimum))
        g_true = theta1.grad
        g_hard_minimum = theta2.grad
        g_perturbed_minimum = theta3.grad

        # The gradient should be close to zero for the two first values.
        npt.assert_almost_equal(g_true[:2].cpu().numpy(), np.array([0.0, 0.0]))
        self.assertLess(torch.norm(g_perturbed_minimum), torch.norm(g_hard_minimum))
        self.assertLess(torch.norm(g_hard_minimum), torch.norm(g_true))
        for g in [g_true, g_hard_minimum, g_perturbed_minimum]:
            self.assertAlmostEqual(torch.sum(g).item(), 0.0, 5)


if __name__ == '__main__':
    absltest.main()
