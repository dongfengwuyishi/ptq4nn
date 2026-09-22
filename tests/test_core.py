"""Small CPU checks; no datasets or pretrained checkpoints required."""
import logging
import unittest

import numpy as np
import torch
from easydict import EasyDict
from ptq4snn.quantization.quantized_module import QNeuron
from ptq4snn.model.common import LIFNeuron
from ptq4snn.solver.bit_allocation import allocate_bits_per_channel


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    @staticmethod
    def neuron_config():
        return EasyDict(bit=4, learnable_factor=False, ch_axis=0,
                        factor_init_by_observe=True, limit_mem_range=True,
                        idea1=True, scale_mode='bridge')

    def test_disabled_membrane_quantization_matches_lif(self):
        original = LIFNeuron(tau=2.0, step_mode='m')
        quantized = QNeuron(tau=2.0, step_mode='m', qconfig=self.neuron_config())
        x = torch.randn(4, 2, 3, 4, 4)
        torch.testing.assert_close(quantized((x, torch.ones(3))), original(x))
        torch.testing.assert_close(quantized.mem, original.mem)

    def test_bridge_with_channelwise_bits(self):
        node = QNeuron(qconfig=self.neuron_config())
        node.set_bit([2, 4, 8])
        node.set_factor_init_mask(True)
        weight_scale = torch.tensor([0.1, 0.2, 0.05])
        node.enable_observer()
        node((torch.zeros(2, 3, 2, 2), weight_scale))
        node.disable_observer()
        node.reset()
        node.enable_fake_quant()
        shifts = node.factor_exp.detach()
        torch.testing.assert_close(shifts, shifts.round())
        self.assertGreater(shifts.unique().numel(), 1)
        for x in torch.randn(4, 2, 3, 2, 2):
            output = node((x, weight_scale))
            codes = node.mem / weight_scale.view(1, 3, 1, 1)
            torch.testing.assert_close(codes, codes.round(), atol=1e-5, rtol=1e-5)
            self.assertTrue(torch.all(codes >= node.min_val.view(1, 3, 1, 1) - 1e-5))
            self.assertTrue(torch.all(codes <= node.max_val.view(1, 3, 1, 1) + 1e-5))
            self.assertTrue(torch.all((output == 0) | (output == 1)))

    def test_mpba_element_weighted_budget(self):
        class ToySNN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gain = torch.nn.Parameter(torch.ones(()))
                self.large = LIFNeuron()
                self.small = LIFNeuron()

            def forward(self, x):
                a = self.large(x * self.gain)
                b = self.small(x[:, :, ::2, ::2] * self.gain)
                return a.mean((2, 3)) + b.mean((2, 3))

        model = ToySNN()
        scores = torch.linspace(0, 1, 128)
        activity = {name: {'rate': scores, 'mem': torch.ones(128)}
                    for name in ('large', 'small')}
        cfg = EasyDict(activity_mem_weight=0, activity_rate_weight=0.8,
                       loss_salience_weight=0.2, target_avg_bits=4,
                       activity_percentile_low=50, activity_percentile_high=96)
        allocation = allocate_bits_per_channel(
            activity, logging.getLogger('test'), cfg, model=model,
            cali_data=torch.randn(1, 128, 4, 4),
            salience_map={name: scores for name in activity},
        )
        average = (allocation['large'].sum() * 16 + allocation['small'].sum() * 4) / (128 * 20)
        self.assertAlmostEqual(average, 4.0, delta=0.05)
        for bits in allocation.values():
            self.assertTrue(set(bits).issubset({2, 4, 8}))
            self.assertTrue(np.all(np.diff(bits) >= 0))
            self.assertEqual(bits[-1], 8)
            self.assertEqual(bits[0], 2)


if __name__ == '__main__':
    unittest.main()
