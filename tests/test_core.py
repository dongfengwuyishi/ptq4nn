"""Small CPU checks; no datasets or pretrained checkpoints required."""
import copy
import logging
import unittest

import numpy as np
import torch
from easydict import EasyDict
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

from ptq.observer import lsq_unified_scale_optim
from ptq.main import CalibrationSubset
from ptq.quantized_module import QuantMultiStepLIFNode
from ptq4snn.model.common import LIFNeuron
from ptq4snn.solver.bit_allocation import allocate_bits_per_channel


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_calibration_subset_receives_image_transforms(self):
        from PIL import Image
        from timm.data import create_loader

        class Images(torch.utils.data.Dataset):
            transform = None

            def __len__(self):
                return 4

            def __getitem__(self, index):
                image = Image.new('RGB', (16, 16))
                return self.transform(image) if self.transform else image, index

        subset = CalibrationSubset(Images(), [1, 3])
        loader = create_loader(subset, input_size=(3, 8, 8), batch_size=2,
                               is_training=False, use_prefetcher=False,
                               num_workers=0, persistent_workers=False)
        images, indices = next(iter(loader))
        self.assertEqual(images.shape, (2, 3, 8, 8))
        self.assertEqual(indices.tolist(), [1, 3])

    def test_disabled_membrane_quantization_matches_lif(self):
        original = MultiStepLIFNode(tau=2.0, backend='torch')
        quantized = QuantMultiStepLIFNode(copy.deepcopy(original), mem_bit=4)
        quantized.mem_fake_quant.disable_observer()
        quantized.mem_fake_quant.disable_fake_quant()
        x = torch.randn(4, 2, 3, 4, 4)
        torch.testing.assert_close(quantized(x), original(x))
        torch.testing.assert_close(quantized.original_lif.v, original.v)

    def test_bridge_and_recurrent_state_grid(self):
        node = QuantMultiStepLIFNode(MultiStepLIFNode(backend='torch'), mem_bit=4)
        node._pot_k = torch.tensor([-1., 0., 2.])
        weight_scale = torch.tensor([0.1, 0.2, 0.05])
        node.enable_fake_quant()
        output = node((torch.randn(4, 2, 3, 2, 2), weight_scale))
        expected_scale = torch.tensor([0.05, 0.2, 0.2])
        torch.testing.assert_close(node.mem_fake_quant.scale, expected_scale)
        codes = node.original_lif.v_seq / expected_scale.view(1, 1, 3, 1, 1)
        torch.testing.assert_close(codes, codes.round())
        self.assertGreaterEqual(codes.min().item(), -8)
        self.assertLessEqual(codes.max().item(), 7)
        self.assertTrue(torch.all((output == 0) | (output == 1)))

    def test_shared_scale_optimization(self):
        weights = torch.randn(3, 32) * 0.2
        membrane = torch.randn(3, 48) * 0.4
        scale = lsq_unified_scale_optim(
            weights, membrane, torch.full((3,), 0.05),
            torch.tensor([0., 1., 2.]), 4, 4, num_iters=3,
        )
        self.assertEqual(scale.shape, (3,))
        self.assertTrue(torch.all(torch.isfinite(scale) & (scale > 0)))

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
