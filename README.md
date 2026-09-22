# PTQ4SNN

Code for **[PTQ4SNN: Membrane-Aware Post-Training Quantization for Spiking Neural Networks](https://arxiv.org/abs/2608.07066)**.

[Paper](https://arxiv.org/abs/2608.07066) · [PDF](https://arxiv.org/pdf/2608.07066)

## Overview

PTQ4SNN quantizes both weights and recurrent membrane states using a small calibration
set. Its channel-wise scale bridge couples membrane and weight scales through a
power-of-two factor, while mixed-precision bit allocation (MPBA) assigns 2/4/8-bit
membrane precision using firing activity and quantization sensitivity.

![PTQ4SNN framework: mixed-bit membrane quantization and Unified Scale Bridge](assets/overview.png)

## Method in the paper

| Backbone | Channel-wise scale bridge | Membrane precision |
| --- | --- | --- |
| Spike-Driven Transformer (SDT) | Yes | Channel-wise MPBA (2/4/8 bit) |
| Convolutional SNNs (SEW-ResNet / VGG) | Yes | Channel-wise MPBA (2/4/8 bit) |

In the paper, **W4/M4** denotes 4-bit weights and an element-count-weighted average
membrane precision of approximately 4 bits. MPBA assigns 2, 4, or 8 bits to individual
membrane channels according to firing activity and quantization sensitivity. The
Unified Scale Bridge is calibrated under the assigned channel-wise bit widths;
protected layers or channels can retain higher precision.

## Code structure

The repository provides calibration and evaluation entry points for SDT and
convolutional SNNs, with configurations for ImageNet, CIFAR10-DVS, and CIFAR-100.

```text
ptq/                 SDT quantization, scale calibration, reconstruction, and CLI
model/, module/      Spike-Driven Transformer backbone
ptq4snn/
  model/             Convolutional backbones and LIF neurons
  quantization/      Weight and membrane quantizers
  solver/            Activity/sensitivity statistics, MPBA, reconstruction, and CLI
  utils/             Data loading and evaluation
exp/
  imagenet/          SDT-8-768, T=4
  cifar10-dvs/        SDT-2-256, T=10
  sew-resnet18/       SEW-ResNet18 on CIFAR-100
tests/               Small CPU checks
```

## Requirements

Use Python 3.10. Install a compatible CUDA-enabled PyTorch/torchvision pair, then:

```bash
pip install -r requirements.txt
# Choose the CuPy package matching your CUDA runtime for the original SDT backend:
pip install cupy-cuda11x  # CUDA 11; use cupy-cuda12x for CUDA 12
```

The CPU checks were validated with PyTorch 2.5.1, torchvision 0.20.1, NumPy 1.26.4,
timm 0.6.12, and SpikingJelly 0.0.0.0.14. Calibration and evaluation use CUDA.

## Data and checkpoints

Run all commands from the repository root and update paths and `gpu` in your chosen
configuration. Example paths use this layout:

```text
data/
  imagenet/
    train/<class>/*.JPEG
    val/<class>/*.JPEG
  cifar10_dvs/          SpikingJelly event/frame data
  cifar-100-python/     Extracted CIFAR-100 data
pretrained/
  sdt_8_768.pth
  sdt_2_256_dvs.pth.tar
  sew_resnet18_cifar100.pth
```

- **ImageNet:** use the standard class-folder training and validation splits.
- **CIFAR10-DVS:** the loader uses SpikingJelly's 10-frame representation and a seeded
  90/10 class-stratified training/test split.
- **CIFAR-100:** extract the dataset under `data/` before running the convolutional example.
- **SDT checkpoints:** use timm-compatible state dictionaries matching the architecture
  in the config. See the [upstream SDT repository](https://github.com/BICLab/Spike-Driven-Transformer)
  for pretrained backbones, and match the checkpoint architecture and preprocessing
  to the configuration.
- **Convolutional checkpoints:** the solver loads serialized `ptq4snn.model` model
  objects, with the dataset, class count, and time steps already configured.
  Use trusted checkpoints in this serialized-model format.

## Calibration and evaluation

```bash
# SDT-8-768 / ImageNet
python -m ptq.main --config exp/imagenet/config.yml

# SDT-2-256 / CIFAR10-DVS
python -m ptq.main --config exp/cifar10-dvs/config.yml

# SEW-ResNet18 / CIFAR-100: bridge + MPBA
python -m ptq4snn.solver.main --config exp/sew-resnet18/config.yml \
  --log_save_dir output/sew-resnet18
```

SDT selects a seeded pool of up to `calibration_samples: 1024` training examples;
membrane calibration and reconstruction consume their configured batch limits.
Validation/test data are evaluated separately.

The convolutional example combines firing activity and sensitivity with weights
`0.8` and `0.2`. Its `target_avg_bits: 4` applies to non-stem membrane states;
the first membrane layer is protected at 16 bits. The checkpoint determines the
convolutional model's time steps.

| Setting | Effect |
| --- | --- |
| SDT: `scale_bridge: unify` | Select per-channel integer shifts, then optimize shared scales |
| Convolutional: `m_qconfig.scale_mode: bridge` | Enable interval-calibrated power-of-two scaling |
| Convolutional: `mix_precise: true` | Allocate channel-wise 2/4/8-bit membrane precision |
| Convolutional: `mix_precise: false` | Use the configured uniform non-stem membrane precision |
| Convolutional: `scale_mode: reuse` / `observer` | Run the corresponding scale ablation |

For SDT, `fake_quant: adaround` selects weight reconstruction.

Outputs include logs, SDT scale details, convolutional evaluation metrics, and quantized
checkpoints. The **fake-quantization** models use floating-point tensors to simulate
low-bit arithmetic; reported bit counts describe theoretical storage.

## Tests

```bash
python -m unittest discover -s tests -v
```

The five CPU checks cover LIF equivalence with quantization disabled, scale bridging
and recurrent-state quantization, shared-scale optimization, the MPBA bit budget,
and calibration image preprocessing.

## Citation

If you find this work useful, please cite:

```bibtex
@article{xie2026ptq4snn,
  title={PTQ4SNN: Membrane-Aware Post-Training Quantization for Spiking Neural Networks},
  author={Xie, Hui and Shi, Tong and Qin, Haotong and Liu, Aishan and Liu, Xiaode and Guo, Jinyang},
  journal={arXiv preprint arXiv:2608.07066},
  year={2026},
  url={https://arxiv.org/abs/2608.07066}
}
```

## Acknowledgements and license

The SDT backbone is based on [Spike-Driven Transformer](https://github.com/BICLab/Spike-Driven-Transformer)
by Man Yao et al. This code uses [SpikingJelly](https://github.com/fangwei123456/spikingjelly)
and [timm](https://github.com/huggingface/pytorch-image-models).

The existing repository's [Apache-2.0 license](LICENSE) is retained, along with the
[upstream SDT license text](licenses/SDT-Apache-2.0.txt).

For questions about this repository, please open a GitHub issue.
