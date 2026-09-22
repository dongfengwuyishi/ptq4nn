import logging
import torch
from .fake_quant import QuantizeBase
from .quantized_module import QNeuron

logger = logging.getLogger("quantization")


def _is_full_precision_neuron(submodule):
    n_bits = getattr(submodule, "n_bits", None)
    if n_bits is None:
        return False
    if isinstance(n_bits, torch.Tensor):
        return bool(torch.all(n_bits >= 32).item())
    try:
        return int(n_bits) >= 32
    except Exception:
        return False


def enable_calibration_woquantization(model, quantizer_type="fake_quant"):
    logger.info("Enable observer and Disable quantize for {}".format(quantizer_type))
    for name, submodule in model.named_modules():
        if isinstance(submodule, QuantizeBase):
            if quantizer_type not in name:
                logger.debug("The except_quantizer is {}".format(name))
                submodule.disable_observer()
                submodule.disable_fake_quant()
                continue
            logger.debug("Enable observer and Disable quant: {}".format(name))
            submodule.enable_observer()
            submodule.disable_fake_quant()
        elif isinstance(submodule, QNeuron):
            if _is_full_precision_neuron(submodule):
                logger.debug("Skip full-precision neuron: {}".format(name))
                submodule.disable_observer()
                submodule.disable_fake_quant()
                continue
            logger.debug("Enable neuron: {}".format(name))
            submodule.enable_observer()
            submodule.disable_fake_quant()


def enable_quantization(model, quantizer_type="fake_quant"):
    logger.info("Disable observer and Enable quantize.")
    for name, submodule in model.named_modules():
        if isinstance(submodule, QuantizeBase):
            if quantizer_type not in name:
                logger.debug("The except_quantizer is {}".format(name))
                submodule.disable_observer()
                submodule.disable_fake_quant()
                continue
            logger.debug("Disable observer and Enable quant: {}".format(name))
            submodule.disable_observer()
            submodule.enable_fake_quant()
        elif isinstance(submodule, QNeuron):
            if _is_full_precision_neuron(submodule):
                logger.debug("Skip full-precision neuron: {}".format(name))
                submodule.disable_observer()
                submodule.disable_fake_quant()
                continue
            logger.debug("Enable neuron: {}".format(name))
            submodule.disable_observer()
            submodule.enable_fake_quant()


def disable_all(model):
    logger.info("Disable observer and disable quantize.")
    for name, submodule in model.named_modules():
        if isinstance(submodule, QuantizeBase):
            logger.debug("Disable observer and disable quant: {}".format(name))
            submodule.disable_observer()
            submodule.disable_fake_quant()
        elif isinstance(submodule, QNeuron):
            logger.debug("Disable neuron: {}".format(name))
            submodule.disable_observer()
            submodule.disable_fake_quant()
