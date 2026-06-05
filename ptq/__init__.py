"""
PTQ (Post-Training Quantization) for Spike-Driven Transformer

文件组织结构 (参考 ptq4snn):
├── observer.py        - Observer 类 (统计数据分布)
├── fake_quant.py      - FakeQuantize 类 (伪量化)
├── quantized_module.py - 量化模块 (QuantConv2d, QuantLinear, etc.)
├── util_quant.py      - 量化工具函数
├── state.py           - 状态管理 (enable/disable quantization)
├── quantize.py        - 量化主逻辑 (quantize_model, fold_bn, etc.)
├── main.py            - 命令行入口
└── __init__.py        - 导出

使用示例:
    from ptq import quantize_model, PTQConfig

    config = PTQConfig(
        weight_bit=4, first_layer_bit=16, last_layer_bit=8, first_mem_bit=16,
    )
    quant_model = quantize_model(model, config)
"""

# Observer
from .observer import (
    ObserverBase,
    MinMaxObserver,
    AvgMinMaxObserver,
    PercentileObserver,
    MSEObserver,
    get_observer,
    OBSERVER_MAP,
)

# FakeQuantize
from .fake_quant import (
    FakeQuantizeBase,
    FixedFakeQuantize,
    AdaRoundFakeQuantize,
    get_fake_quantize,
    FAKEQUANT_MAP,
)

# 量化模块
from .quantized_module import (
    QuantConv2d,
    QuantLinear,
    QuantMultiStepLIFNode,
    QuantBatchNorm2d,
    fold_bn_into_conv,
    # 向后兼容的别名
    QConv2d,
    QLinear,
    QMultiStepLIFNode,
    QBatchNorm2d,
)

# 工具函数
from .util_quant import (
    round_ste,
    fake_quantize_per_tensor_affine,
    fake_quantize_per_channel_affine,
)

# 状态管理
from .state import (
    enable_quantization,
    disable_quantization,
    enable_observer,
    disable_observer,
    calibrate_model,
    get_quant_state,
    print_quant_state,
    set_bit,
    freeze_quantization,
    get_quantized_layers,
    count_quantized_params,
)

# 量化主逻辑
from .quantize import (
    PTQConfig,
    quantize_model,
    adaround_optimize,
    fold_bn,
    replace_with_quantized,
    find_layers_to_quantize,
    find_lif_nodes,
    find_bn_after_conv,
    get_model_size,
    calculate_quantized_model_size,
    print_quantized_model,
)

# 主函数入口
from .main import main as ptq_main

__all__ = [
    # Observer
    'ObserverBase',
    'MinMaxObserver',
    'AvgMinMaxObserver',
    'PercentileObserver',
    'MSEObserver',
    'get_observer',
    'OBSERVER_MAP',

    # FakeQuantize
    'FakeQuantizeBase',
    'FixedFakeQuantize',
    'AdaRoundFakeQuantize',
    'get_fake_quantize',
    'FAKEQUANT_MAP',

    # 量化模块
    'QuantConv2d',
    'QuantLinear',
    'QuantMultiStepLIFNode',
    'QuantBatchNorm2d',
    'fold_bn_into_conv',
    'QConv2d',  # 别名
    'QLinear',  # 别名
    'QMultiStepLIFNode',  # 别名
    'QBatchNorm2d',  # 别名

    # 工具函数
    'round_ste',
    'fake_quantize_per_tensor_affine',
    'fake_quantize_per_channel_affine',

    # 状态管理
    'enable_quantization',
    'disable_quantization',
    'enable_observer',
    'disable_observer',
    'calibrate_model',
    'get_quant_state',
    'print_quant_state',
    'set_bit',
    'freeze_quantization',
    'get_quantized_layers',
    'count_quantized_params',

    # 量化主逻辑
    'PTQConfig',
    'quantize_model',
    'adaround_optimize',
    'fold_bn',
    'replace_with_quantized',
    'find_layers_to_quantize',
    'find_lif_nodes',
    'find_bn_after_conv',
    'get_model_size',
    'calculate_quantized_model_size',
    'print_quantized_model',

    # 主函数
    'ptq_main',
]
