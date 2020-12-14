"""
The architecture of SiamFC
Written by Heng Fan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from Train.Config import *
import brevitas.nn as qnn
from brevitas.core.quant import QuantType
import enum

from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType


class NetType(enum.Enum):
    X1_1 = 0
    X1_2 = 1
    X1_3 = 2
    X1_4 = 3
    X1_5 = 4
    X1_6 = 5

    X2_1 = 7
    X2_2 = 8
    X2_3 = 9
    X2_4 = 10
    X2_5 = 11
    X2_6 = 12

    X3_1 = 13
    X3_2 = 14
    X3_3 = 15
    X3_4 = 16
    X3_5 = 17
    X3_6 = 18

    X4_1 = 19
    X4_2 = 20
    X4_3 = 21
    X4_4 = 22
    X4_5 = 23
    X4_6 = 24

    # FINN_INT2 = 25


class MyQuantReLU(nn.Module):
    def __init__(self, max_val, quant_type, bit_width, scaling_impl_type=ScalingImplType.CONST,
                 restrict_scaling_type=RestrictValueType.LOG_FP):
        super(MyQuantReLU, self).__init__()
        self._min_val_act = -1
        self._max_val_act = 1 - 2 / (2 ** bit_width)
        self._min_val = 0
        self._max_val = max_val
        self._act = qnn.QuantHardTanh(scaling_impl_type=scaling_impl_type, restrict_scaling_type=restrict_scaling_type,
                                      min_val=self._min_val_act, max_val=self._max_val_act, quant_type=quant_type,
                                      bit_width=bit_width)
        self._scale = (self._max_val - self._min_val) / (self._max_val_act - self._min_val_act)

    def forward(self, x):
        x = self._act(x / self._scale - 1)
        x = self._scale * x
        x = x + self._scale
        return x


class SiamNet(nn.Module):
    def __init__(self, network_type=None):
        super(SiamNet, self).__init__()

        self.network_type = network_type

        if network_type == NetType.X1_1:
            # X1.1
            FIRST_LAYER_QUANT_TYPE = QuantType.FP
            FIRST_LAYER_BIT_WIDTH = 32
            WEIGHT_QUANT_TYPE = QuantType.FP
            WEIGHT_BIT_WIDTH = 32
            LAST_LAYER_BIT_WIDTH = 32
            LAST_LAYER_QUANT_TYPE = QuantType.FP
            ACTIVATION_QUANT_TYPE = QuantType.FP
            ACTIVATION_BIT_WIDTH = 32
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            ACTIVATION_MIN_VAL = 0
            print("Network type: ")
            print(network_type)
            self.conv_features = nn.Sequential(
                # MyQuantReLU(max_val=1 - 2 ** (1-ZERO_ACT_BIT_WIDTH),
                #             quant_type=ZERO_ACT_QUANT_TYPE,
                #             bit_width=ZERO_ACT_BIT_WIDTH,
                #             scaling_impl_type=ScalingImplType.CONST,
                #             restrict_scaling_type=RestrictValueType.POWER_OF_TWO),
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                                         weight_quant_type=FIRST_LAYER_QUANT_TYPE, weight_bit_width=FIRST_LAYER_BIT_WIDTH,
                                         bias=False),
                nn.BatchNorm2d(96, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=2, stride=2),

                qnn.QuantConv2d(96, 256, 5, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(256, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                qnn.QuantConv2d(256, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=3, stride=3),

                qnn.QuantConv2d(384, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),


                qnn.QuantConv2d(384, 256, 3, weight_quant_type=LAST_LAYER_QUANT_TYPE,
                                             weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False),

            )
        elif network_type == NetType.X1_2:
            #X1.2
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 8
            WEIGHT_QUANT_TYPE = QuantType.INT
            WEIGHT_BIT_WIDTH = 16
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 8
            ACTIVATION_QUANT_TYPE = QuantType.INT
            ACTIVATION_BIT_WIDTH = 32
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            ACTIVATION_MIN_VAL = 0
            ##
            print("Network type: ")
            print(network_type)
            self.conv_features = nn.Sequential(
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                                         weight_quant_type=FIRST_LAYER_QUANT_TYPE, weight_bit_width=FIRST_LAYER_BIT_WIDTH,
                                         bias=False),
                nn.BatchNorm2d(96, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=2, stride=2),

                qnn.QuantConv2d(96, 256, 5, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(256, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                qnn.QuantConv2d(256, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=3, stride=3),

                qnn.QuantConv2d(384, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),


                qnn.QuantConv2d(384, 256, 3, weight_quant_type=LAST_LAYER_QUANT_TYPE,
                                             weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False),

            )
        elif network_type == NetType.X1_3:
            #X1.3
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 8
            WEIGHT_QUANT_TYPE = QuantType.INT
            WEIGHT_BIT_WIDTH = 8
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 8
            ACTIVATION_QUANT_TYPE = QuantType.INT
            ACTIVATION_BIT_WIDTH = 32
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            ACTIVATION_MIN_VAL = 0
            ##
            print("Network type: ")
            print(network_type)
            self.conv_features = nn.Sequential(
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                                         weight_quant_type=FIRST_LAYER_QUANT_TYPE, weight_bit_width=FIRST_LAYER_BIT_WIDTH,
                                         bias=False),
                nn.BatchNorm2d(96, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=2, stride=2),

                qnn.QuantConv2d(96, 256, 5, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(256, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                qnn.QuantConv2d(256, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=3, stride=3),

                qnn.QuantConv2d(384, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),


                qnn.QuantConv2d(384, 256, 3, weight_quant_type=LAST_LAYER_QUANT_TYPE,
                                             weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False),

            )
        elif network_type == NetType.X1_4:
            #X1.4
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 8
            WEIGHT_QUANT_TYPE = QuantType.INT
            WEIGHT_BIT_WIDTH = 4
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 8
            ACTIVATION_QUANT_TYPE = QuantType.INT
            ACTIVATION_BIT_WIDTH = 32
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            ACTIVATION_MIN_VAL = 0
            ##
            print("Network type: ")
            print(network_type)
            self.conv_features = nn.Sequential(
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                                         weight_quant_type=FIRST_LAYER_QUANT_TYPE, weight_bit_width=FIRST_LAYER_BIT_WIDTH,
                                         bias=False),
                nn.BatchNorm2d(96, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=2, stride=2),

                qnn.QuantConv2d(96, 256, 5, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(256, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                qnn.QuantConv2d(256, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=3, stride=3),

                qnn.QuantConv2d(384, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),


                qnn.QuantConv2d(384, 256, 3, weight_quant_type=LAST_LAYER_QUANT_TYPE,
                                             weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False),

            )
        elif network_type == NetType.X1_5:
            #X1.5
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 8
            WEIGHT_QUANT_TYPE = QuantType.INT
            WEIGHT_BIT_WIDTH = 2
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 8
            ACTIVATION_QUANT_TYPE = QuantType.INT
            ACTIVATION_BIT_WIDTH = 32
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            ACTIVATION_MIN_VAL = 0
            ##
            print("Network type: ")
            print(network_type)
            self.conv_features = nn.Sequential(
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                                         weight_quant_type=FIRST_LAYER_QUANT_TYPE, weight_bit_width=FIRST_LAYER_BIT_WIDTH,
                                         bias=False),
                nn.BatchNorm2d(96, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=2, stride=2),

                qnn.QuantConv2d(96, 256, 5, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(256, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                qnn.QuantConv2d(256, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=3, stride=3),

                qnn.QuantConv2d(384, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),


                qnn.QuantConv2d(384, 256, 3, weight_quant_type=LAST_LAYER_QUANT_TYPE,
                                             weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False),

            )
        elif network_type == NetType.X1_6:
            #X1.6
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 8
            WEIGHT_QUANT_TYPE = QuantType.BINARY
            WEIGHT_BIT_WIDTH = 1
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 8
            ACTIVATION_QUANT_TYPE = QuantType.INT
            ACTIVATION_BIT_WIDTH = 32
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            ACTIVATION_MIN_VAL = 0
            ##
            print("Network type: ")
            print(network_type)
            self.conv_features = nn.Sequential(
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                                         weight_quant_type=FIRST_LAYER_QUANT_TYPE, weight_bit_width=FIRST_LAYER_BIT_WIDTH,
                                         bias=False),
                nn.BatchNorm2d(96, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=2, stride=2),

                qnn.QuantConv2d(96, 256, 5, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(256, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                qnn.QuantConv2d(256, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=3, stride=3),

                qnn.QuantConv2d(384, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),


                qnn.QuantConv2d(384, 256, 3, weight_quant_type=LAST_LAYER_QUANT_TYPE,
                                             weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False),

            )
        elif network_type == NetType.X2_1:
            # X2.1
            FIRST_LAYER_QUANT_TYPE = QuantType.FP
            FIRST_LAYER_BIT_WIDTH = 16
            WEIGHT_QUANT_TYPE = QuantType.FP
            WEIGHT_BIT_WIDTH = 16
            LAST_LAYER_QUANT_TYPE = QuantType.FP
            LAST_LAYER_BIT_WIDTH = 16
            ACTIVATION_QUANT_TYPE = QuantType.FP
            ACTIVATION_BIT_WIDTH = 16
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            ACTIVATION_MIN_VAL = 0
            print("Network type: ")
            print(network_type)
            self.conv_features = nn.Sequential(
                # MyQuantReLU(max_val=1 - 2 ** (1-ZERO_ACT_BIT_WIDTH),
                #             quant_type=ZERO_ACT_QUANT_TYPE,
                #             bit_width=ZERO_ACT_BIT_WIDTH,
                #             scaling_impl_type=ScalingImplType.CONST,
                #             restrict_scaling_type=RestrictValueType.POWER_OF_TWO),
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                                         weight_quant_type=FIRST_LAYER_QUANT_TYPE, weight_bit_width=FIRST_LAYER_BIT_WIDTH,
                                         bias=False),
                nn.BatchNorm2d(96, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=2, stride=2),

                qnn.QuantConv2d(96, 256, 5, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(256, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                qnn.QuantConv2d(256, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=3, stride=3),

                qnn.QuantConv2d(384, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),


                qnn.QuantConv2d(384, 256, 3, weight_quant_type=LAST_LAYER_QUANT_TYPE,
                                             weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False),

            )
        elif network_type == NetType.X2_2:
            #X2.2
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 8
            WEIGHT_QUANT_TYPE = QuantType.INT
            WEIGHT_BIT_WIDTH = 16
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 8
            ACTIVATION_QUANT_TYPE = QuantType.INT
            ACTIVATION_BIT_WIDTH = 16
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            ACTIVATION_MIN_VAL = 0
            ##
            print("Network type: ")
            print(network_type)
            self.conv_features = nn.Sequential(
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                                         weight_quant_type=FIRST_LAYER_QUANT_TYPE, weight_bit_width=FIRST_LAYER_BIT_WIDTH,
                                         bias=False),
                nn.BatchNorm2d(96, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=2, stride=2),

                qnn.QuantConv2d(96, 256, 5, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(256, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                qnn.QuantConv2d(256, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=3, stride=3),

                qnn.QuantConv2d(384, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),


                qnn.QuantConv2d(384, 256, 3, weight_quant_type=LAST_LAYER_QUANT_TYPE,
                                             weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False),

            )
        elif network_type == NetType.X2_3:
            #X2.3
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 8
            WEIGHT_QUANT_TYPE = QuantType.INT
            WEIGHT_BIT_WIDTH = 8
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 8
            ACTIVATION_QUANT_TYPE = QuantType.INT
            ACTIVATION_BIT_WIDTH = 16
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            ACTIVATION_MIN_VAL = 0
            ##
            print("Network type: ")
            print(network_type)
            self.conv_features = nn.Sequential(
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                                         weight_quant_type=FIRST_LAYER_QUANT_TYPE, weight_bit_width=FIRST_LAYER_BIT_WIDTH,
                                         bias=False),
                nn.BatchNorm2d(96, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=2, stride=2),

                qnn.QuantConv2d(96, 256, 5, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(256, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                qnn.QuantConv2d(256, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=3, stride=3),

                qnn.QuantConv2d(384, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),


                qnn.QuantConv2d(384, 256, 3, weight_quant_type=LAST_LAYER_QUANT_TYPE,
                                             weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False),

            )
        elif network_type == NetType.X2_4:
            #X2.4
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 8
            WEIGHT_QUANT_TYPE = QuantType.INT
            WEIGHT_BIT_WIDTH = 4
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 8
            ACTIVATION_QUANT_TYPE = QuantType.INT
            ACTIVATION_BIT_WIDTH = 16
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            ACTIVATION_MIN_VAL = 0
            ##
            print("Network type: ")
            print(network_type)
            self.conv_features = nn.Sequential(
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                                         weight_quant_type=FIRST_LAYER_QUANT_TYPE, weight_bit_width=FIRST_LAYER_BIT_WIDTH,
                                         bias=False),
                nn.BatchNorm2d(96, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=2, stride=2),

                qnn.QuantConv2d(96, 256, 5, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(256, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                qnn.QuantConv2d(256, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=3, stride=3),

                qnn.QuantConv2d(384, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),


                qnn.QuantConv2d(384, 256, 3, weight_quant_type=LAST_LAYER_QUANT_TYPE,
                                             weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False),

            )
        elif network_type == NetType.X2_5:
            #X2.5
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 8
            WEIGHT_QUANT_TYPE = QuantType.INT
            WEIGHT_BIT_WIDTH = 2
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 8
            ACTIVATION_QUANT_TYPE = QuantType.INT
            ACTIVATION_BIT_WIDTH = 16
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            ACTIVATION_MIN_VAL = 0
            ##
            print("Network type: ")
            print(network_type)
            self.conv_features = nn.Sequential(
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                                         weight_quant_type=FIRST_LAYER_QUANT_TYPE, weight_bit_width=FIRST_LAYER_BIT_WIDTH,
                                         bias=False),
                nn.BatchNorm2d(96, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=2, stride=2),

                qnn.QuantConv2d(96, 256, 5, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(256, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                qnn.QuantConv2d(256, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=3, stride=3),

                qnn.QuantConv2d(384, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),


                qnn.QuantConv2d(384, 256, 3, weight_quant_type=LAST_LAYER_QUANT_TYPE,
                                             weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False),

            )
        elif network_type == NetType.X2_6:
            #X2.6
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 8
            WEIGHT_QUANT_TYPE = QuantType.BINARY
            WEIGHT_BIT_WIDTH = 1
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 8
            ACTIVATION_QUANT_TYPE = QuantType.INT
            ACTIVATION_BIT_WIDTH = 16
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            ACTIVATION_MIN_VAL = 0
            ##
            print("Network type: ")
            print(network_type)
            self.conv_features = nn.Sequential(
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                                         weight_quant_type=FIRST_LAYER_QUANT_TYPE, weight_bit_width=FIRST_LAYER_BIT_WIDTH,
                                         bias=False),
                nn.BatchNorm2d(96, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=2, stride=2),

                qnn.QuantConv2d(96, 256, 5, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(256, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                qnn.QuantConv2d(256, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=3, stride=3),

                qnn.QuantConv2d(384, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),


                qnn.QuantConv2d(384, 256, 3, weight_quant_type=LAST_LAYER_QUANT_TYPE,
                                             weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False),

            )
        elif network_type == NetType.X3_1:
            # X3.1
            FIRST_LAYER_QUANT_TYPE = QuantType.FP
            FIRST_LAYER_BIT_WIDTH = 8
            WEIGHT_QUANT_TYPE = QuantType.FP
            WEIGHT_BIT_WIDTH = 8
            LAST_LAYER_QUANT_TYPE = QuantType.FP
            LAST_LAYER_BIT_WIDTH = 8
            ACTIVATION_QUANT_TYPE = QuantType.FP
            ACTIVATION_BIT_WIDTH = 8
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            ACTIVATION_MIN_VAL = 0
            print("Network type: ")
            print(network_type)
            self.conv_features = nn.Sequential(
                # MyQuantReLU(max_val=1 - 2 ** (1-ZERO_ACT_BIT_WIDTH),
                #             quant_type=ZERO_ACT_QUANT_TYPE,
                #             bit_width=ZERO_ACT_BIT_WIDTH,
                #             scaling_impl_type=ScalingImplType.CONST,
                #             restrict_scaling_type=RestrictValueType.POWER_OF_TWO),
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                                         weight_quant_type=FIRST_LAYER_QUANT_TYPE, weight_bit_width=FIRST_LAYER_BIT_WIDTH,
                                         bias=False),
                nn.BatchNorm2d(96, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=2, stride=2),

                qnn.QuantConv2d(96, 256, 5, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(256, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                qnn.QuantConv2d(256, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=3, stride=3),

                qnn.QuantConv2d(384, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),


                qnn.QuantConv2d(384, 256, 3, weight_quant_type=LAST_LAYER_QUANT_TYPE,
                                             weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False),

            )
        elif network_type == NetType.X3_2:
            #X3.2
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 4
            WEIGHT_QUANT_TYPE = QuantType.INT
            WEIGHT_BIT_WIDTH = 16
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 4
            ACTIVATION_QUANT_TYPE = QuantType.INT
            ACTIVATION_BIT_WIDTH = 8
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            ACTIVATION_MIN_VAL = 0
            ##
            print("Network type: ")
            print(network_type)
            self.conv_features = nn.Sequential(
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                                         weight_quant_type=FIRST_LAYER_QUANT_TYPE, weight_bit_width=FIRST_LAYER_BIT_WIDTH,
                                         bias=False),
                nn.BatchNorm2d(96, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=2, stride=2),

                qnn.QuantConv2d(96, 256, 5, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(256, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                qnn.QuantConv2d(256, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=3, stride=3),

                qnn.QuantConv2d(384, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),


                qnn.QuantConv2d(384, 256, 3, weight_quant_type=LAST_LAYER_QUANT_TYPE,
                                             weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False),

            )
        elif network_type == NetType.X3_3:
            #X3.3
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 4
            WEIGHT_QUANT_TYPE = QuantType.INT
            WEIGHT_BIT_WIDTH = 8
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 4
            ACTIVATION_QUANT_TYPE = QuantType.INT
            ACTIVATION_BIT_WIDTH = 8
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            ACTIVATION_MIN_VAL = 0
            ##
            print("Network type: ")
            print(network_type)
            self.conv_features = nn.Sequential(
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                                         weight_quant_type=FIRST_LAYER_QUANT_TYPE, weight_bit_width=FIRST_LAYER_BIT_WIDTH,
                                         bias=False),
                nn.BatchNorm2d(96, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=2, stride=2),

                qnn.QuantConv2d(96, 256, 5, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(256, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                qnn.QuantConv2d(256, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=3, stride=3),

                qnn.QuantConv2d(384, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),


                qnn.QuantConv2d(384, 256, 3, weight_quant_type=LAST_LAYER_QUANT_TYPE,
                                             weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False),

            )
        elif network_type == NetType.X3_4:
            #X3.4
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 4
            WEIGHT_QUANT_TYPE = QuantType.INT
            WEIGHT_BIT_WIDTH = 4
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 4
            ACTIVATION_QUANT_TYPE = QuantType.INT
            ACTIVATION_BIT_WIDTH = 8
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            ACTIVATION_MIN_VAL = 0
            ##
            print("Network type: ")
            print(network_type)
            self.conv_features = nn.Sequential(
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                                         weight_quant_type=FIRST_LAYER_QUANT_TYPE, weight_bit_width=FIRST_LAYER_BIT_WIDTH,
                                         bias=False),
                nn.BatchNorm2d(96, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=2, stride=2),

                qnn.QuantConv2d(96, 256, 5, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(256, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                qnn.QuantConv2d(256, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=3, stride=3),

                qnn.QuantConv2d(384, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),


                qnn.QuantConv2d(384, 256, 3, weight_quant_type=LAST_LAYER_QUANT_TYPE,
                                             weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False),

            )
        elif network_type == NetType.X3_5:
            #X3.5
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 4
            WEIGHT_QUANT_TYPE = QuantType.INT
            WEIGHT_BIT_WIDTH = 2
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 4
            ACTIVATION_QUANT_TYPE = QuantType.INT
            ACTIVATION_BIT_WIDTH = 8
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            ACTIVATION_MIN_VAL = 0
            ##
            print("Network type: ")
            print(network_type)
            self.conv_features = nn.Sequential(
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                                         weight_quant_type=FIRST_LAYER_QUANT_TYPE, weight_bit_width=FIRST_LAYER_BIT_WIDTH,
                                         bias=False),
                nn.BatchNorm2d(96, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=2, stride=2),

                qnn.QuantConv2d(96, 256, 5, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(256, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                qnn.QuantConv2d(256, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=3, stride=3),

                qnn.QuantConv2d(384, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),


                qnn.QuantConv2d(384, 256, 3, weight_quant_type=LAST_LAYER_QUANT_TYPE,
                                             weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False),

            )
        elif network_type == NetType.X3_6:
            #X3.6
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 4
            WEIGHT_QUANT_TYPE = QuantType.BINARY
            WEIGHT_BIT_WIDTH = 1
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 4
            ACTIVATION_QUANT_TYPE = QuantType.INT
            ACTIVATION_BIT_WIDTH = 8
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            ACTIVATION_MIN_VAL = 0
            ##
            print("Network type: ")
            print(network_type)
            self.conv_features = nn.Sequential(
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                                         weight_quant_type=FIRST_LAYER_QUANT_TYPE, weight_bit_width=FIRST_LAYER_BIT_WIDTH,
                                         bias=False),
                nn.BatchNorm2d(96, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=2, stride=2),

                qnn.QuantConv2d(96, 256, 5, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(256, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                qnn.QuantConv2d(256, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=3, stride=3),

                qnn.QuantConv2d(384, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),


                qnn.QuantConv2d(384, 256, 3, weight_quant_type=LAST_LAYER_QUANT_TYPE,
                                             weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False),

            )
        elif network_type == NetType.X4_1:
            # X4.1
            FIRST_LAYER_QUANT_TYPE = QuantType.FP
            FIRST_LAYER_BIT_WIDTH = 8
            WEIGHT_QUANT_TYPE = QuantType.FP
            WEIGHT_BIT_WIDTH = 8
            LAST_LAYER_QUANT_TYPE = QuantType.FP
            LAST_LAYER_BIT_WIDTH = 8
            ACTIVATION_QUANT_TYPE = QuantType.FP
            ACTIVATION_BIT_WIDTH = 8
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            ACTIVATION_MIN_VAL = 0
            print("Network type: ")
            print(network_type)
            self.conv_features = nn.Sequential(
                # MyQuantReLU(max_val=1 - 2 ** (1-ZERO_ACT_BIT_WIDTH),
                #             quant_type=ZERO_ACT_QUANT_TYPE,
                #             bit_width=ZERO_ACT_BIT_WIDTH,
                #             scaling_impl_type=ScalingImplType.CONST,
                #             restrict_scaling_type=RestrictValueType.POWER_OF_TWO),
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                                         weight_quant_type=FIRST_LAYER_QUANT_TYPE, weight_bit_width=FIRST_LAYER_BIT_WIDTH,
                                         bias=False),
                nn.BatchNorm2d(96, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=2, stride=2),

                qnn.QuantConv2d(96, 256, 5, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(256, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                qnn.QuantConv2d(256, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=3, stride=3),

                qnn.QuantConv2d(384, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),


                qnn.QuantConv2d(384, 256, 3, weight_quant_type=LAST_LAYER_QUANT_TYPE,
                                             weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False),

            )
        elif network_type == NetType.X4_2:
            #X4.2
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 4
            WEIGHT_QUANT_TYPE = QuantType.INT
            WEIGHT_BIT_WIDTH = 16
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 4
            ACTIVATION_QUANT_TYPE = QuantType.INT
            ACTIVATION_BIT_WIDTH = 8
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            ACTIVATION_MIN_VAL = 0
            ##
            print("Network type: ")
            print(network_type)
            self.conv_features = nn.Sequential(
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                                         weight_quant_type=FIRST_LAYER_QUANT_TYPE, weight_bit_width=FIRST_LAYER_BIT_WIDTH,
                                         bias=False),
                nn.BatchNorm2d(96, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=2, stride=2),

                qnn.QuantConv2d(96, 256, 5, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(256, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                qnn.QuantConv2d(256, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=3, stride=3),

                qnn.QuantConv2d(384, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),


                qnn.QuantConv2d(384, 256, 3, weight_quant_type=LAST_LAYER_QUANT_TYPE,
                                             weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False),

            )
        elif network_type == NetType.X4_3:
            #X4.3
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 4
            WEIGHT_QUANT_TYPE = QuantType.INT
            WEIGHT_BIT_WIDTH = 8
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 4
            ACTIVATION_QUANT_TYPE = QuantType.INT
            ACTIVATION_BIT_WIDTH = 8
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            ACTIVATION_MIN_VAL = 0
            ##
            print("Network type: ")
            print(network_type)
            self.conv_features = nn.Sequential(
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                                         weight_quant_type=FIRST_LAYER_QUANT_TYPE, weight_bit_width=FIRST_LAYER_BIT_WIDTH,
                                         bias=False),
                nn.BatchNorm2d(96, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=2, stride=2),

                qnn.QuantConv2d(96, 256, 5, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(256, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                qnn.QuantConv2d(256, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=3, stride=3),

                qnn.QuantConv2d(384, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),


                qnn.QuantConv2d(384, 256, 3, weight_quant_type=LAST_LAYER_QUANT_TYPE,
                                             weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False),

            )
        elif network_type == NetType.X4_4:
            #X4.4
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 4
            WEIGHT_QUANT_TYPE = QuantType.INT
            WEIGHT_BIT_WIDTH = 4
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 4
            ACTIVATION_QUANT_TYPE = QuantType.INT
            ACTIVATION_BIT_WIDTH = 8
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            ACTIVATION_MIN_VAL = 0
            ##
            print("Network type: ")
            print(network_type)
            self.conv_features = nn.Sequential(
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                                         weight_quant_type=FIRST_LAYER_QUANT_TYPE, weight_bit_width=FIRST_LAYER_BIT_WIDTH,
                                         bias=False),
                nn.BatchNorm2d(96, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=2, stride=2),

                qnn.QuantConv2d(96, 256, 5, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(256, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                qnn.QuantConv2d(256, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=3, stride=3),

                qnn.QuantConv2d(384, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),


                qnn.QuantConv2d(384, 256, 3, weight_quant_type=LAST_LAYER_QUANT_TYPE,
                                             weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False),

            )
        elif network_type == NetType.X4_5:
            #X4.5
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 4
            WEIGHT_QUANT_TYPE = QuantType.INT
            WEIGHT_BIT_WIDTH = 2
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 4
            ACTIVATION_QUANT_TYPE = QuantType.INT
            ACTIVATION_BIT_WIDTH = 8
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            ACTIVATION_MIN_VAL = 0
            ##
            print("Network type: ")
            print(network_type)
            self.conv_features = nn.Sequential(
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                                         weight_quant_type=FIRST_LAYER_QUANT_TYPE, weight_bit_width=FIRST_LAYER_BIT_WIDTH,
                                         bias=False),
                nn.BatchNorm2d(96, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=2, stride=2),

                qnn.QuantConv2d(96, 256, 5, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(256, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                qnn.QuantConv2d(256, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=3, stride=3),

                qnn.QuantConv2d(384, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),


                qnn.QuantConv2d(384, 256, 3, weight_quant_type=LAST_LAYER_QUANT_TYPE,
                                             weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False),

            )
        elif network_type == NetType.X4_6:
            #X4.6
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 4
            WEIGHT_QUANT_TYPE = QuantType.BINARY
            WEIGHT_BIT_WIDTH = 1
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 4
            ACTIVATION_QUANT_TYPE = QuantType.INT
            ACTIVATION_BIT_WIDTH = 8
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            ACTIVATION_MIN_VAL = 0
            ##
            print("Network type: ")
            print(network_type)
            self.conv_features = nn.Sequential(
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                                         weight_quant_type=FIRST_LAYER_QUANT_TYPE, weight_bit_width=FIRST_LAYER_BIT_WIDTH,
                                         bias=False),
                nn.BatchNorm2d(96, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=2, stride=2),

                qnn.QuantConv2d(96, 256, 5, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(256, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                qnn.QuantConv2d(256, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),

                nn.MaxPool2d(kernel_size=3, stride=3),

                qnn.QuantConv2d(384, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                             weight_bit_width=WEIGHT_BIT_WIDTH, bias=False),
                nn.BatchNorm2d(384, eps=1e-4),
                MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                        ACTIVATION_SCALING_IMPL_TYPE),


                qnn.QuantConv2d(384, 256, 3, weight_quant_type=LAST_LAYER_QUANT_TYPE,
                                             weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False),

            )
        else:
            print("Error")
            exit()

        # adjust layer as in the original SiamFC in matconvnet
        self.adjust = nn.Conv2d(1, 1, 1, 1)

        # initialize weights
        self._initialize_weight()

        self.config = Config()

    def forward(self, z, x):
        """
        forward pass
        z: examplare, BxCxHxW
        x: search region, BxCxH1xW1
        """
        # get features for z and x
        z_feat = self.conv_features(z)
        x_feat = self.conv_features(x)

        # correlation of z and z
        xcorr_out = self.xcorr(z_feat, x_feat)

        score = self.adjust(xcorr_out)

        return score

    def xcorr(self, z, x):
        """
        correlation layer as in the original SiamFC (convolution process in fact)
        """
        batch_size_x, channel_x, w_x, h_x = x.shape
        x = torch.reshape(x, (1, batch_size_x * channel_x, w_x, h_x))

        # group convolution
        out = F.conv2d(x, z, groups=batch_size_x)

        batch_size_out, channel_out, w_out, h_out = out.shape
        xcorr_out = torch.reshape(out, (channel_out, batch_size_out, w_out, h_out))

        return xcorr_out

    def _initialize_weight(self):
        """
        initialize network parameters
        """
        tmp_layer_idx = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                tmp_layer_idx = tmp_layer_idx + 1
                if tmp_layer_idx < 6:
                    # kaiming initialization
                    nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                else:
                    # initialization for adjust layer as in the original paper
                    m.weight.data.fill_(1e-3)
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def weight_loss(self, prediction, label, weight):
        """
        weighted cross entropy loss
        """

        # if self.network_type != NetType.BASELINE:
        loss = F.binary_cross_entropy_with_logits(prediction, label, weight, reduction='mean')
        # else:
        #     loss = F.binary_cross_entropy_with_logits(prediction, label, weight,
        #                                               size_average=False) / self.config.batch_size

        return loss
