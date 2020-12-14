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
    FINN_W2_A2 = 25
    FINN_W2_A2_X2 = 26
    FINN_W2_A2_X3 = 27
    FINN_W2_A2_X3_not_seq = 100
    FINN_W2_A8_X4 = 28
    FINN_W2_A4_X5 = 29
    FINN_W2_A16_X6 = 30
    FINN_W2_A2_X10 = 31
    FINN_W2_A2_X11 = 32
    FINN_W1_A1_X12 = 33
    FINN_W1_A1_X13 = 34
    FINN_W2_A4_X14 = 35
    FINN_W2_A8_X15 = 36
    FINN_W2_A16_X16 = 37
    FINN_W2_A2_X11_orig = 101
    FINN_W2_A1_X17 = 38
    FINN_W2_A1_X18 = 39
    FINN_W2_A2_X19 = 40
    FINN_W2_A4_X20 = 41
    FINN_W2_A8_X21 = 42
    FINN_W2_A16_X22 = 43
    FINN_W2_A1_X23 = 44
    FINN_W32_A32_X24_FP = 45
    FINN_W32_A32_X25_FP_z256_x130 = 46
    FINN_W2_A2_X26_z256_x130 = 47
    FINN_W2_A8_X27_z256_x130 = 48


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


class ZeroAct(nn.Module):
    def __init__(self, quant_type, bit_width, scaling_impl_type=ScalingImplType.CONST,
                 restrict_scaling_type=RestrictValueType.LOG_FP):
        super(ZeroAct, self).__init__()
        self._min_val_act = 0
        self._max_val_act = 255
        self._min_val = 0
        self._max_val = 255
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

        if network_type == NetType.FINN_W2_A2:
            # FINN_W2_A2
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 2
            WEIGHT_QUANT_TYPE = QuantType.INT
            WEIGHT_BIT_WIDTH = 2
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 2
            ACTIVATION_QUANT_TYPE = QuantType.INT
            ACTIVATION_BIT_WIDTH = 2
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
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
        elif network_type == NetType.FINN_W2_A2_X2:
            # FINN_W2_A2_X2
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 2
            WEIGHT_QUANT_TYPE = QuantType.INT
            WEIGHT_BIT_WIDTH = 2
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 2
            ACTIVATION_QUANT_TYPE = QuantType.INT
            ACTIVATION_BIT_WIDTH = 2
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            print("Network type: ")
            print(network_type)
            #self.conv_features = nn.Sequential(
                # MyQuantReLU(max_val=1 - 2 ** (1-ZERO_ACT_BIT_WIDTH),
                #             quant_type=ZERO_ACT_QUANT_TYPE,
                #             bit_width=ZERO_ACT_BIT_WIDTH,
                #             scaling_impl_type=ScalingImplType.CONST,
                #             restrict_scaling_type=RestrictValueType.POWER_OF_TWO),
            self.layer1 = qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                                     weight_quant_type=FIRST_LAYER_QUANT_TYPE, weight_bit_width=FIRST_LAYER_BIT_WIDTH,
                                     bias=False)
            self.layer2 = nn.BatchNorm2d(96, eps=1e-4)
            self.layer3 = MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                    ACTIVATION_SCALING_IMPL_TYPE)

            self.layer4 = nn.MaxPool2d(kernel_size=2, stride=2)

            self.layer5 = qnn.QuantConv2d(96, 256, 5, weight_quant_type=WEIGHT_QUANT_TYPE,
                                         weight_bit_width=WEIGHT_BIT_WIDTH, bias=False)
            self.layer6 = nn.BatchNorm2d(256, eps=1e-4)
            self.layer7 = MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                    ACTIVATION_SCALING_IMPL_TYPE)

            self.layer8 = qnn.QuantConv2d(256, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                         weight_bit_width=WEIGHT_BIT_WIDTH, bias=False)
            self.layer9 = nn.BatchNorm2d(384, eps=1e-4)
            self.layer10 = MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                    ACTIVATION_SCALING_IMPL_TYPE)

            self.layer11 = nn.MaxPool2d(kernel_size=3, stride=3)

            self.layer12 = qnn.QuantConv2d(384, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,
                                         weight_bit_width=WEIGHT_BIT_WIDTH, bias=False)
            self.layer13 = nn.BatchNorm2d(384, eps=1e-4)
            self.layer14 = MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,
                                    ACTIVATION_SCALING_IMPL_TYPE)


            self.layer15 = qnn.QuantConv2d(384, 256, 3, weight_quant_type=LAST_LAYER_QUANT_TYPE,
                                         weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False)


        elif network_type == NetType.FINN_W2_A2_X3_not_seq:

            # FINN_W2_A2_X3_not_seq

            FIRST_LAYER_QUANT_TYPE = QuantType.INT

            FIRST_LAYER_BIT_WIDTH = 2

            WEIGHT_QUANT_TYPE = QuantType.INT

            WEIGHT_BIT_WIDTH = 2

            LAST_LAYER_QUANT_TYPE = QuantType.INT

            LAST_LAYER_BIT_WIDTH = 2

            ACTIVATION_QUANT_TYPE = QuantType.INT

            ACTIVATION_BIT_WIDTH = 2

            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST

            ACTIVATION_MAX_VAL = 6

            print("Network type: ")

            print(network_type)

            self.layer1 = qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,

                                          weight_quant_type=FIRST_LAYER_QUANT_TYPE,
                                          weight_bit_width=FIRST_LAYER_BIT_WIDTH,

                                          bias=False)

            self.layer2 = nn.BatchNorm2d(96, eps=1e-4)

            self.layer3 = MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,

                                      ACTIVATION_SCALING_IMPL_TYPE)

            # self.layer4 = nn.MaxPool2d(kernel_size=2, stride=2)

            self.layer5 = qnn.QuantConv2d(96, 256, 5, weight_quant_type=WEIGHT_QUANT_TYPE,

                                          weight_bit_width=WEIGHT_BIT_WIDTH, bias=False)

            self.layer6 = nn.BatchNorm2d(256, eps=1e-4)

            self.layer7 = MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,

                                      ACTIVATION_SCALING_IMPL_TYPE)

            self.layer8 = qnn.QuantConv2d(256, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,

                                          weight_bit_width=WEIGHT_BIT_WIDTH, bias=False)

            self.layer9 = nn.BatchNorm2d(384, eps=1e-4)

            self.layer10 = MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,

                                       ACTIVATION_SCALING_IMPL_TYPE)

            # self.layer11 = nn.MaxPool2d(kernel_size=3, stride=3)

            self.layer12 = qnn.QuantConv2d(384, 384, 3, weight_quant_type=WEIGHT_QUANT_TYPE,

                                           weight_bit_width=WEIGHT_BIT_WIDTH, bias=False)

            self.layer13 = nn.BatchNorm2d(384, eps=1e-4)

            self.layer14 = MyQuantReLU(ACTIVATION_MAX_VAL, ACTIVATION_QUANT_TYPE, ACTIVATION_BIT_WIDTH,

                                       ACTIVATION_SCALING_IMPL_TYPE)

            self.layer15 = qnn.QuantConv2d(384, 256, 3, weight_quant_type=LAST_LAYER_QUANT_TYPE,

                                           weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False)

            self.zero_act = MyQuantReLU(max_val=1 - 2 ** (-7),

                                        quant_type=QuantType.INT,

                                        bit_width=8,

                                        scaling_impl_type=ScalingImplType.CONST,

                                        restrict_scaling_type=RestrictValueType.POWER_OF_TWO)

        elif network_type == NetType.FINN_W2_A2_X3:

            # FINN_W2_A2_X3
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 2
            WEIGHT_QUANT_TYPE = QuantType.INT
            WEIGHT_BIT_WIDTH = 2
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 2
            ACTIVATION_QUANT_TYPE = QuantType.INT
            ACTIVATION_BIT_WIDTH = 2
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            print("Network type: ")
            print(network_type)

            self.conv_features = nn.Sequential(
                MyQuantReLU(max_val=1 - 2 ** (-7),
                    quant_type=QuantType.INT,
                    bit_width=8,
                    scaling_impl_type=ScalingImplType.CONST,
                    restrict_scaling_type=RestrictValueType.POWER_OF_TWO),
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                    weight_quant_type=FIRST_LAYER_QUANT_TYPE,
                    weight_bit_width=FIRST_LAYER_BIT_WIDTH,
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
                    weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False)
            )

        elif network_type == NetType.FINN_W2_A8_X4:

            # FINN_W2_A2_X3
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 2
            WEIGHT_QUANT_TYPE = QuantType.INT
            WEIGHT_BIT_WIDTH = 2
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 2
            ACTIVATION_QUANT_TYPE = QuantType.INT
            ACTIVATION_BIT_WIDTH = 8
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            print("Network type: ")
            print(network_type)

            self.conv_features = nn.Sequential(
                MyQuantReLU(max_val=1 - 2 ** (-7),
                    quant_type=QuantType.INT,
                    bit_width=8,
                    scaling_impl_type=ScalingImplType.CONST,
                    restrict_scaling_type=RestrictValueType.POWER_OF_TWO),
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                    weight_quant_type=FIRST_LAYER_QUANT_TYPE,
                    weight_bit_width=FIRST_LAYER_BIT_WIDTH,
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
                    weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False)
            )

        elif network_type == NetType.FINN_W2_A4_X5:

            # FINN_W2_A2_X3
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 2
            WEIGHT_QUANT_TYPE = QuantType.INT
            WEIGHT_BIT_WIDTH = 2
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 2
            ACTIVATION_QUANT_TYPE = QuantType.INT
            ACTIVATION_BIT_WIDTH = 4
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            print("Network type: ")
            print(network_type)

            self.conv_features = nn.Sequential(
                MyQuantReLU(max_val=1 - 2 ** (-7),
                    quant_type=QuantType.INT,
                    bit_width=8,
                    scaling_impl_type=ScalingImplType.CONST,
                    restrict_scaling_type=RestrictValueType.POWER_OF_TWO),
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                    weight_quant_type=FIRST_LAYER_QUANT_TYPE,
                    weight_bit_width=FIRST_LAYER_BIT_WIDTH,
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
                    weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False)
            )

        elif network_type == NetType.FINN_W2_A16_X6:

            # FINN_W2_A2_X3
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 2
            WEIGHT_QUANT_TYPE = QuantType.INT
            WEIGHT_BIT_WIDTH = 2
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 2
            ACTIVATION_QUANT_TYPE = QuantType.INT
            ACTIVATION_BIT_WIDTH = 16
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            print("Network type: ")
            print(network_type)

            self.conv_features = nn.Sequential(
                MyQuantReLU(max_val=1 - 2 ** (-15),
                    quant_type=QuantType.INT,
                    bit_width=16,
                    scaling_impl_type=ScalingImplType.CONST,
                    restrict_scaling_type=RestrictValueType.POWER_OF_TWO),
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                    weight_quant_type=FIRST_LAYER_QUANT_TYPE,
                    weight_bit_width=FIRST_LAYER_BIT_WIDTH,
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
                    weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False)
            )

        elif network_type == NetType.FINN_W2_A2_X10:

            # FINN_W2_A2_X10
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 2
            WEIGHT_QUANT_TYPE = QuantType.INT
            WEIGHT_BIT_WIDTH = 2
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 2
            ACTIVATION_QUANT_TYPE = QuantType.INT
            ACTIVATION_BIT_WIDTH = 2
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            print("Network type: ")
            print(network_type)

            self.conv_features = nn.Sequential(
                # MyQuantReLU(max_val=1 - 2 ** (-7),
                #     quant_type=QuantType.INT,
                #     bit_width=8,
                #     scaling_impl_type=ScalingImplType.CONST,
                #     restrict_scaling_type=RestrictValueType.POWER_OF_TWO),
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                    weight_quant_type=FIRST_LAYER_QUANT_TYPE,
                    weight_bit_width=FIRST_LAYER_BIT_WIDTH,
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
                    weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False)
            )


        elif network_type == NetType.FINN_W2_A2_X11:

            # FINN_W2_A2_X11
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 2
            WEIGHT_QUANT_TYPE = QuantType.INT
            WEIGHT_BIT_WIDTH = 2
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 2
            ACTIVATION_QUANT_TYPE = QuantType.INT
            ACTIVATION_BIT_WIDTH = 2
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            print("Network type: ")
            print(network_type)

            self.conv_features = nn.Sequential(
                ZeroAct(max_val=1 - 2 ** (-7),
                    quant_type=QuantType.INT,
                    bit_width=8,
                    scaling_impl_type=ScalingImplType.CONST,
                    restrict_scaling_type=RestrictValueType.POWER_OF_TWO),
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                    weight_quant_type=FIRST_LAYER_QUANT_TYPE,
                    weight_bit_width=FIRST_LAYER_BIT_WIDTH,
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
                    weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False)
            )


        elif network_type == NetType.FINN_W1_A1_X12:

            # FINN_W1_A1_X12
            FIRST_LAYER_QUANT_TYPE = QuantType.BINARY
            FIRST_LAYER_BIT_WIDTH = 1
            WEIGHT_QUANT_TYPE = QuantType.BINARY
            WEIGHT_BIT_WIDTH = 1
            LAST_LAYER_QUANT_TYPE = QuantType.BINARY
            LAST_LAYER_BIT_WIDTH = 1
            ACTIVATION_QUANT_TYPE = QuantType.BINARY
            ACTIVATION_BIT_WIDTH = 1
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            print("Network type: ")
            print(network_type)

            self.conv_features = nn.Sequential(
                # MyQuantReLU(max_val=1 - 2 ** (-7),
                #     quant_type=QuantType.INT,
                #     bit_width=8,
                #     scaling_impl_type=ScalingImplType.CONST,
                #     restrict_scaling_type=RestrictValueType.POWER_OF_TWO),
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                    weight_quant_type=FIRST_LAYER_QUANT_TYPE,
                    weight_bit_width=FIRST_LAYER_BIT_WIDTH,
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
                    weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False)
            )


        elif network_type == NetType.FINN_W1_A1_X13:

            # FINN_W1_A1_X13
            FIRST_LAYER_QUANT_TYPE = QuantType.BINARY
            FIRST_LAYER_BIT_WIDTH = 1
            WEIGHT_QUANT_TYPE = QuantType.BINARY
            WEIGHT_BIT_WIDTH = 1
            LAST_LAYER_QUANT_TYPE = QuantType.BINARY
            LAST_LAYER_BIT_WIDTH = 1
            ACTIVATION_QUANT_TYPE = QuantType.BINARY
            ACTIVATION_BIT_WIDTH = 1
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            print("Network type: ")
            print(network_type)

            self.conv_features = nn.Sequential(
                MyQuantReLU(max_val=1 - 2 ** (-7),
                    quant_type=QuantType.INT,
                    bit_width=8,
                    scaling_impl_type=ScalingImplType.CONST,
                    restrict_scaling_type=RestrictValueType.POWER_OF_TWO),
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                    weight_quant_type=FIRST_LAYER_QUANT_TYPE,
                    weight_bit_width=FIRST_LAYER_BIT_WIDTH,
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
                    weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False)
            )

        elif network_type == NetType.FINN_W2_A4_X14:

            # FINN_W2_A4_X14
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 2
            WEIGHT_QUANT_TYPE = QuantType.INT
            WEIGHT_BIT_WIDTH = 2
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 2
            ACTIVATION_QUANT_TYPE = QuantType.INT
            ACTIVATION_BIT_WIDTH = 4
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            print("Network type: ")
            print(network_type)

            self.conv_features = nn.Sequential(
                # MyQuantReLU(max_val=1 - 2 ** (-7),
                #     quant_type=QuantType.INT,
                #     bit_width=8,
                #     scaling_impl_type=ScalingImplType.CONST,
                #     restrict_scaling_type=RestrictValueType.POWER_OF_TWO),
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                                weight_quant_type=FIRST_LAYER_QUANT_TYPE,
                                weight_bit_width=FIRST_LAYER_BIT_WIDTH,
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
                                weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False)
            )

        elif network_type == NetType.FINN_W2_A8_X15:

            # FINN_W2_A8_X15
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 2
            WEIGHT_QUANT_TYPE = QuantType.INT
            WEIGHT_BIT_WIDTH = 2
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 2
            ACTIVATION_QUANT_TYPE = QuantType.INT
            ACTIVATION_BIT_WIDTH = 8
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            print("Network type: ")
            print(network_type)

            self.conv_features = nn.Sequential(
                # MyQuantReLU(max_val=1 - 2 ** (-7),
                #     quant_type=QuantType.INT,
                #     bit_width=8,
                #     scaling_impl_type=ScalingImplType.CONST,
                #     restrict_scaling_type=RestrictValueType.POWER_OF_TWO),
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                                weight_quant_type=FIRST_LAYER_QUANT_TYPE,
                                weight_bit_width=FIRST_LAYER_BIT_WIDTH,
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
                                weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False)
            )

        elif network_type == NetType.FINN_W2_A16_X16:

            # FINN_W2_A16_X16
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 2
            WEIGHT_QUANT_TYPE = QuantType.INT
            WEIGHT_BIT_WIDTH = 2
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 2
            ACTIVATION_QUANT_TYPE = QuantType.INT
            ACTIVATION_BIT_WIDTH = 16
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            print("Network type: ")
            print(network_type)

            self.conv_features = nn.Sequential(
                # MyQuantReLU(max_val=1 - 2 ** (-7),
                #     quant_type=QuantType.INT,
                #     bit_width=8,
                #     scaling_impl_type=ScalingImplType.CONST,
                #     restrict_scaling_type=RestrictValueType.POWER_OF_TWO),
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                                weight_quant_type=FIRST_LAYER_QUANT_TYPE,
                                weight_bit_width=FIRST_LAYER_BIT_WIDTH,
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
                                weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False)
            )

        elif network_type == NetType.FINN_W2_A1_X17:

            # FINN_W2_A1_X17
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 2
            WEIGHT_QUANT_TYPE = QuantType.INT
            WEIGHT_BIT_WIDTH = 2
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 2
            ACTIVATION_QUANT_TYPE = QuantType.BINARY
            ACTIVATION_BIT_WIDTH = 1
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            print("Network type: ")
            print(network_type)

            self.conv_features = nn.Sequential(
                MyQuantReLU(max_val=1 - 2 ** (-7),
                    quant_type=QuantType.INT,
                    bit_width=8,
                    scaling_impl_type=ScalingImplType.CONST,
                    restrict_scaling_type=RestrictValueType.POWER_OF_TWO),
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                                weight_quant_type=FIRST_LAYER_QUANT_TYPE,
                                weight_bit_width=FIRST_LAYER_BIT_WIDTH,
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
                                weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False)
            )

        elif network_type == NetType.FINN_W2_A1_X18:

            # FINN_W2_A1_X18
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 2
            WEIGHT_QUANT_TYPE = QuantType.INT
            WEIGHT_BIT_WIDTH = 2
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 2
            ACTIVATION_QUANT_TYPE = QuantType.BINARY
            ACTIVATION_BIT_WIDTH = 1
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            print("Network type: ")
            print(network_type)

            self.conv_features = nn.Sequential(
                # MyQuantReLU(max_val=1 - 2 ** (-7),
                #     quant_type=QuantType.INT,
                #     bit_width=8,
                #     scaling_impl_type=ScalingImplType.CONST,
                #     restrict_scaling_type=RestrictValueType.POWER_OF_TWO),
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                                weight_quant_type=FIRST_LAYER_QUANT_TYPE,
                                weight_bit_width=FIRST_LAYER_BIT_WIDTH,
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
                                weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False)
            )

        elif network_type == NetType.FINN_W2_A2_X19:

            # FINN_W2_A2_X19
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 2
            WEIGHT_QUANT_TYPE = QuantType.INT
            WEIGHT_BIT_WIDTH = 2
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 2
            ACTIVATION_QUANT_TYPE = QuantType.INT
            ACTIVATION_BIT_WIDTH = 2
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            print("Network type: ")
            print(network_type)

            self.conv_features = nn.Sequential(
                ZeroAct(quant_type=QuantType.INT,
                    bit_width=8,
                    scaling_impl_type=ScalingImplType.CONST,
                    restrict_scaling_type=RestrictValueType.POWER_OF_TWO),
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                    weight_quant_type=FIRST_LAYER_QUANT_TYPE,
                    weight_bit_width=FIRST_LAYER_BIT_WIDTH,
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
                    weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False)
            )

        elif network_type == NetType.FINN_W2_A4_X20:

            # FINN_W2_A4_X20
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 2
            WEIGHT_QUANT_TYPE = QuantType.INT
            WEIGHT_BIT_WIDTH = 2
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 2
            ACTIVATION_QUANT_TYPE = QuantType.INT
            ACTIVATION_BIT_WIDTH = 4
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            print("Network type: ")
            print(network_type)

            self.conv_features = nn.Sequential(
                ZeroAct(quant_type=QuantType.INT,
                    bit_width=8,
                    scaling_impl_type=ScalingImplType.CONST,
                    restrict_scaling_type=RestrictValueType.POWER_OF_TWO),
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                    weight_quant_type=FIRST_LAYER_QUANT_TYPE,
                    weight_bit_width=FIRST_LAYER_BIT_WIDTH,
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
                    weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False)
            )

        elif network_type == NetType.FINN_W2_A8_X21:

            # FINN_W2_A8_X21
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 2
            WEIGHT_QUANT_TYPE = QuantType.INT
            WEIGHT_BIT_WIDTH = 2
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 2
            ACTIVATION_QUANT_TYPE = QuantType.INT
            ACTIVATION_BIT_WIDTH = 8
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            print("Network type: ")
            print(network_type)

            self.conv_features = nn.Sequential(
                ZeroAct(quant_type=QuantType.INT,
                    bit_width=8,
                    scaling_impl_type=ScalingImplType.CONST,
                    restrict_scaling_type=RestrictValueType.POWER_OF_TWO),
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                    weight_quant_type=FIRST_LAYER_QUANT_TYPE,
                    weight_bit_width=FIRST_LAYER_BIT_WIDTH,
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
                    weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False)
            )

        elif network_type == NetType.FINN_W2_A16_X22:

            # FINN_W2_A16_X22
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 2
            WEIGHT_QUANT_TYPE = QuantType.INT
            WEIGHT_BIT_WIDTH = 2
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 2
            ACTIVATION_QUANT_TYPE = QuantType.INT
            ACTIVATION_BIT_WIDTH = 16
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            print("Network type: ")
            print(network_type)

            self.conv_features = nn.Sequential(
                ZeroAct(quant_type=QuantType.INT,
                    bit_width=8,
                    scaling_impl_type=ScalingImplType.CONST,
                    restrict_scaling_type=RestrictValueType.POWER_OF_TWO),
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                    weight_quant_type=FIRST_LAYER_QUANT_TYPE,
                    weight_bit_width=FIRST_LAYER_BIT_WIDTH,
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
                    weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False)
            )

        elif network_type == NetType.FINN_W2_A1_X23:

            # FINN_W2_A16_X22
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 2
            WEIGHT_QUANT_TYPE = QuantType.INT
            WEIGHT_BIT_WIDTH = 2
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 2
            ACTIVATION_QUANT_TYPE = QuantType.BINARY
            ACTIVATION_BIT_WIDTH = 1
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            print("Network type: ")
            print(network_type)

            self.conv_features = nn.Sequential(
                ZeroAct(quant_type=QuantType.INT,
                    bit_width=8,
                    scaling_impl_type=ScalingImplType.CONST,
                    restrict_scaling_type=RestrictValueType.POWER_OF_TWO),
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                    weight_quant_type=FIRST_LAYER_QUANT_TYPE,
                    weight_bit_width=FIRST_LAYER_BIT_WIDTH,
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
                    weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False)
            )

        elif network_type == NetType.FINN_W32_A32_X24_FP:

            # FINN_W32_A32_X24_FP
            FIRST_LAYER_QUANT_TYPE = QuantType.FP
            FIRST_LAYER_BIT_WIDTH = 32
            WEIGHT_QUANT_TYPE = QuantType.FP
            WEIGHT_BIT_WIDTH = 32
            LAST_LAYER_QUANT_TYPE = QuantType.FP
            LAST_LAYER_BIT_WIDTH = 32
            ACTIVATION_QUANT_TYPE = QuantType.FP
            ACTIVATION_BIT_WIDTH = 32
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            print("Network type: ")
            print(network_type)

            self.conv_features = nn.Sequential(
                ZeroAct(quant_type=QuantType.INT,
                        bit_width=8,
                        scaling_impl_type=ScalingImplType.CONST,
                        restrict_scaling_type=RestrictValueType.POWER_OF_TWO),
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                                weight_quant_type=FIRST_LAYER_QUANT_TYPE,
                                weight_bit_width=FIRST_LAYER_BIT_WIDTH,
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
                                weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False)
            )

        elif network_type == NetType.FINN_W32_A32_X25_FP_z256_x130:

            # FINN_W32_A32_X25_FP_z256_x130
            FIRST_LAYER_QUANT_TYPE = QuantType.FP
            FIRST_LAYER_BIT_WIDTH = 32
            WEIGHT_QUANT_TYPE = QuantType.FP
            WEIGHT_BIT_WIDTH = 32
            LAST_LAYER_QUANT_TYPE = QuantType.FP
            LAST_LAYER_BIT_WIDTH = 32
            ACTIVATION_QUANT_TYPE = QuantType.FP
            ACTIVATION_BIT_WIDTH = 32
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            print("Network type: ")
            print(network_type)

            self.conv_features = nn.Sequential(
                ZeroAct(quant_type=QuantType.INT,
                        bit_width=8,
                        scaling_impl_type=ScalingImplType.CONST,
                        restrict_scaling_type=RestrictValueType.POWER_OF_TWO),
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                                weight_quant_type=FIRST_LAYER_QUANT_TYPE,
                                weight_bit_width=FIRST_LAYER_BIT_WIDTH,
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
                                weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False)
            )

        elif network_type == NetType.FINN_W2_A2_X26_z256_x130:

            # FINN_W2_A2_X26_z256_x130
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 2
            WEIGHT_QUANT_TYPE = QuantType.INT
            WEIGHT_BIT_WIDTH = 2
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 2
            ACTIVATION_QUANT_TYPE = QuantType.INT
            ACTIVATION_BIT_WIDTH = 2
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            print("Network type: ")
            print(network_type)

            self.conv_features = nn.Sequential(
                ZeroAct(quant_type=QuantType.INT,
                        bit_width=8,
                        scaling_impl_type=ScalingImplType.CONST,
                        restrict_scaling_type=RestrictValueType.POWER_OF_TWO),
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                                weight_quant_type=FIRST_LAYER_QUANT_TYPE,
                                weight_bit_width=FIRST_LAYER_BIT_WIDTH,
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
                                weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False)
            )

        elif network_type == NetType.FINN_W2_A8_X27_z256_x130:

            # FINN_W2_A8_X27_z256_x130
            FIRST_LAYER_QUANT_TYPE = QuantType.INT
            FIRST_LAYER_BIT_WIDTH = 2
            WEIGHT_QUANT_TYPE = QuantType.INT
            WEIGHT_BIT_WIDTH = 2
            LAST_LAYER_QUANT_TYPE = QuantType.INT
            LAST_LAYER_BIT_WIDTH = 2
            ACTIVATION_QUANT_TYPE = QuantType.INT
            ACTIVATION_BIT_WIDTH = 8
            ACTIVATION_SCALING_IMPL_TYPE = ScalingImplType.CONST
            ACTIVATION_MAX_VAL = 6
            print("Network type: ")
            print(network_type)

            self.conv_features = nn.Sequential(
                ZeroAct(quant_type=QuantType.INT,
                        bit_width=8,
                        scaling_impl_type=ScalingImplType.CONST,
                        restrict_scaling_type=RestrictValueType.POWER_OF_TWO),
                qnn.QuantConv2d(in_channels=3, out_channels=96, kernel_size=11,
                                weight_quant_type=FIRST_LAYER_QUANT_TYPE,
                                weight_bit_width=FIRST_LAYER_BIT_WIDTH,
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
                                weight_bit_width=LAST_LAYER_BIT_WIDTH, bias=False)
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
