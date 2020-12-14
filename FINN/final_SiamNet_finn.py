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
    FINN_W2_A2_X28 = 49
    FINN_W2_A2_X29 = 50
    FINN_W2_A2_X30 = 51
    FINN_W2_A2_X31 = 52


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


def create_model(first_layer_quant_type, first_layer_bit_width, weight_quant_type, weight_bit_width,
                last_layer_quant_type, last_layer_bit_width, activation_quant_type, activation_bit_width,
                activation_scaling_impl_type, activation_max_val, chan_mult=1):
    return nn.Sequential(
        ZeroAct(quant_type=QuantType.INT,
                bit_width=8,
                scaling_impl_type=ScalingImplType.CONST,
                restrict_scaling_type=RestrictValueType.POWER_OF_TWO),
        qnn.QuantConv2d(in_channels=3, out_channels=int(chan_mult*96), kernel_size=11,
                        weight_quant_type=first_layer_quant_type,
                        weight_bit_width=first_layer_bit_width,
                        bias=False),

        nn.BatchNorm2d(int(chan_mult*96), eps=1e-4),
        MyQuantReLU(activation_max_val, activation_quant_type, activation_bit_width,
                    activation_scaling_impl_type),

        nn.MaxPool2d(kernel_size=2, stride=2),

        qnn.QuantConv2d(int(chan_mult*96), int(chan_mult*256), 5, weight_quant_type=weight_quant_type,
                        weight_bit_width=weight_bit_width, bias=False),
        nn.BatchNorm2d(int(chan_mult*256), eps=1e-4),
        MyQuantReLU(activation_max_val, activation_quant_type, activation_bit_width,
                    activation_scaling_impl_type),

        qnn.QuantConv2d(int(chan_mult*256), int(chan_mult*384), 3, weight_quant_type=weight_quant_type,
                        weight_bit_width=weight_bit_width, bias=False),
        nn.BatchNorm2d(int(chan_mult*384), eps=1e-4),
        MyQuantReLU(activation_max_val, activation_quant_type, activation_bit_width,
                    activation_scaling_impl_type),

        nn.MaxPool2d(kernel_size=3, stride=3),

        qnn.QuantConv2d(int(chan_mult*384), int(chan_mult*384), 3, weight_quant_type=weight_quant_type,
                        weight_bit_width=weight_bit_width, bias=False),

        nn.BatchNorm2d(int(chan_mult*384), eps=1e-4),
        MyQuantReLU(activation_max_val, activation_quant_type, activation_bit_width,
                    activation_scaling_impl_type),

        qnn.QuantConv2d(int(chan_mult*384), int(chan_mult*256), 3, weight_quant_type=last_layer_quant_type,
                        weight_bit_width=last_layer_bit_width, bias=False)
    )


class SiamNet(nn.Module):
    def __init__(self, network_type=None):
        super(SiamNet, self).__init__()
        self.network_type = network_type

        if network_type == NetType.FINN_W2_A2_X28:
            print("Network type: ")
            print(network_type)
            self.conv_features = create_model(first_layer_quant_type = QuantType.INT,
                                                first_layer_bit_width = 2,
                                                weight_quant_type = QuantType.INT,
                                                weight_bit_width = 2,
                                                last_layer_quant_type = QuantType.INT,
                                                last_layer_bit_width = 2,
                                                activation_quant_type = QuantType.INT,
                                                activation_bit_width = 2,
                                                activation_scaling_impl_type = ScalingImplType.CONST,
                                                activation_max_val = 6,
                                                chan_mult = 0.5)

        elif network_type == NetType.FINN_W2_A2_X29:
            print("Network type: ")
            print(network_type)
            self.conv_features = create_model(first_layer_quant_type = QuantType.INT,
                                                first_layer_bit_width = 2,
                                                weight_quant_type = QuantType.INT,
                                                weight_bit_width = 2,
                                                last_layer_quant_type = QuantType.INT,
                                                last_layer_bit_width = 2,
                                                activation_quant_type = QuantType.INT,
                                                activation_bit_width = 2,
                                                activation_scaling_impl_type = ScalingImplType.CONST,
                                                activation_max_val = 6,
                                                chan_mult = 0.25)

        elif network_type == NetType.FINN_W2_A2_X30:
            print("Network type: ")
            print(network_type)
            self.conv_features = create_model(first_layer_quant_type = QuantType.INT,
                                                first_layer_bit_width = 2,
                                                weight_quant_type = QuantType.INT,
                                                weight_bit_width = 2,
                                                last_layer_quant_type = QuantType.INT,
                                                last_layer_bit_width = 2,
                                                activation_quant_type = QuantType.INT,
                                                activation_bit_width = 2,
                                                activation_scaling_impl_type = ScalingImplType.CONST,
                                                activation_max_val = 6,
                                                chan_mult = 0.125)

        elif network_type == NetType.FINN_W2_A2_X31:
            print("Network type: ")
            print(network_type)
            self.conv_features = create_model(first_layer_quant_type=QuantType.INT,
                                              first_layer_bit_width=2,
                                              weight_quant_type=QuantType.INT,
                                              weight_bit_width=2,
                                              last_layer_quant_type=QuantType.INT,
                                              last_layer_bit_width=2,
                                              activation_quant_type=QuantType.INT,
                                              activation_bit_width=2,
                                              activation_scaling_impl_type=ScalingImplType.CONST,
                                              activation_max_val=6,
                                              chan_mult=0.0625)

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
