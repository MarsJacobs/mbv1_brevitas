"""
Modified from: https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/models

MIT License

Copyright (c) 2019 Xilinx, Inc (Alessandro Pappalardo)
Copyright (c) 2018 Oleg Sémery

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


__all__ = ['quant_mobilenet_v1_cifar10']

from torch import nn
from torch.nn import Sequential
import torch.nn.functional as F

from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantAvgPool2d
from brevitas.quant import IntBias
from brevitas.quant.scaled_int import Int8ActPerTensorFloat

from .common import CommonIntActQuant, CommonUintActQuant
from .common import CommonIntWeightPerChannelQuant, CommonIntWeightPerTensorQuant, LSQ_weight_quant_2bits, PACT_weight_quant_2bits, PACT_activation_quant_2bits, CommonInputQuant

FIRST_LAYER_BIT_WIDTH = 8


class DwsConvBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            stride,
            bit_width,
            pw_activation_scaling_per_channel=False):
        super(DwsConvBlock, self).__init__()
        self.dw_conv = ConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            groups=in_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            weight_bit_width=bit_width,
            act_bit_width=bit_width)
        self.pw_conv = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            weight_bit_width=bit_width,
            act_bit_width=bit_width,
            activation_scaling_per_channel=pw_activation_scaling_per_channel)

    def forward(self, x):
        
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class ConvBlock(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            weight_bit_width,
            act_bit_width,
            stride=1,
            padding=0,
            groups=1,
            bn_eps=1e-5,
            activation_scaling_per_channel=False,
            first_block = False):
        super(ConvBlock, self).__init__()
        
        
        if first_block:
            self.conv = QuantConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
            weight_quant=LSQ_weight_quant_2bits,
            weight_bit_width=weight_bit_width
            )
        
        else:
            self.conv = QuantConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
            weight_quant=LSQ_weight_quant_2bits,
            weight_bit_width=weight_bit_width)
        
        
        self.bn = nn.BatchNorm2d(num_features=out_channels, eps=bn_eps)
        self.activation = QuantReLU(
            act_quant=PACT_activation_quant_2bits,
            bit_width=act_bit_width,
            return_quant_tensor=True)
        
        
        


    def forward(self, x):
        
        x = self.conv(x)
        
        x = self.bn(x)
        x_input = x
        x = self.activation(x)
        import pdb; pdb.set_trace()
        #x = F.relu(x)
        return x


class MobileNet(nn.Module):

    def __init__(
            self,
            channels,
            first_stage_stride,
            bit_width,
            in_channels=3,
            num_classes=10):
        super(MobileNet, self).__init__()
        #init_block_channels = channels[0][0]

        self.features = Sequential()
        
        init_block = ConvBlock(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            weight_bit_width=FIRST_LAYER_BIT_WIDTH,
            activation_scaling_per_channel=True,
            act_bit_width=bit_width,
            first_block=True)
        self.features.add_module('init_block', init_block)
        in_channels = 32

        for i, channels_per_stage in enumerate(channels[1:]):
            stage = Sequential()
            pw_activation_scaling_per_channel = i < len(channels[1:]) - 1
            
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and ((i != 0) or first_stage_stride) else 1
                mod = DwsConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    bit_width=bit_width,
                    pw_activation_scaling_per_channel=pw_activation_scaling_per_channel)
                stage.add_module('unit{}'.format(j + 1), mod)
                in_channels = out_channels
            self.features.add_module('stage{}'.format(i + 1), stage)
        self.final_pool = QuantAvgPool2d(kernel_size=2, stride=None, bit_width=bit_width)
        self.output = QuantLinear(
            in_channels, num_classes,
            bias=True,
            bias_quant=IntBias,
            weight_quant=CommonIntWeightPerTensorQuant,
            weight_bit_width=bit_width)
        
        #self.linear = nn.Linear(1024, 10)
        

    def forward(self, x):
        x = self.features(x)
        x = self.final_pool(x)
        #x = F.avg_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        out = self.output(x)
        #out = self.linear(x)
        return out


def quant_mobilenet_v1_cifar10(cfg):

    #channels = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]
    channels = [[32], [64], [128, 128], [256, 256], [512, 512, 512, 512, 512, 512], [1024, 1024]]
    first_stage_stride = False
    width_scale = float(cfg.get('MODEL', 'WIDTH_SCALE'))
    bit_width = cfg.getint('QUANT', 'BIT_WIDTH')
    
    # if width_scale != 1.0:
    #     channels = [[int(cij * width_scale) for cij in ci] for ci in channels]

    net = MobileNet(
        channels=channels,
        first_stage_stride=first_stage_stride,
        bit_width=bit_width)

    return net



