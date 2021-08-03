#
# mDKL
#
# Copyright (c) Siemens AG, 2021
# Authors:
# Zhiliang Wu <zhiliang.wu@siemens.com>
# License-Identifier: MIT

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from torchinfo import summary


################################################################################
# the code is based on the open-source implementation of resnet 18
# https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet18-mnist.ipynb

def conv3x3(in_planes, out_planes, stride=1, output_padding=0):
    """3x3 convolution transpose with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3,
                              stride=stride,
                              padding=1, output_padding=output_padding,
                              bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        if stride == 1:
            self.conv2 = conv3x3(planes, planes, stride)
        else:
            self.conv2 = conv3x3(planes, planes, stride, output_padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        #         print('hi')
        #         print(out.size())
        #         print(residual.size())
        #         print('hi')

        out += residual
        out = self.relu(out)

        return out


class InvResNet(nn.Module):

    def __init__(self, block, layers, n_input_features, conv_config,
                 normalize=False):
        # kernel_size=14, stride=2, padding=0,
        self.inplanes = n_input_features

        super(InvResNet, self).__init__()

        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0])

        # the configuration here has to be computed manually here
        self.conv1 = nn.ConvTranspose2d(64, 3, **conv_config,
                                        bias=False)
        # self.bn1 = nn.BatchNorm2d(3)
        # self.relu = nn.ReLU(inplace=True)
        self.normalize = normalize


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** .5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes * block.expansion,
                                   kernel_size=1, stride=stride,
                                   output_padding=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)

        x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
        if self.normalize:
            x = torch.sigmoid(x)

        return x

################################################################################
# The following code is based on the densenet121 tutorial from rasbt
# https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-densenet121-mnist.ipynb


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate,
                 memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.memory_efficient and any(
                prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.ConvTranspose2d(num_input_features,
                                                   num_output_features,
                                                   kernel_size=3, stride=2,
                                                   padding=1, output_padding=1))


class InvDenseNet121(nn.Module):

    def __init__(self, growth_rate=32, block_config=(16, 8, 4, 2),
                 num_init_featuremaps=512, bn_size=4, drop_rate=0,
                 memory_efficient=False):

        super(InvDenseNet121, self).__init__()

        # First convolution
        in_channels = 1024

        self.features = nn.Sequential(OrderedDict([
            # the commented conv transpose requires 25M parameters
            # ('conv0', nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels//2,
            #                     kernel_size=7, stride=1,
            #                     padding=0, bias=False)), # bias is redundant when using batchnorm
            ('conv0', nn.Conv2d(in_channels, in_channels//2, kernel_size=1,
                                stride=1,
                                bias=False)),
            ('norm0', nn.BatchNorm2d(num_features=in_channels//2)),
            ('relu0', nn.ReLU(inplace=True)),
        ]))

        # Each denseblock
        num_features = num_init_featuremaps
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            # if i != len(block_config) - 1:
            trans = _Transition(num_input_features=num_features,
                                num_output_features=num_features // 4)
            self.features.add_module('transition%d' % (i + 1), trans)
            num_features = num_features // 4

        # the final conv transpose, manual computation for the shape required
        self.deconv_final = nn.ConvTranspose2d(32, 3, kernel_size=34, stride=2,
                                               padding=0)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x = F.interpolate(x, scale_factor=7, mode='nearest')
        x = self.features(x)
        x = self.deconv_final(x)

        return x

################################################################################

class ConvolutionalAutoencoder(torch.nn.Module):

    def __init__(self, encoder, decoder, n_features=512, unpool_scale=None):
        super(ConvolutionalAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.n_features = n_features
        if unpool_scale:
            self.unpool_scale = unpool_scale
        else:
            self.unpool_scale = None

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), self.n_features, 1, 1)
        if self.unpool_scale:
            x = F.interpolate(x, scale_factor=self.unpool_scale, mode='nearest')
        x = self.decoder(x)

        return x


if __name__ == '__main__':
    dc = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    conv_config = {'kernel_size': 14, 'stride': 2, 'padding': 0}
    # model = InvResNet(block=BasicBlock, layers=[2, 2, 2, 2],
    #                   n_input_features=512, conv_config=conv_config)
    # model.to(dc)
    #
    # _ = summary(model, (2, 512, 1, 1), col_names=('input_size', 'output_size',
    #                                               'num_params', 'kernel_size'))

    model = InvDenseNet121()
    model.to(dc)

    _ = summary(model, (2, 1024, 1, 1), col_names=('input_size', 'output_size',
                                                   'num_params', 'kernel_size'))
