from .decode_head import BaseDecodeHead
from ..builder import HEADS

import warnings
from torch.nn import init
import torch
import torch.nn as nn
from torch import nn
import torch.nn.functional as F

__all__ = ['EIU_Net']


class reshaped(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(reshaped, self).__init__()
        self.reshape = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                     nn.Upsample(size=scale_factor, mode='bilinear'), )

    def forward(self, input):
        return self.reshape(input)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


# # SE block add to U-net
def conv3x3(in_planes, out_planes, stride=1, bias=False, group=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=group, bias=bias)


# # CBAM Convolutional block attention module
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class SoftPooling2D(torch.nn.Module):
    def __init__(self, kernel_size, strides=None, padding=0, ceil_mode=False, count_include_pad=True,
                 divisor_override=None):
        super(SoftPooling2D, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size, strides, padding, ceil_mode, count_include_pad, divisor_override)

    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp * x)
        return x / x_exp_pool


def downsample_soft():
    return SoftPooling2D(2, 2)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max', 'sp']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'sp':
                sf_pool_f = SoftPooling2D((x.size(2), x.size(3)), (x.size(2), x.size(3)))
                sf_pool = sf_pool_f(x)
                channel_att_raw = self.mlp(sf_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        # scalecoe = F.sigmoid(channel_att_sum)
        channel_att_sum = channel_att_sum.reshape(channel_att_sum.shape[0], 4, 4)
        avg_weight = torch.mean(channel_att_sum, dim=2).unsqueeze(2)
        avg_weight = avg_weight.expand(channel_att_sum.shape[0], 4, 4).reshape(channel_att_sum.shape[0], 16)
        scale = F.sigmoid(avg_weight).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x * scale, scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        # spa_scale = scale.expand_as(x)
        # print(spa_scale.shape)
        return x * scale, scale


class SpatialAtten(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, stride=1):
        super(SpatialAtten, self).__init__()
        self.conv1 = BasicConv(in_size, out_size, kernel_size, stride=stride,
                               padding=(kernel_size - 1) // 2, relu=True)
        self.conv2 = BasicConv(out_size, out_size, kernel_size=1, stride=stride,
                               padding=0, relu=True, bn=False)

    def forward(self, x):
        residual = x
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        spatial_att = F.sigmoid(x_out).unsqueeze(4).permute(0, 1, 4, 2, 3)
        spatial_att = spatial_att.expand(spatial_att.shape[0], 4, 4, spatial_att.shape[3],
                                         spatial_att.shape[4]).reshape(
            spatial_att.shape[0], 16, spatial_att.shape[3], spatial_att.shape[4])
        x_out = residual * spatial_att

        x_out += residual

        return x_out, spatial_att


class Scale_atten_block_softpool(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['lp', 'sp'], no_spatial=False):
        super(Scale_atten_block_softpool, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialAtten(gate_channels, gate_channels // reduction_ratio)

    def forward(self, x):
        x_out, ca_atten = self.ChannelGate(x)
        if not self.no_spatial:
            x_out, sa_atten = self.SpatialGate(x_out)

        return x_out, ca_atten, sa_atten


class scale_atten_convblock_softpool(nn.Module):
    def __init__(self, in_size, out_size, stride=1, downsample=None, use_cbam=True, no_spatial=False, drop_out=False):
        super(scale_atten_convblock_softpool, self).__init__()
        # if stride != 1 or in_size != out_size:
        #     downsample = nn.Sequential(
        #         nn.Conv2d(in_size, out_size,
        #                   kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(out_size),
        #     )
        self.downsample = downsample
        self.stride = stride
        self.no_spatial = no_spatial
        self.dropout = drop_out

        self.relu = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(in_size, out_size)
        self.bn3 = nn.BatchNorm2d(out_size)

        if use_cbam:
            self.cbam = Scale_atten_block_softpool(in_size, reduction_ratio=4, no_spatial=self.no_spatial)  # out_size
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out, scale_c_atten, scale_s_atten = self.cbam(x)

            # scale_c_atten = nn.Sigmoid()(scale_c_atten)
            # scale_s_atten = nn.Sigmoid()(scale_s_atten)
            # scale_atten = channel_atten_c * spatial_atten_s

        # scale_max = torch.argmax(scale_atten, dim=1, keepdim=True)
        # scale_max_soft = get_soft_label(input_tensor=scale_max, num_class=8)
        # scale_max_soft = scale_max_soft.permute(0, 3, 1, 2)
        # scale_atten_soft = scale_atten * scale_max_soft

        out += residual
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.dropout:
            out = nn.Dropout2d(0.5)(out)

        return out


def xy_avg_max(x_y: list):
    res_avg = []
    res_max = []
    for idx in x_y:
        avg_pool = F.adaptive_avg_pool2d(idx, 1)
        max_pool = F.adaptive_max_pool2d(idx, 1)
        res_avg.append(avg_pool)
        res_max.append(max_pool)
    res = res_avg + res_max
    return torch.cat(res, dim=1)


class ChannelAtt(nn.Module):
    def __init__(self, channels):
        super(ChannelAtt, self).__init__()
        self.channels = channels

        self.bn2 = nn.BatchNorm2d(self.channels, momentum=0.9, affine=True)

    def forward(self, x):
        residual = x

        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = torch.sigmoid(x) * residual  #

        return x


class MultiScaleAttention(nn.Module):
    def __init__(self, x_ch, y_ch, out_ch, resize_mode='bilinear'):
        super(MultiScaleAttention, self).__init__()
        self.conv_x = nn.Sequential(
            nn.Conv2d(x_ch, y_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(y_ch, momentum=0.9),
            nn.ReLU(inplace=True),
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(y_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=0.9),
            nn.ReLU(inplace=True),
        )

        self.resize_mode = resize_mode

        self.conv_xy = nn.Sequential(
            nn.Conv2d(4 * y_ch, y_ch // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(y_ch // 2, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(y_ch // 2, y_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(y_ch, momentum=0.9),
        )
        self.channel_att = ChannelAtt(channels=y_ch)

    def prepare(self, x, y):
        x = self.prepare_x(x)
        y = self.prepare_y(x, y)
        return x, y

    def prepare_x(self, x):
        x = self.conv_x(x)
        return x

    def prepare_y(self, x, y):
        y_expand = F.interpolate(y, x.shape[2:], mode=self.resize_mode, align_corners=True)
        return y_expand

    def fuse(self, x, y):
        attention = xy_avg_max([x, y])
        attention = self.channel_att(self.conv_xy(attention))

        out = x * attention + y * (1 - attention)
        # out = self.conv_out(out)
        return out

    def forward(self, x, y):
        x, y = self.prepare(x, y)
        out = self.fuse(x, y)
        return out


class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class PSAModule(nn.Module):

    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(PSAModule, self).__init__()
        self.conv_1 = conv(inplans, planes // 4, kernel_size=conv_kernels[0], padding=conv_kernels[0] // 2,
                           stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, planes // 4, kernel_size=conv_kernels[1], padding=conv_kernels[1] // 2,
                           stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, planes // 4, kernel_size=conv_kernels[2], padding=conv_kernels[2] // 2,
                           stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, planes // 4, kernel_size=conv_kernels[3], padding=conv_kernels[3] // 2,
                           stride=stride, groups=conv_groups[3])
        self.se = SEWeightModule(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out


class EPSABlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, conv_kernels=[3, 5, 7, 9],
                 conv_groups=[1, 4, 8, 16]):
        super(EPSABlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = PSAModule(planes, planes, stride=stride, conv_kernels=conv_kernels, conv_groups=conv_groups)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.shortcut = nn.Sequential(
            conv1x1(inplanes, planes * self.expansion),
            nn.BatchNorm2d(planes * self.expansion)
        )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class SoftPooling2D(torch.nn.Module):
    def __init__(self, kernel_size, strides=None, padding=0, ceil_mode=False, count_include_pad=True,
                 divisor_override=None):
        super(SoftPooling2D, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size, strides, padding, ceil_mode, count_include_pad, divisor_override)

    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp * x)
        return x / x_exp_pool


def downsample_soft():
    return SoftPooling2D(2, 2)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 512
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6()))
        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


def Conv3x3BNReLU(in_channels, out_channels, stride, groups):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,
                  groups=groups),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def Conv1x1BNReLU(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def Conv1x1BN(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
        nn.BatchNorm2d(out_channels)
    )


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion_factor=6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        mid_channels = (in_channels * expansion_factor)

        self.bottleneck = nn.Sequential(
            Conv1x1BNReLU(in_channels, mid_channels),
            Conv3x3BNReLU(mid_channels, mid_channels, stride, groups=mid_channels),
            Conv1x1BN(mid_channels, out_channels)
        )

        if self.stride == 1:
            self.shortcut = Conv1x1BN(in_channels, out_channels)

    def forward(self, x):
        out = self.bottleneck(x)
        out = (out + self.shortcut(x)) if self.stride == 1 else out
        return out


# initalize the module
def init_weights(net, init_type='normal'):
    # print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


# compute model params
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


warnings.filterwarnings('ignore')


class ResEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = out + residual
        out = self.relu(out)
        return out


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UP(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UP, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.up(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


@HEADS.register_module()
class EIU_Net(BaseDecodeHead):
    def __init__(self, n_channels, n_classes, **kwargs):
        super(EIU_Net, self).__init__(**kwargs)
        self.n_channels = n_channels
        self.n_classes = n_classes

        filters = [32, 64, 128, 256, 512]

        self.enc_input = ResEncoder(self.n_channels, filters[0])
        self.encoder_1 = InvertedResidual(filters[0], filters[1])

        self.encoder_2 = InvertedResidual(filters[1], filters[2])

        self.encoder_3 = InvertedResidual(filters[2], filters[3])

        self.encoder_4 = EPSABlock(filters[3], 128)
        self.downsample = downsample_soft()
        self.aspp = ASPP(filters[4], [6, 12, 18])

        self.decoder_4 = UP(filters[4], filters[3])
        self.double_conv_4 = DoubleConv(filters[4], filters[3])
        self.decoder_3 = UP(filters[3], filters[2])
        self.double_conv_3 = DoubleConv(filters[3], filters[2])
        self.decoder_2 = UP(filters[2], filters[1])
        self.double_conv_2 = DoubleConv(filters[2], filters[1])
        self.decoder_1 = UP(filters[1], filters[0])
        self.double_conv_1 = DoubleConv(filters[1], filters[0])

        self.reshape_4 = reshaped(in_size=256, out_size=4, scale_factor=(224, 320))
        self.reshape_3 = reshaped(in_size=128, out_size=4, scale_factor=(224, 320))
        self.reshape_2 = reshaped(in_size=64, out_size=4, scale_factor=(224, 320))
        self.reshape_1 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1)
        self.scale_att = scale_atten_convblock_softpool(in_size=16, out_size=4)
        # self.scale_att = scale_attention(16, 4)

        self.final = OutConv(4, self.n_classes)

        self.mul_scale_att_1 = MultiScaleAttention(32, 64, 64)
        self.mul_scale_att_2 = MultiScaleAttention(64, 128, 128)
        self.mul_scale_att_3 = MultiScaleAttention(128, 256, 256)

        # self.out = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

        # initialize_weights(self)

    def forward(self, x):
        enc_input = self.enc_input(x)  # 32 [-1, 32, 224, 320]
        down1 = self.downsample(enc_input)  # [-1, 32, 112, 160]
        enc_1 = self.encoder_1(down1)  # 64 [-1, 64, 112, 160]
        mid_attention_1 = self.downsample(self.mul_scale_att_1(enc_input, enc_1))
        down2 = self.downsample(enc_1)  # [-1, 64, 56, 80]
        enc_2 = self.encoder_2(down2)  # 128  [-1, 128, 56, 80]
        mid_attention_2 = self.downsample(self.mul_scale_att_2(enc_1, enc_2))
        down3 = self.downsample(enc_2)  # [-1, 128, 28, 40]
        enc_3 = self.encoder_3(down3)  # 256  [-1, 256, 28, 40]
        mid_attention_3 = self.downsample(self.mul_scale_att_3(enc_2, enc_3))
        down4 = self.downsample(enc_3)  # [-1, 256, 28, 40]
        enc_4 = self.encoder_4(down4)  # 512 [-1, 512, 14, 20]
        enc_4 = self.aspp(enc_4)  # ASPP [-1, 512, 14, 20]
        up4 = self.decoder_4(enc_4)  # 256
        concat_4 = torch.cat((mid_attention_3, up4), dim=1)
        up4 = self.double_conv_4(concat_4)
        up3 = self.decoder_3(up4)  # 128
        concat_3 = torch.cat((mid_attention_2, up3), dim=1)
        up3 = self.double_conv_3(concat_3)
        up2 = self.decoder_2(up3)  # 64
        concat_2 = torch.cat((mid_attention_1, up2), dim=1)
        up2 = self.double_conv_2(concat_2)
        up1 = self.decoder_1(up2)  # 32
        concat_1 = torch.cat((enc_input, up1), dim=1)
        up1 = self.double_conv_1(concat_1)

        dsv4 = self.reshape_4(up4)  # [16, 4, 224, 320]
        dsv3 = self.reshape_3(up3)
        dsv2 = self.reshape_2(up2)
        dsv1 = self.reshape_1(up1)
        dsv_cat = torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1)  # [16, 16, 224, 320]
        out = self.scale_att(dsv_cat)  # [16, 4, 224, 300]

        final = self.final(out)  # 2

        # out = self.out(final)

        return final
