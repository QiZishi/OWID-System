import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torchvision._internally_replaced_utils import load_state_dict_from_url
from typing import Callable, Any, List
from utils.utils import load_weights_from_state_dict, fuse_conv_bn
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
from pytorch_wavelets import DWTForward
import torch.nn.functional as F
from torch.nn import init

__all__ = [
    'shuffleconvnet'
]

model_urls = {
    'shuffleconvnet': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_femto_1k_224_ema.pt',
}

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

# 定义余弦分类器
class Cos_Classifier(nn.Module):
    """ plain cosine classifier """

    def __init__(self,  in_dim=640, num_classes=10,  scale=16, bias=False):
        # in_dim是输入特征的维度，num_classes是类别数，scale是缩放因子，bias是是否使用偏置
        super(Cos_Classifier, self).__init__()
        self.in_dim = in_dim
        self.scale = scale
        self.weight = Parameter(torch.Tensor(num_classes, in_dim).cuda())
        self.bias = Parameter(torch.Tensor(num_classes).cuda(), requires_grad=bias)
        self.init_weights()

    def init_weights(self):
        self.bias.data.fill_(0.)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, **kwargs):
        device = x.device  # 获取x的设备
        ex = x / torch.norm(x.clone(), 2, 1, keepdim=True)
        ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
        out = torch.mm(ex, (self.scale * ew.t()).to(device)) + self.bias.to(device)

# ACBlock

class ACBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False,
                 use_affine=True, reduce_gamma=False, gamma_init=None ):
        super(ACBlock, self).__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size,kernel_size), stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=False,
                                         padding_mode=padding_mode)
            self.square_bn = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)


            if padding - kernel_size // 2 >= 0:
                #   Common use case. E.g., k=3, p=1 or k=5, p=2
                self.crop = 0
                #   Compared to the KxK layer, the padding of the 1xK layer and Kx1 layer should be adjust to align the sliding windows (Fig 2 in the paper)
                hor_padding = [padding - kernel_size // 2, padding]
                ver_padding = [padding, padding - kernel_size // 2]
            else:
                #   A negative "padding" (padding - kernel_size//2 < 0, which is not a common use case) is cropping.
                #   Since nn.Conv2d does not support negative padding, we implement it manually
                self.crop = kernel_size // 2 - padding
                hor_padding = [0, padding]
                ver_padding = [padding, 0]

            self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, 1),
                                      stride=stride,
                                      padding=ver_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)

            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                                      stride=stride,
                                      padding=hor_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)

            if reduce_gamma:
                self.init_gamma(1.0 / 3)

            if gamma_init is not None:
                assert not reduce_gamma
                self.init_gamma(gamma_init)


    def _fuse_bn_tensor(self, conv, bn):
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std

    def _add_to_square_kernel(self, square_kernel, asym_kernel):
        asym_h = asym_kernel.size(2)
        asym_w = asym_kernel.size(3)
        square_h = square_kernel.size(2)
        square_w = square_kernel.size(3)
        square_kernel[:, :, square_h // 2 - asym_h // 2: square_h // 2 - asym_h // 2 + asym_h,
                square_w // 2 - asym_w // 2: square_w // 2 - asym_w // 2 + asym_w] += asym_kernel

    def get_equivalent_kernel_bias(self):
        hor_k, hor_b = self._fuse_bn_tensor(self.hor_conv, self.hor_bn)
        ver_k, ver_b = self._fuse_bn_tensor(self.ver_conv, self.ver_bn)
        square_k, square_b = self._fuse_bn_tensor(self.square_conv, self.square_bn)
        self._add_to_square_kernel(square_k, hor_k)
        self._add_to_square_kernel(square_k, ver_k)
        return square_k, hor_b + ver_b + square_b


    def switch_to_deploy(self):
        deploy_k, deploy_b = self.get_equivalent_kernel_bias()
        self.deploy = True
        self.fused_conv = nn.Conv2d(in_channels=self.square_conv.in_channels, out_channels=self.square_conv.out_channels,
                                    kernel_size=self.square_conv.kernel_size, stride=self.square_conv.stride,
                                    padding=self.square_conv.padding, dilation=self.square_conv.dilation, groups=self.square_conv.groups, bias=True,
                                    padding_mode=self.square_conv.padding_mode)
        self.__delattr__('square_conv')
        self.__delattr__('square_bn')
        self.__delattr__('hor_conv')
        self.__delattr__('hor_bn')
        self.__delattr__('ver_conv')
        self.__delattr__('ver_bn')
        self.fused_conv.weight.data = deploy_k
        self.fused_conv.bias.data = deploy_b
    



    def init_gamma(self, gamma_value):
        init.constant_(self.square_bn.weight, gamma_value)
        init.constant_(self.ver_bn.weight, gamma_value)
        init.constant_(self.hor_bn.weight, gamma_value)
        print('init gamma of square, ver and hor as ', gamma_value)

    def single_init(self):
        init.constant_(self.square_bn.weight, 1.0)
        init.constant_(self.ver_bn.weight, 0.0)
        init.constant_(self.hor_bn.weight, 0.0)
        print('init gamma of square as 1, ver and hor as 0')

    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs = self.square_conv(input)
            square_outputs = self.square_bn(square_outputs)
            if self.crop > 0:
                ver_input = input[:, :, :, self.crop:-self.crop]
                hor_input = input[:, :, self.crop:-self.crop, :]
            else:
                ver_input = input
                hor_input = input
            vertical_outputs = self.ver_conv(ver_input)
            vertical_outputs = self.ver_bn(vertical_outputs)
            horizontal_outputs = self.hor_conv(hor_input)
            horizontal_outputs = self.hor_bn(horizontal_outputs)
            result = square_outputs + vertical_outputs + horizontal_outputs
            return result


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
# HWD小波变化下采样
class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            # LayerNorm(out_ch, eps=1e-6)
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True), 
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x

# 三重通道相加注意力 有一点用

class MDRAttention(nn.Module):
    def __init__(self, channel=512, reduction=16, e_lambda=1e-4,kernel_size=3):
        super().__init__()
        self.channel = channel  # 通道数
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化层
        self.gn = nn.GroupNorm(channel,channel)  # 分组归一化层
        self.silu = nn.SiLU()  # SiLU激活函数,用于最后三个注意力的融合
        # 以下四个参数用于计算通道注意力（1D)、空间注意力(2D)和全局注意力simam(3D)
        # 定义通道注意力ECA
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sconv = ACBlock(2,1,kernel_size=kernel_size,padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数
        self.e_lambda = e_lambda # simam需要的参数

    def init_weights(self):
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # 前向传播函数
        b, c, h, w = x.size()
        # 计算通道注意力(ECA,原来的是SE)
        x_channel = self.avg_pool(x)
        x_channel = x_channel.squeeze(-1).permute(0, 2, 1)
        x_channel = self.conv(x_channel)
        x_channel = self.sigmoid(x_channel)
        x_channel = x_channel.permute(0, 2, 1).unsqueeze(-1)
        x_channel = x_channel.expand_as(x)
        x_channel = x * x_channel.expand_as(x) # 通道注意力特征图

        # 计算空间注意力
        x_spatial_max, _ = torch.max(x, dim=1, keepdim=True)
        x_spatial_avg = torch.mean(x, dim=1, keepdim=True)
        x_spatial = torch.cat([x_spatial_max,x_spatial_avg],dim=1)
        x_spatial = self.sconv(x_spatial)
        x_spatial = x*self.sigmoid(x_spatial) # 空间注意力特征图

        # 计算simam
        b_simam, c_simam, h_simam, w_simam = x.size()
        n_simam = w_simam * h_simam - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n_simam + self.e_lambda)) + 0.5
        x_simam = x * self.sigmoid(y) # simam特征图

        # 将三个注意力的结果进行平均 （三个特征都不sigmoid再加效果不如通道先sigmoid，其他两个不sigmoid）
        # x_out = x+x *self.sigmoid(x_channel + x_spatial + y)  # 将三个注意力的结果进行平均
        # 对三个特征图相加取平均，然后加上残差
        x_out = x+1 / 3  *(x_channel + x_spatial + x_simam)
        return x_out


class ParNetAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.sse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.Sigmoid()
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel)
        )
        self.conv3x3 = ACBlock(in_channels=channel, out_channels=channel, kernel_size=3, padding=1)
        self.silu = nn.SiLU()

    def forward(self, x):
        b, c, _, _ = x.size()
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.sse(x) * x
        y = self.silu(x1 + x2 + x3)
        return y

    def switch_to_deploy(self):
            self.conv1x1 = nn.Sequential(
                fuse_conv_bn(self.conv1x1[0], self.conv1x1[1]),
            )






# SimAM 无参注意力，剔除无关信息
class Simam_module(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(Simam_module, self).__init__()
        self.act = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.act(y)
# 三重通道混洗注意力，三路 没用
class TripletShuffleAttention(nn.Module):
    def __init__(self, channel=512, reduction=16, G=8,e_lambda=1e-4,kernel_size=3):
        super().__init__()
        self.G = G  # 分组数
        self.channel = channel  # 通道数
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化层
        self.gn = nn.GroupNorm(channel // (3 * G), channel // (3 * G))  # 分组归一化层
        # 以下四个参数用于计算通道注意力（1D)、空间注意力(2D)和全局注意力simam(3D)
        # self.cweight = Parameter(torch.zeros(1, channel // (3 * G), 1, 1))
        # self.cbias = Parameter(torch.ones(1, channel // (3 * G), 1, 1))
        # 定义通道注意力ECA
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sweight = Parameter(torch.zeros(1, channel // (3 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (3 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数
        self.e_lambda = e_lambda # simam需要的参数

    def init_weights(self):
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    @staticmethod
    def channel_shuffle(x, groups):
        # 通道混洗函数
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, -1, h, w)
        return x

    def forward(self, x):
        # 前向传播函数
        b, c, h, w = x.size()
        x = x.view(b * self.G, -1, h, w)  # 将输入特征分组
        x_0, x_1, x_2 = x.chunk(3, dim=1)  # 将特征分割成三部分


        # 计算通道注意力(ECA,原来的是SE)
        x_channel = self.avg_pool(x_0)
        x_channel = x_channel.squeeze(-1).permute(0, 2, 1)
        x_channel = self.conv(x_channel)
        x_channel = self.sigmoid(x_channel)
        x_channel = x_channel.permute(0, 2, 1).unsqueeze(-1)
        x_channel = x_0 * x_channel.expand_as(x_0)


        # 计算空间注意力
        x_spatial = self.gn(x_1)
        x_spatial = self.sweight * x_spatial + self.sbias
        x_spatial = x_1 * self.sigmoid(x_spatial)

        # 计算simam
        b_simam, c_simam, h_simam, w_simam = x_2.size()
        n_simam = w_simam * h_simam - 1
        x_minus_mu_square = (x_2 - x_2.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n_simam + self.e_lambda)) + 0.5
        x_simam = x_2 * self.sigmoid(y)

        # 将三部分特征沿通道轴连接起来
        out = torch.cat([x_channel, x_spatial,x_simam], dim=1)
        out = out.contiguous().view(b, -1, h, w)

        # 通道混洗
        out = self.channel_shuffle(out, 3)
        return out

# ShuffleSimAttention 没用 eca通道注意力+gn空间注意力+simam全局三维注意力（先用两路eca和gn，最后simam）
class ShuffleSimAttention(nn.Module):
    def __init__(self, channel=512, reduction=16, G=8,e_lambda=1e-4,kernel_size=3):
        super().__init__()
        self.G = G  # 分组数
        self.channel = channel  # 通道数
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化层
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))  # 分组归一化层
        # 以下四个参数用于计算通道注意力（1D)、空间注意力(2D)和全局注意力simam(3D)
        # self.cweight = Parameter(torch.zeros(1, channel // (3 * G), 1, 1))
        # self.cbias = Parameter(torch.ones(1, channel // (3 * G), 1, 1))
        # 定义通道注意力ECA
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数
        self.e_lambda = e_lambda # simam需要的参数

    def init_weights(self):
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    @staticmethod
    def channel_shuffle(x, groups):
        # 通道混洗函数
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, -1, h, w)
        return x

    def forward(self, x):
        # 前向传播函数
        b, c, h, w = x.size()
        # 对进行过simam的特征进行通道注意力和空间注意力
        x = x.view(b * self.G, -1, h, w)  # 将输入特征分组
        x_0, x_1  = x.chunk(2, dim=1)  # 将特征分割成2部分


        # 计算通道注意力(ECA,原来的是SE)
        x_channel = self.avg_pool(x_0)
        x_channel = x_channel.squeeze(-1).permute(0, 2, 1)
        x_channel = self.conv(x_channel)
        x_channel = self.sigmoid(x_channel)
        x_channel = x_channel.permute(0, 2, 1).unsqueeze(-1)
        x_channel = x_0 * x_channel.expand_as(x_0)


        # 计算空间注意力
        x_spatial = self.gn(x_1)
        x_spatial = self.sweight * x_spatial + self.sbias
        x_spatial = x_1 * self.sigmoid(x_spatial)

        # 将两部分特征沿通道轴连接起来
        out = torch.cat([x_channel, x_spatial], dim=1)
        out = out.contiguous().view(b, -1, h, w)

        # 通道混洗
        x = self.channel_shuffle(out, 2)

        # 先对输出进行simam
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        x = x * self.sigmoid(y)
        return x



# 原来的shuffleattention
class ShuffleAttention(nn.Module):
    def __init__(self, channel=512, reduction=16, G=8):
        super().__init__()
        self.G = G  # 分组数
        self.channel = channel  # 通道数
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化层
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))  # 分组归一化层
        # 以下四个参数用于计算通道注意力和空间注意力
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数

    def init_weights(self):
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    @staticmethod
    def channel_shuffle(x, groups):
        # 通道混洗函数
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, -1, h, w)
        return x

    def forward(self, x):
        # 前向传播函数
        b, c, h, w = x.size()
        x = x.view(b * self.G, -1, h, w)  # 将输入特征分组
        x_0, x_1 = x.chunk(2, dim=1)  # 将特征分割成两部分

        # 计算通道注意力
        x_channel = self.avg_pool(x_0)
        x_channel = self.cweight * x_channel + self.cbias
        x_channel = x_0 * self.sigmoid(x_channel)

        # 计算空间注意力
        x_spatial = self.gn(x_1)
        x_spatial = self.sweight * x_spatial + self.sbias
        x_spatial = x_1 * self.sigmoid(x_spatial)

        # 将两部分特征沿通道轴连接起来
        out = torch.cat([x_channel, x_spatial], dim=1)
        out = out.contiguous().view(b, -1, h, w)

        # 通道混洗
        out = self.channel_shuffle(out, 2)
        return out


# 先sim后两路通道注意力和空间注意力融合 没用
class SimDoubleAttention(nn.Module):
    def __init__(self, channel=512, reduction=16, e_lambda=1e-4,kernel_size=3):
        super().__init__()
        self.channel = channel  # 通道数
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化层
        self.gn = nn.GroupNorm(channel,channel)  # 分组归一化层
        self.silu = nn.SiLU()  # SiLU激活函数,用于最后三个注意力的融合
        # 以下四个参数用于计算通道注意力（1D)、空间注意力(2D)和全局注意力simam(3D)
        # 定义通道注意力ECA
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sweight = Parameter(torch.zeros(1, channel, 1, 1))
        self.sbias = Parameter(torch.ones(1, channel, 1, 1))
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数
        self.e_lambda = e_lambda # simam需要的参数

    def init_weights(self):
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # 计算simam
        b_simam, c_simam, h_simam, w_simam = x.size()
        n_simam = w_simam * h_simam - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n_simam + self.e_lambda)) + 0.5
        x = x * self.sigmoid(y)
        # 计算通道注意力(ECA,原来的是SE)
        x_channel = self.avg_pool(x)
        x_channel = x_channel.squeeze(-1).permute(0, 2, 1)
        x_channel = self.conv(x_channel)
        x_channel = self.sigmoid(x_channel)
        x_channel = x_channel.permute(0, 2, 1).unsqueeze(-1)
        x_channel = x * x_channel.expand_as(x)

        # 计算空间注意力
        x_spatial = self.gn(x)
        x_spatial = self.sweight * x_spatial + self.sbias
        x_spatial = x * self.sigmoid(x_spatial)

        # 将两个注意力的结果进行平均
        x_out = self.silu(x_channel + x_spatial)  # 将两个注意力的结果进行平均

        return x_out




def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    # 获取输入张量的大小
    batchsize, num_channels, height, width = x.size()
    # 计算每个组的通道数
    channels_per_group = num_channels // groups

    # 将输入张量的形状改变为(batchsize, groups, channels_per_group, height, width)
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # 交换第二和第三维度
    x = torch.transpose(x, 1, 2).contiguous()

    # 将张量的形状改变为(batchsize, -1, height, width)
    x = x.view(batchsize, -1, height, width)

    # 返回新的张量
    return x

class InvertedResidual(nn.Module):
    def __init__(
        self,
        dim: int,  # 通道数
        mode: int,  # 模式
        drop_path=0.
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.mode = mode

        # 计算分支特征数，即通道数的一半
        branch_features = dim // 2

        # 如果模式mode为2，则branch1为3*3，d=2的深度空洞卷积+1*1点卷积
        # 模式mode为2时，通道为输入通道一半；模式mode为1时，通道为dim
        if self.mode > 1:
            self.branch1 = nn.Sequential(
                # 3*3，d=2的深度空洞卷积
                self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=1, padding=2, dilation=2),
                nn.BatchNorm2d(branch_features),  # 批量归一化

                nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),  # 1x1卷积
                nn.BatchNorm2d(branch_features),  # 批量归一化
                nn.ReLU(inplace=True),  # ReLU激活函数
            )
            branch2_dim = branch_features

        else:
            # 如果模式mode为1，branch1为空
            self.branch1 = nn.Sequential()
            branch2_dim = dim
        # 模式mode为2时，通道为输入通道一半；模式mode为1时，通道为输入通道
        self.branch2 = nn.Sequential(
                # 3*3深度卷积
                self.depthwise_conv(branch2_dim, branch2_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(branch2_dim),
            )
        self.pwconv1 = nn.Linear(branch2_dim, 4 * branch2_dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.ReLU(inplace=True)
        self.grn = GRN(4 * branch2_dim)
        self.pwconv2 = nn.Linear(4 * branch2_dim, branch2_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    @staticmethod
    def depthwise_conv(
        i: int,
        o: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        dilation: int = 1,
    ) -> nn.Conv2d:
        return nn.Conv2d(i, o, kernel_size, stride, padding, groups=i, bias=bias, dilation=dilation)


    @staticmethod
    def acblock(
        i: int,
        o: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ) -> ACBlock:
        return ACBlock(i, o, kernel_size, stride, padding, groups=i, dilation=dilation)

    def forward(self, x: Tensor) -> Tensor:
        # 模式为2时，将输入特征分为两部分，分别经过branch1和branch2，然后通道cat再混洗
        if self.mode == 2:
            x1, x2 = x.chunk(2, dim=1)
            x1 = self.branch1(x1)
            x2 = self.branch2(x2)
            x2 = x2.permute(0, 2, 3, 1)
            x2 = self.pwconv1(x2)
            x2 = self.act(x2)
            x2 = self.grn(x2)
            x2 = self.pwconv2(x2)
            x2 = x2.permute(0, 3, 1, 2)
            x2 = self.drop_path(x2)
            out = torch.cat((x1, x2), dim=1)
            # 通道混洗
            out = channel_shuffle(out, 2)
        else:
            # 模式为1时，倒转瓶颈残差，直接add
            input = x
            x = self.branch2(x)
            x = x.permute(0, 2, 3, 1)
            x = self.pwconv1(x)
            x = self.act(x)
            x = self.grn(x)
            x = self.pwconv2(x)
            x = x.permute(0, 3, 1, 2)
            x = self.drop_path(x)
            out = input + x

        return out

    def switch_to_deploy(self):
        if len(self.branch1) > 0:
            self.branch1 = nn.Sequential(
                fuse_conv_bn(self.branch1[0], self.branch1[1]),
                fuse_conv_bn(self.branch1[2], self.branch1[3]),
                self.branch1[4]

            )

        self.branch2 = nn.Sequential(
                fuse_conv_bn(self.branch2[0], self.branch2[1]),
            )

class ShuffleConvNet(nn.Module):
    def __init__(
        self,
        stages_repeats: List[int],
        stages_out_channels: List[int],
        num_classes: int = 1000,
        drop_path_rate=0.,
        inverted_residual: Callable[..., nn.Module] = InvertedResidual
    ) -> None:
        super(ShuffleConvNet, self).__init__()

        if len(stages_repeats) != 4: # 改成4stage
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels
        # 将下采样模块独立出来
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        input_channels = 3
        output_channels = self._stage_out_channels[0]
        stem = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=7, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(output_channels),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    nn.Conv2d(self._stage_out_channels[i], self._stage_out_channels[i+1], kernel_size=2, stride=2),
                    nn.BatchNorm2d(self._stage_out_channels[i+1]),
            )
            self.downsample_layers.append(downsample_layer)
        # 主干模块stage
        self.stages = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(stages_repeats))]
        cur = 0
        for i in range(4): # 共有4个stage
            stage = nn.Sequential(
                *[inverted_residual(dim=self._stage_out_channels[i], mode=2 if j == 0 else 1, drop_path=dp_rates[cur + j]) for j in range(stages_repeats[i])]
            )

            # 这个地方加注意力
            if i != 3:
                # 不给最后一个stage加注意力
                stage.append(MDRAttention(self._stage_out_channels[i]))


            self.stages.append(stage)
            cur += stages_repeats[i]
        self.conv5 = nn.Sequential(
            nn.Conv2d(self._stage_out_channels[-2], self._stage_out_channels[-1], 1, 1, 0, bias=False),
            nn.BatchNorm2d(self._stage_out_channels[-1]),
            nn.ReLU(inplace=True),
        )

        # 使用余弦分类器，替换全连接层，output_channels是最后一个卷积层的输出通道数
        self.fc = Cos_Classifier(self._stage_out_channels[-1],num_classes)
        # 初始化权重
        self.apply(self._init_weights)

    # 部署模式轻量化conv和bn融合
    def switch_to_deploy(self):
        for i in range(4):
            self.downsample_layers[i] = nn.Sequential(
                fuse_conv_bn(self.downsample_layers[i][0], self.downsample_layers[i][1]),
                )

        self.conv5 = nn.Sequential(
            fuse_conv_bn(self.conv5[0], self.conv5[1]),
            self.conv5[2]
        )
    # 初始化权重
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if m.weight is not None:
                nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    def _forward_impl(self, x: Tensor, need_fea=False) -> Tensor:
        if need_fea:
            features, features_fc = self.forward_features(x, need_fea)
            return features, features_fc, self.fc(features_fc)

        else:
            # See note [TorchScript super()]
            x = self.forward_features(x)
            x = self.fc(x)
            return x

    def forward(self, x: Tensor, need_fea=False) -> Tensor:
        return self._forward_impl(x, need_fea)
    
    def forward_features(self, x, need_fea=False):
        if need_fea:
            features = []
            for i in range(4):
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)
            x = self.conv5(x)
            return features, x.mean([2, 3])
        else:
            for i in range(4):
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)
            x = self.conv5(x)
            x = x.mean([2, 3])  # globalpool
            return x


    def cam_layer(self):

        # return self.stages[-1]
        return self.conv5



def _shuffleconvnet(arch: str, pretrained: bool, progress: bool, *args: Any, **kwargs: Any) -> ShuffleConvNet:
    model = ShuffleConvNet(*args, **kwargs)

    if pretrained:
        # model_url = 'runs/shuffleconvnet_oil_aug_noattention/best.pt'
        model_url = 'run_data/checkpoint/moco_best.pth.tar'
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            # state_dict = load_state_dict_from_url(model_url, progress=progress)
            # 加载预训练模型
            ckpt = torch.load(model_url)
            # model_ckpt = ckpt['model']
            # state_dict = model_ckpt.state_dict()
            # model = load_weights_from_state_dict(model, state_dict)
            # 从ckpt中提取'best_acc'并打印
            print(f"最优准确率best_acc: {ckpt['best_acc']:.3f}\n")
            # 重命名权重的名字
            state_dict = ckpt['state_dict']
            for k in list(state_dict.keys()):
                # 保留 encoder_q 层的权重，但不包括嵌入层
                if k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):
                    # 删除前缀
                    state_dict[k[len("encoder_q."):]] = state_dict[k]
                # 删除重命名或未使用的权重
                del state_dict[k]
            # 加载状态字典
            model = load_weights_from_state_dict(model, state_dict)

    return model


def shuffleconvnet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ShuffleConvNet:
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shuffleconvnet('shuffleconvnet', pretrained, progress,
                         [2, 2, 6, 2], [24, 48, 96, 192, 1024], **kwargs)
            # 768



if __name__ == '__main__':
    inputs = torch.rand((1, 3, 224, 224))
    model = shuffleconvnet(pretrained=False)
    model.eval()
    out = model(inputs)
    print('out shape:{}'.format(out.size()))
    feas, fea_fc, out = model(inputs, True)
    for idx, fea in enumerate(feas):
        print('feature {} shape:{}'.format(idx + 1, fea.size()))
    print('fc shape:{}'.format(fea_fc.size()))
    print('out shape:{}'.format(out.size()))