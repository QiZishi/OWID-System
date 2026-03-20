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
    'repshufflenet'
]

model_urls = {
    'repshufflenet': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
}

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
        ex = x / torch.norm(x.clone(), 2, 1, keepdim=True)
        ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
        out = torch.mm(ex, self.scale * ew.t()) + self.bias
        return out

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

# shuffleattention eca通道注意力+gn空间注意力+simam全局三维注意力（先simam，然后用两路eca和gn）
class ShuffleAttention(nn.Module):
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
        # 先对输入进行simam
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        x = x * self.sigmoid(y)
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
        out = self.channel_shuffle(out, 2)
        return out
# 原来的shuffleattention
class aShuffleAttention(nn.Module):
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
        inp: int,  # 输入通道数
        oup: int,  # 输出通道数
        stride: int  # 步长
    ) -> None:
        super(InvertedResidual, self).__init__()

        # 检查步长是否在1到3之间，如果不在则抛出错误
        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        # 计算分支特征数，即输出通道数的一半
        branch_features = oup // 2
        # 断言步长不为1或输入通道数等于分支特征数的两倍
        assert (self.stride != 1) or (inp == branch_features << 1)

        # 如果步长大于1，定义branch1为一系列卷积、批量归一化和ReLU激活函数
        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),  # 深度卷积
                nn.BatchNorm2d(inp),  # 批量归一化
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),  # 1x1卷积
                nn.BatchNorm2d(branch_features),  # 批量归一化
                nn.ReLU(inplace=True),  # ReLU激活函数
            )
        else:
            # 如果步长为1，branch1为空
            self.branch1 = nn.Sequential()
        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(
        i: int,
        o: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False
    ) -> nn.Conv2d:
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

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
            self.branch2[2],
            fuse_conv_bn(self.branch2[3], self.branch2[4]),
            fuse_conv_bn(self.branch2[5], self.branch2[6]),
            self.branch2[7]
        )

class ShuffleNetV2(nn.Module):
    def __init__(
        self,
        stages_repeats: List[int],
        stages_out_channels: List[int],
        num_classes: int = 1000,
        inverted_residual: Callable[..., nn.Module] = InvertedResidual
    ) -> None:
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.hwd_down_sample = Down_wt(input_channels, input_channels)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Static annotations for mypy
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential
        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            # 这个地方插入注意力
            # 如果name不是最后一个satge，则插入注意力
            # if name != 'stage4':
            # seq.append(ShuffleAttention(output_channels))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        # self.fc = nn.Linear(output_channels, num_classes)
        # 使用余弦分类器，替换全连接层，output_channels是最后一个卷积层的输出通道数
        self.fc = Cos_Classifier(output_channels,num_classes)

    def switch_to_deploy(self):
        self.conv1 = nn.Sequential(
            fuse_conv_bn(self.conv1[0], self.conv1[1]),
            self.conv1[2]
        )
        self.conv5 = nn.Sequential(
            fuse_conv_bn(self.conv5[0], self.conv5[1]),
            self.conv5[2]
        )

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
        x = self.conv1(x)
        # x = self.maxpool(x)
        x = self.hwd_down_sample(x) # 这个相应的设置代码self也注释了
        if need_fea:
            x2 = self.stage2(x)
            x3 = self.stage3(x2)
            x4 = self.stage4(x3)
            x4 = self.conv5(x4)
            return [x, x2, x3, x4], x4.mean([2, 3])
        else:
            # See note [TorchScript super()]
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
            x = self.conv5(x)
            x = x.mean([2, 3])  # globalpool
            return x
    
    def cam_layer(self):
        return self.stage4


def _shufflenetv2(arch: str, pretrained: bool, progress: bool, *args: Any, **kwargs: Any) -> ShuffleNetV2:
    model = ShuffleNetV2(*args, **kwargs)

    if pretrained:
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            model = load_weights_from_state_dict(model, state_dict)
    return model


def repshufflenet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ShuffleNetV2:
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('repshufflenet', pretrained, progress,
                         [4, 8, 4], [24, 48, 96, 192, 1024], **kwargs)




if __name__ == '__main__':
    inputs = torch.rand((1, 3, 224, 224))
    model = repshufflenet(pretrained=True)
    model.eval()
    out = model(inputs)
    print('out shape:{}'.format(out.size()))
    feas, fea_fc, out = model(inputs, True)
    for idx, fea in enumerate(feas):
        print('feature {} shape:{}'.format(idx + 1, fea.size()))
    print('fc shape:{}'.format(fea_fc.size()))
    print('out shape:{}'.format(out.size()))