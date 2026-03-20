import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torchvision._internally_replaced_utils import load_state_dict_from_url
from typing import Callable, Any, List
from utils.utils import load_weights_from_state_dict, fuse_conv_bn

__all__ = [
    'DCPM_CNN'
]

model_urls = {
    'shufflenetv2_x1.0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',

}



class DilatedConvolutionResidualModule(nn.Module):
    # DCR模块，包含两个3x3空洞卷积层和一个残差连接
    # 输入通道数in_channels，输出通道数out_channels
    def __init__(self, in_channels, out_channels):
        super(DilatedConvolutionResidualModule, self).__init__()
        # 第一层3x3空洞卷积，空洞率为2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.relu = nn.ReLU()
        # 第二层3x3空洞卷积，空洞率为5
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=5, dilation=5)
        self.bn = nn.BatchNorm2d(out_channels)
        # 如果输入和输出通道数不一致，则添加一个1x1卷积层用于调整通道数
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn(out)
        out += self.shortcut(residual)  # 加上残差连接
        out = self.relu(out)
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

        # 空洞率为2，padding为2，从而保持尺寸，副路，输入为输入通道一半（划分）
        self.branch1 = nn.Sequential(
                self.depthwise_conv(inp//2, inp//2, kernel_size=3, stride=self.stride, padding=2,dilation = 2),  # 深度卷积
                nn.BatchNorm2d(inp//2),  # 批量归一化
                nn.Conv2d(inp//2, branch_features, kernel_size=1, stride=1, padding=0, bias=False),  # 1x1卷积
                nn.BatchNorm2d(branch_features),  # 批量归一化
                nn.ReLU(inplace=True),  # ReLU激活函数
            )
        # 空洞率为1，主路，输入为输入通道一半（划分）
        self.branch2 = nn.Sequential(
            nn.Conv2d(inp //2,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1,dilation = 1),
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
        bias: bool = False,
        dilation: int = 2
    ) -> nn.Conv2d:
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i,dilation=dilation)

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=1)
        out = torch.cat((self.branch1(x1), self.branch2(x2)), dim=1)
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
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 添加DCR,论文里是连续两层
        self.DCR = nn.Sequential(
            DilatedConvolutionResidualModule(24, 24),
            DilatedConvolutionResidualModule(24, 24)
            )
        

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
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(output_channels, num_classes)

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
        x = self.maxpool(x)
        x = self.DCR(x) # 添加DCR
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




def DCPM_CNN(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ShuffleNetV2:
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # 根据论文，需要改成[2,4,2]的形式，以适应层数
    return _shufflenetv2('shufflenetv2_x1.0', pretrained, progress,
                         [2, 4, 2], [24, 116, 232, 464, 1024], **kwargs)




if __name__ == '__main__':
    inputs = torch.rand((1, 3, 224, 224))
    model = DCPM_CNN(pretrained=True)
    model.eval()
    out = model(inputs)
    print('out shape:{}'.format(out.size()))
    feas, fea_fc, out = model(inputs, True)
    for idx, fea in enumerate(feas):
        print('feature {} shape:{}'.format(idx + 1, fea.size()))
    print('fc shape:{}'.format(fea_fc.size()))
    print('out shape:{}'.format(out.size()))