import torch
import warnings
import numpy as np
from torch import nn
from torch import Tensor
from torchvision.ops.misc import ConvNormActivation
from torchvision._internally_replaced_utils import load_state_dict_from_url
from torchvision.models._utils import _make_divisible
from typing import Callable, Any, Optional, List
from utils.utils import load_weights_from_state_dict, fuse_conv_bn

__all__ = ['LA_cnn']

# torch.autograd.set_detect_anomaly(True)
model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


# necessary for backwards compatibility
class _DeprecatedConvBNAct(ConvNormActivation):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "The ConvBNReLU/ConvBNActivation classes are deprecated and will be removed in future versions. "
            "Use torchvision.ops.misc.ConvNormActivation instead.", FutureWarning)
        if kwargs.get("norm_layer", None) is None:
            kwargs["norm_layer"] = nn.BatchNorm2d
        if kwargs.get("activation_layer", None) is None:
            kwargs["activation_layer"] = nn.ReLU6
        super().__init__(*args, **kwargs)


ConvBNReLU = _DeprecatedConvBNAct
ConvBNActivation = _DeprecatedConvBNAct

class StochasticPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 这是LA_CNN的做法
        # 对x做限制和加一个eps，都是为了防止x_avg变为nan
        x = torch.clamp(x, 0, 1)
        eps = 1e-8
        x_avg = x / (torch.sum(x, dim=(2, 3), keepdim=True) + eps)
        if torch.isnan(x_avg).any():
            print("nan found in stochastic pooling")
        # 返回输入与归一化后的输入的元素乘积
        # 不要把注释写在return里面啊，不然输出里面包含了注释！！！
        return x * x_avg


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sto_pool = StochasticPooling()
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        sto_out = self.fc(self.sto_pool(x))
        # if torch.isnan(sto_out).any():
        #     print("nan found")
        out = avg_out + max_out + sto_out
        # # 如果out为nan，则报警
        # if torch.isnan(out).any():
        #     print("nan found in channel attention")
        out = self.random_threshold(out)
        # 如果out为nan，则报警
        # if torch.isnan(out).any():
        #     print("nan——new found in channel attention")
        return out * x

    def random_threshold(self, in_tensor):
        random_tensor = torch.rand_like(in_tensor)
        out_tensor = torch.where(in_tensor > random_tensor, torch.full_like(in_tensor, 1.0), torch.full_like(in_tensor, 1e-6))
        in_tensor = out_tensor * in_tensor
        return in_tensor




class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
        
class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer,
                                             activation_layer=nn.ReLU6))
        layers.extend([
            # dw
            ConvNormActivation(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer,
                               activation_layer=nn.ReLU6),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),

        ])
        layers.append(ChannelAttention(oup))
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
    
    def switch_to_deploy(self):
        if len(self.conv) == 4:
            self.conv = nn.Sequential(
                fuse_conv_bn(self.conv[0][0], self.conv[0][1]),
                self.conv[0][2],
                fuse_conv_bn(self.conv[1][0], self.conv[1][1]),
                self.conv[1][2],
                fuse_conv_bn(self.conv[2], self.conv[3]),
            )
        else:
            self.conv = nn.Sequential(
                fuse_conv_bn(self.conv[0][0], self.conv[0][1]),
                self.conv[0][2],
                fuse_conv_bn(self.conv[1], self.conv[2]),
            )

class LA_CNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(LA_CNN, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [ConvNormActivation(3, input_channel, stride=2, norm_layer=norm_layer,
                                                        activation_layer=nn.ReLU6)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(ConvNormActivation(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer,
                                           activation_layer=nn.ReLU6))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor, need_fea=False) -> Tensor:
        if need_fea:
            features, features_fc = self.forward_features(x, need_fea)
            x = self.classifier(features_fc)
            return features, features_fc, x
        else:
            x = self.forward_features(x)
            x = self.classifier(x)
            return x

    def forward(self, x: Tensor, need_fea=False) -> Tensor:
        return self._forward_impl(x, need_fea)
    
    def forward_features(self, x, need_fea=False):
        if need_fea:
            input_size = x.size(2)
            scale = [4, 8, 16, 32]
            features = [None, None, None, None]
            for idx, layer in enumerate(self.features):
                x = layer(x)
                if input_size // x.size(2) in scale:
                    features[scale.index(input_size // x.size(2))] = x
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            return features, x
        else:
            x = self.features(x)
            # Cannot use "squeeze" as batch-size can be 1
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            return x
    
    # 定义 cam_layer 方法
    # 返回模型的最后一层，这是进行类激活映射（CAM）的目标层 最后一层通常包含了模型对输入数据的最终理解。
    def cam_layer(self):
        return self.features[-1]
    
    def switch_to_deploy(self):
        self.features[0] = nn.Sequential(
            fuse_conv_bn(self.features[0][0], self.features[0][1]),
            self.features[0][2]
        )
        self.features[-1] = nn.Sequential(
            fuse_conv_bn(self.features[-1][0], self.features[-1][1]),
            self.features[-1][2]
        )


def LA_cnn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> LA_CNN:
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = LA_CNN(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model = load_weights_from_state_dict(model, state_dict)
    return model

if __name__ == '__main__':
    inputs = torch.rand((1, 3, 224, 224))
    model = LA_cnn(pretrained=True)
    model.eval()
    out = model(inputs)
    print('out shape:{}'.format(out.size()))
    feas, fea_fc, out = model(inputs, True)
    for idx, fea in enumerate(feas):
        print('feature {} shape:{}'.format(idx + 1, fea.size()))
    print('fc shape:{}'.format(fea_fc.size()))
    print('out shape:{}'.format(out.size()))