import torch
from torch import nn
from torch import Tensor

class OilNet_2D(nn.Module):
    def __init__(self):
        super(OilNet_2D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=8,kernel_size=3,stride=1,padding=2),
            nn.Tanh(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=2),
            nn.Tanh(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=2),
            nn.Tanh(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=16,kernel_size=3,stride=1,padding=2),
            nn.Tanh(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=8,kernel_size=3,stride=1,padding=2),
            nn.Tanh(),
        )
        # 如果stride为none，那么stride默认为kernel_size
        self.pool = nn.MaxPool2d(kernel_size=2,stride=1,padding=0)
        # 8是最后一个卷积层的输出通道数，325*245是第五层的输出图片大小。
        # 12是最后的输出类别数
        self.fc = nn.Linear(8*325*245, 12)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: Tensor, need_fea=False) -> Tensor:
        if need_fea:
            features, features_fc = self.forward_features(x, need_fea)
            return features, features_fc, self.fc(features_fc)
         # 当need_fea为False时,直接返回最后网络的输出.
        else:
            x = self.forward_features(x)
            x = self.fc(x)
        return x
    def forward_features(self, x, need_fea=False):
        x = self.conv1(x)
        input = self.pool(x)
        layers = [self.conv2, self.pool, self.conv3, self.pool, self.conv4, self.pool, self.conv5, self.pool]
         # 当need_fea为True的时候,我们需要遍历所有的features.要的是卷积层输出，不是池化层输出！
        if need_fea:
            features = [x]
            x = input
            for layer in layers:
                x = layer(x)
                if isinstance(layer, torch.nn.modules.conv.Conv2d):
                    features.append(x)
            features_fc = torch.flatten(x, 1) # 输入全连接层前的特征
            return features, features_fc
        # 当need_fea为False的时候,我们就不需要遍历features.
        # 简单来说就是直接返回输入全连接层前的特征即可.
        else:
            x = input
            for layer in layers:
                x = layer(x)
            x = torch.flatten(x, 1)  # 输入全连接层前的特征
            return x

    # 返回特征层最后一个block就可以.主要是做热力图可视化,当然也可以直接返回整个features.
    def cam_layer(self):
        return self.conv5 # 返回最后一层



# 编写一个测试主函数
if __name__ == '__main__':
    # 创建模型实例
    model = OilNet_2D()
    # 创建一个随机的320*240的单通道图像
    input = torch.randn(1, 1, 320, 240)
    # 将图像通过模型
    output = model(input)
    # 打印输出
    print(output)
    # 检查输出的大小是否为(1, 12)，因为我们有12个类别
    assert output.size() == (1, 12)