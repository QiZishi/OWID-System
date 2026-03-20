import torch
from torch import nn
from torch import Tensor

class Light_CNN(nn.Module):
    def __init__(self):
        super(Light_CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=5,stride=2,padding=2),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool2d(kernel_size=3,stride=1,padding=0)
        self.classifier = nn.Sequential(
            nn.Linear(320000, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 12),
            # nn.Softmax()
        )
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
            return features, features_fc, self.classifier(features_fc)
         # 当need_fea为False时,直接返回最后网络的输出.
        else:
            x = self.forward_features(x)
            x = self.classifier(x)
        return x
    def forward_features(self, x, need_fea=False):
        x = self.conv1(x)
        input = self.pool(x)
        layers = [self.conv2, self.pool, self.conv3, self.pool]
         # 当need_fea为True的时候,我们需要遍历所有的features.
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
            x = torch.flatten(x, 1)
            return x

    # 返回特征层最后一个block就可以.主要是做热力图可视化,当然也可以直接返回整个features.
    def cam_layer(self):
        return self.conv3 # 返回最后一个卷积层



# 编写一个测试主函数
if __name__ == '__main__':
    # 创建模型实例
    model = Light_CNN()
    # 创建一个随机的320*240的单通道图像
    input = torch.randn(1, 1, 112, 112)
    # 将图像通过模型
    output = model(input)
    # 打印输出
    print(output)
    # 检查输出的大小是否为(1, 12)，因为我们有12个类别
    assert output.size() == (1, 12)