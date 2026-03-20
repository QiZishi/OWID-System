import torch
from torch import nn
from torch import Tensor

class LeNet_CNN(nn.Module):
    def __init__(self):
        super(LeNet_CNN, self).__init__()
        self.features = nn.Sequential(
            # 采用same策略，使得输入和输出的尺寸相同
            nn.Conv2d(in_channels=1,out_channels=20,kernel_size=10,stride=1,padding=10//2),
            nn.ReLU(),
            # 采用LRN层
            nn.LocalResponseNorm(5),
        )
        self.pool = nn.MaxPool2d(kernel_size=6,stride=3,padding=0)
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Sequential(
            nn.Linear(20*32*32, 12),
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
        x = self.features(x)
         # 当need_fea为True的时候,我们需要遍历所有的features.
        if need_fea:
            features = x
            x = self.pool(x)
            x = self.dropout(x)
            features_fc = torch.flatten(x, 1) # 输入全连接层前的特征
            return features, features_fc
        # 当need_fea为False的时候,我们就不需要遍历features.
        # 简单来说就是直接返回输入全连接层前的特征即可.
        else:
            x = self.pool(x)
            x = self.dropout(x)
            x = torch.flatten(x, 1)
            return x

    # 返回特征层最后一个block就可以.主要是做热力图可视化,当然也可以直接返回整个features.
    def cam_layer(self):
        return self.features[-1] # 返回最后一个卷积层



# 编写一个测试主函数
if __name__ == '__main__':
    # 创建模型实例
    model = LeNet_CNN()
    # 创建一个随机的100*100的单通道图像
    input = torch.randn(1, 1, 100, 100)
    # 将图像通过模型
    output = model(input)
    # 打印输出
    print(output)
    # 检查输出的大小是否为(1, 12)，因为我们有12个类别
    assert output.size() == (1, 12)